# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Viewer serving artifacts derived from a single evaluation (contracts.md §4, §6).

The hosted benchmark publishes, alongside the score parquet and summary, the artifacts the
interactive viewer streams: the Class-4 match-up parquet in its row-group serving layout, the
per-dataset eddy detection census, the multiscale field pyramid with its viewer manifest, and
the year-mode error-geography and per-start RMSD JSON. This module produces those same
artifacts from the objects a local ``evaluate`` already has in hand, reusing the numerical core
(``oceanbench.runner.matchups``, ``oceanbench.runner.realism``, ``oceanbench.core.eddies`` and
``oceanbench.pyramids``); no new science lives here, only the serving-layout shaping.
"""

from dataclasses import dataclass, field
import datetime
import json
import os
from pathlib import Path
import subprocess

import numpy
import pandas
import pyarrow
import pyarrow.compute
import pyarrow.parquet
import xarray

from oceanbench.core import eddies as eddies_core
from oceanbench.core.schema_validation import load_schema
from oceanbench.core.version import __version__ as OCEANBENCH_VERSION
from oceanbench.publish.aggregate import DEFAULT_SEED as _LEAD_CURVE_BOOTSTRAP_SEED, _confidence_interval
from oceanbench.pyramids import build_pyramid, viewer_layers
from oceanbench.runner import realism

MATCHUP_PARQUET_FILENAME = "class4-matchups.parquet"
EDDY_CENSUS_FILENAME = "eddies.json"
YEAR_ERROR_GEOGRAPHY_FILENAME = "year-error-geography.json"
YEAR_RMSD_BY_START_FILENAME = "year-rmsd-by-start.json"
RMSD_BY_DEPTH_FILENAME = "rmsd-by-depth.json"

_RMSD_BY_DEPTH_SCHEMA_VERSION = 1

MAXIMUM_ROW_GROUP_ROWS = 200_000

_MATCHUP_SOURCE_COLUMNS = [
    "variable",
    "depth_bin",
    "lead_day",
    "start_date",
    "latitude",
    "longitude",
    "observation_value",
    "model_value",
]
_MATCHUP_TARGET_SCHEMA = pyarrow.schema(
    [
        ("variable", pyarrow.string()),
        ("depth_bin", pyarrow.string()),
        ("lead_day", pyarrow.int16()),
        ("start_date", pyarrow.string()),
        ("latitude", pyarrow.float32()),
        ("longitude", pyarrow.float32()),
        ("observation_value", pyarrow.float32()),
        ("model_value", pyarrow.float32()),
        ("abs_error", pyarrow.float32()),
    ]
)

_EDDY_CENSUS_SCHEMA_VERSION = "1"
_EDDY_CENSUS_LEAD_DAYS = (1, 5, 10)
_SEA_SURFACE_HEIGHT_VARIABLE = "sea_surface_height_above_geoid"

# Year-mode super-observation grids: global at 2 degrees, IBI at a quarter degree. The bin
# origins and cell counts match the published year artifacts exactly.
_YEAR_GRIDS = {
    "global": {"lat0": -90, "dlat": 2, "nlat": 90, "lon0": -180, "dlon": 2, "nlon": 180},
    "ibi": {"lat0": 26.0, "dlat": 0.25, "nlat": 121, "lon0": -19.25, "dlon": 0.25, "nlon": 98},
}
_YEAR_LEAD_DAY_COUNT = 10
_YEAR_TARGETS = [
    ("sea_surface_height_above_geoid", "surface", "SSH"),
    ("sea_water_potential_temperature", "0-5m", "T"),
    ("sea_water_salinity", "0-5m", "S"),
    ("eastward_sea_water_velocity", "15m", "u"),
    ("northward_sea_water_velocity", "15m", "v"),
]
_YEAR_GEOGRAPHY_DECIMALS = {"SSH": 4, "T": 3, "S": 4, "u": 4, "v": 4}

# Uncertainty on the per-start point estimates. The RMSD interval reuses the lead-curve method
# (seeded percentile bootstrap, aggregate.py) — same seed and 95% percentile interval — but
# resamples the match-ups of that single start rather than the whole-year start axis, so the band
# reflects the observation spread inside that start, consistent with the pooled per-start
# reduction. Above ``_YEAR_RMSD_CI_EXACT_MAXIMUM`` match-ups the resampling switches to an
# equal-count quantile-binned multinomial bootstrap of the mean squared error (statistically
# equivalent at that n, cost independent of n). The bias interval is the analytic normal interval
# mean ± 1.96 · std(model − obs) / sqrt(n).
_YEAR_RMSD_CI_BOOTSTRAP_DRAWS = 200
_YEAR_RMSD_CI_EXACT_MAXIMUM = 10_000
_YEAR_RMSD_CI_BINS = 1_000
_YEAR_CI_SEED = _LEAD_CURVE_BOOTSTRAP_SEED
_YEAR_CI_CONFIDENCE = 0.95
_BIAS_CI_Z = 1.96

PROVENANCE_KEY = "provenance"
_MATCHUP_PROVENANCE_METADATA_KEY = b"oceanbench_provenance"


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except Exception:  # noqa: BLE001 - provenance is best-effort, absence of git is not an error
        return None
    commit = completed.stdout.strip()
    return commit or None


def provenance_block(*, source: str, parameters: dict | None = None) -> dict:
    """Provenance stamp carried by every viewer artifact.

    Records the emitting library version, the git commit when the source tree is a checkout, the
    UTC generation timestamp and the source dataset identifier the artifact derives from. Any
    relevant generating parameters are carried verbatim under ``parameters``.
    """
    block = {
        "oceanbench_version": OCEANBENCH_VERSION,
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source": source,
    }
    if parameters is not None:
        block["parameters"] = parameters
    return block


@dataclass(frozen=True)
class ViewerArtifactsResult:
    matchup_parquet_path: str | None = None
    eddy_census_path: str | None = None
    pyramid_zarr_path: str | None = None
    pyramid_manifest_path: str | None = None
    column_store_zarr_path: str | None = None
    year_error_geography_path: str | None = None
    year_rmsd_by_start_path: str | None = None
    rmsd_by_depth_path: str | None = None
    class4_bias_records: list[dict] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


def class4_bias_per_start_records(matchups, *, context) -> list[dict]:
    """Per-start signed Class-4 mean error (model minus observation) records from the match-ups.

    One record per ``(variable, depth_bin, lead_day, start_date)`` carrying the signed mean error
    ``mean(model_value - observation_value)`` and its observation count ``n``, in the same
    long-format shape as the Class-4 RMSD records so the existing aggregation derives a mean bias
    (closer to zero is better) and bootstrap interval per variable/depth/lead.
    """
    from oceanbench.core.dataset_utils import VARIABLE_METADATA
    from oceanbench.runner.records import METRIC_CLASS4_BIAS

    valid = matchups.dropna(subset=["model_value", "observation_value"])
    if valid.empty:
        return []
    grouped = (
        valid.assign(signed_error=valid["model_value"] - valid["observation_value"])
        .groupby(["variable", "depth_bin", "lead_day", "start_date"], as_index=False)
        .agg(bias=("signed_error", "mean"), count=("signed_error", "size"))
    )
    return [
        {
            "challenger": context.challenger,
            "challenger_version": context.challenger_version,
            "year": context.year,
            "region": context.region,
            "metric": METRIC_CLASS4_BIAS,
            "reference": "observations",
            "variable": str(row.variable),
            "depth": str(row.depth_bin),
            "lead_day": int(row.lead_day),
            "start_date": numpy.datetime64(row.start_date, "D").astype("datetime64[ns]"),
            "band": None,
            "polarity": None,
            "value": float(row.bias),
            "unit": VARIABLE_METADATA[str(row.variable)][1],
            "n": int(row.count),
            "oceanbench_version": OCEANBENCH_VERSION,
        }
        for row in grouped.itertuples(index=False)
    ]


def _project_matchups(table: pyarrow.Table) -> pyarrow.Table:
    start_date = table.column("start_date")
    if pyarrow.types.is_timestamp(start_date.type):
        start_date = pyarrow.compute.strftime(start_date, format="%Y-%m-%d")
    else:
        start_date = pyarrow.compute.cast(start_date, pyarrow.string())
    model_value = pyarrow.compute.cast(table.column("model_value"), pyarrow.float32())
    observation_value = pyarrow.compute.cast(table.column("observation_value"), pyarrow.float32())
    absolute_error = pyarrow.compute.cast(
        pyarrow.compute.abs(pyarrow.compute.subtract(model_value, observation_value)), pyarrow.float32()
    )
    return pyarrow.table(
        {
            "variable": pyarrow.compute.cast(table.column("variable"), pyarrow.string()),
            "depth_bin": pyarrow.compute.cast(table.column("depth_bin"), pyarrow.string()),
            "lead_day": pyarrow.compute.cast(table.column("lead_day"), pyarrow.int16()),
            "start_date": start_date,
            "latitude": pyarrow.compute.cast(table.column("latitude"), pyarrow.float32()),
            "longitude": pyarrow.compute.cast(table.column("longitude"), pyarrow.float32()),
            "observation_value": observation_value,
            "model_value": model_value,
            "abs_error": absolute_error,
        },
        schema=_MATCHUP_TARGET_SCHEMA,
    )


_MATCHUP_GROUP_COLUMNS = ("start_date", "lead_day", "variable", "depth_bin")


def _group_boundaries(columns: list[numpy.ndarray]) -> numpy.ndarray:
    row_count = len(columns[0])
    changed = numpy.zeros(row_count, dtype=bool)
    changed[0] = True
    for column in columns:
        changed[1:] |= column[1:] != column[:-1]
    return numpy.flatnonzero(changed)


def _projected_sorted_partition(matchups) -> pyarrow.Table:
    source_table = pyarrow.Table.from_pandas(
        matchups[[column for column in _MATCHUP_SOURCE_COLUMNS if column in matchups.columns]],
        preserve_index=False,
    )
    projected = _project_matchups(source_table)
    order = pyarrow.compute.sort_indices(
        projected, sort_keys=[(column, "ascending") for column in _MATCHUP_GROUP_COLUMNS]
    )
    return projected.take(order)


def _write_matchup_row_groups(writer: "pyarrow.parquet.ParquetWriter", projected: pyarrow.Table) -> None:
    row_count = projected.num_rows
    if not row_count:
        return
    group_columns = [projected.column(column).to_numpy(zero_copy_only=False) for column in _MATCHUP_GROUP_COLUMNS]
    boundaries = _group_boundaries(group_columns)
    ends = numpy.append(boundaries[1:], row_count)
    for group_start, group_end in zip(boundaries, ends):
        for chunk_start in range(int(group_start), int(group_end), MAXIMUM_ROW_GROUP_ROWS):
            chunk_end = min(chunk_start + MAXIMUM_ROW_GROUP_ROWS, int(group_end))
            writer.write_table(projected.slice(chunk_start, chunk_end - chunk_start))


def _matchup_provenance_metadata(output_path: str, source: str | None) -> dict:
    provenance = provenance_block(source=source if source is not None else os.path.basename(output_path))
    return {_MATCHUP_PROVENANCE_METADATA_KEY: json.dumps(provenance).encode("utf-8")}


def write_matchup_parquet(matchups, output_path: str, *, source: str | None = None) -> str:
    """Write the Class-4 match-ups to the viewer serving parquet and validate the layout.

    The match-up dataframe (``oceanbench.runner.matchups.class4_matchups``) is projected to the
    nine served columns, sorted by ``(start_date, lead_day, variable, depth_bin)`` and written
    SNAPPY-compressed with one ``(start_date, lead_day, variable, depth_bin)`` group per row group
    so a single-variable view never fetches a row group straddling a variable boundary (a group
    spanning more than ``MAXIMUM_ROW_GROUP_ROWS`` rows is split across consecutive groups). The
    written file is then re-opened and validated by :func:`verify_matchup_parquet`.
    """
    provenance_metadata = _matchup_provenance_metadata(output_path, source)
    projected = _projected_sorted_partition(matchups).replace_schema_metadata(provenance_metadata)
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    writer = pyarrow.parquet.ParquetWriter(output_path, projected.schema, compression="snappy")
    _write_matchup_row_groups(writer, projected)
    writer.close()
    verify_matchup_parquet(output_path)
    return output_path


def write_matchup_parquet_streamed(
    start_partitions, output_path: str, *, source: str | None = None, on_partition=None
) -> str:
    """Stream per-start match-up frames to the serving parquet without materialising the year.

    ``start_partitions`` yields one match-up dataframe per forecast start in ascending
    ``start_date`` order (as :func:`oceanbench.runner.matchups.iter_class4_matchups_by_start`
    produces). Because ``start_date`` is the leading sort key and the starts are disjoint and
    ascending, sorting each start on its own ``(start_date, lead_day, variable, depth_bin)`` keys
    and appending its row groups yields a file byte-for-byte equivalent in row content and layout to
    :func:`write_matchup_parquet` over the concatenated frame — but peak memory is one start, not the
    whole year, which also keeps every written array well under Arrow's 2 GiB single-array limit.
    ``on_partition`` is invoked with each raw (pre-projection) frame for callers that also aggregate
    from the match-ups (e.g. per-start bias records).
    """
    provenance_metadata = _matchup_provenance_metadata(output_path, source)
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    writer = None
    try:
        for partition in start_partitions:
            if on_partition is not None:
                on_partition(partition)
            if partition is None or len(partition) == 0:
                continue
            projected = _projected_sorted_partition(partition).replace_schema_metadata(provenance_metadata)
            if writer is None:
                writer = pyarrow.parquet.ParquetWriter(output_path, projected.schema, compression="snappy")
            _write_matchup_row_groups(writer, projected)
        if writer is None:
            writer = pyarrow.parquet.ParquetWriter(
                output_path, _MATCHUP_TARGET_SCHEMA.with_metadata(provenance_metadata), compression="snappy"
            )
        writer.close()
    except BaseException:
        # A worker (or the projection) may raise mid-stream; the half-written parquet on disk is
        # corrupt and must not be mistaken for a complete artifact, so delete it before re-raising.
        if writer is not None:
            try:
                writer.close()
            except Exception:  # noqa: BLE001 - the file is being discarded anyway
                pass
        Path(output_path).unlink(missing_ok=True)
        raise
    verify_matchup_parquet(output_path)
    return output_path


def verify_matchup_parquet(output_path: str) -> dict:
    """Validate a match-up parquet's serving layout, raising ``ValueError`` on any violation.

    Every row group must hold exactly one ``(start_date, lead_day, variable, depth_bin)`` group
    (column statistics ``min == max`` on all four), carry statistics, stay within
    ``MAXIMUM_ROW_GROUP_ROWS`` rows, and the groups must appear in ascending
    ``(start_date, lead_day, variable, depth_bin)`` order.
    """
    parquet_file = pyarrow.parquet.ParquetFile(output_path)
    metadata = parquet_file.metadata
    names = [parquet_file.schema_arrow.field(index).name for index in range(len(parquet_file.schema_arrow))]
    if names != _MATCHUP_TARGET_SCHEMA.names:
        raise ValueError(f"match-up parquet schema {names} does not match the contract {_MATCHUP_TARGET_SCHEMA.names}")
    group_indices = [names.index(column) for column in _MATCHUP_GROUP_COLUMNS]
    previous_key = None
    for group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(group_index)
        statistics = [row_group.column(column_index).statistics for column_index in group_indices]
        if any(column_statistics is None for column_statistics in statistics):
            raise ValueError(f"row group {group_index} is missing column statistics")
        if any(column_statistics.min != column_statistics.max for column_statistics in statistics):
            raise ValueError(
                f"row group {group_index} mixes more than one (start_date, lead_day, variable, depth_bin) group"
            )
        if row_group.num_rows > MAXIMUM_ROW_GROUP_ROWS:
            raise ValueError(f"row group {group_index} has {row_group.num_rows} rows above the cap")
        key = tuple(column_statistics.min for column_statistics in statistics)
        if previous_key is not None and key < previous_key:
            raise ValueError(f"row group {group_index} is out of (start_date, lead_day, variable, depth_bin) order")
        previous_key = key
    return {"row_groups": metadata.num_row_groups, "rows": metadata.num_rows}


def _clamp(value, low: float, high: float):
    if isinstance(value, list):
        return [min(high, max(low, element)) for element in value]
    return min(high, max(low, value))


def _clamp_eddy(eddy: dict) -> dict:
    return {
        **eddy,
        "longitude": _clamp(eddy["longitude"], -180.0, 180.0),
        "latitude": _clamp(eddy["latitude"], -90.0, 90.0),
        "contour_longitude": _clamp(eddy["contour_longitude"], -180.0, 180.0),
        "contour_latitude": _clamp(eddy["contour_latitude"], -90.0, 90.0),
    }


def _eddy_frame(dataset: xarray.Dataset, detections, contours, lead_day: int) -> dict:
    import jsonschema

    eddy_schema = load_schema("eddies")["$defs"]["eddy"]
    detection_indices = realism._lead_detection_indices(detections, lead_day - 1)
    eddies = []
    for detection_index in detection_indices:
        eddy = _clamp_eddy(realism._eddy_dict(detection_index, detections, contours))
        jsonschema.validate(instance=eddy, schema=eddy_schema)
        eddies.append(eddy)
    return {"lead_day": lead_day, "detections": eddies}


def dataset_eddy_census(
    dataset: xarray.Dataset,
    *,
    dataset_slug: str,
    lead_days: tuple[int, ...] = _EDDY_CENSUS_LEAD_DAYS,
    start_index: int = 0,
    apply_contour_filtering: bool = eddies_core.DEFAULT_APPLY_CONTOUR_FILTERING,
) -> dict:
    """Build a dataset's own mesoscale-eddy detection census (census-only, no reference side).

    One frame per lead day, each listing that dataset's own detections (centre, polarity and
    point-limited contour) with coordinates clamped to the served ranges and validated against the
    eddies schema ``eddy`` definition. The km-based literature detection parameters are stamped,
    including ``apply_contour_filtering`` and the emitting ``oceanbench_version``.
    """
    lead_day_indices = [lead_day - 1 for lead_day in lead_days]
    detections = eddies_core.detect_mesoscale_eddies(
        dataset, first_day_index=start_index, lead_day_indices=lead_day_indices
    )
    if apply_contour_filtering:
        detections = realism._contour_filtered_detections(dataset, detections, start_index)
    contours = realism._contours(dataset, detections, start_index)
    parameters = {
        **eddies_core.default_eddy_detection_parameters(),
        "apply_contour_filtering": apply_contour_filtering,
        "oceanbench_version": OCEANBENCH_VERSION,
    }
    return {
        "kind": "eddy-census",
        "schema_version": _EDDY_CENSUS_SCHEMA_VERSION,
        "variable": _SEA_SURFACE_HEIGHT_VARIABLE,
        "dataset": dataset_slug,
        "parameters": parameters,
        PROVENANCE_KEY: provenance_block(source=dataset_slug, parameters=parameters),
        "frames": [_eddy_frame(dataset, detections, contours, lead_day) for lead_day in lead_days],
    }


def _lead_census_filename(lead_day: int) -> str:
    return f"eddies-lead-{lead_day}.json"


def write_eddy_census(dataset: xarray.Dataset, output_path: str, *, dataset_slug: str, **census_options) -> str:
    """Write a dataset's eddy detection census as one JSON file per lead day, plus a small index.

    Only one lead day is viewed at a time, so each frame is written to its own
    ``eddies-lead-<N>.json`` (payload metadata plus that lead's ``frame``) next to ``output_path``,
    and ``output_path`` itself receives a tiny index listing the per-lead files. Returns the index
    path.
    """
    census = dataset_eddy_census(dataset, dataset_slug=dataset_slug, **census_options)
    directory = Path(os.path.dirname(output_path) or ".")
    directory.mkdir(parents=True, exist_ok=True)
    metadata_keys = ("kind", "schema_version", "variable", "dataset", "parameters", PROVENANCE_KEY)
    metadata = {key: census[key] for key in metadata_keys}

    lead_entries = []
    for frame in census["frames"]:
        lead_filename = _lead_census_filename(frame["lead_day"])
        (directory / lead_filename).write_text(
            json.dumps({**metadata, "frame": frame}, sort_keys=True, indent=2), encoding="utf-8"
        )
        lead_entries.append({"lead_day": frame["lead_day"], "file": lead_filename})

    index = {**metadata, "leads": lead_entries}
    Path(output_path).write_text(json.dumps(index, sort_keys=True, indent=2), encoding="utf-8")
    return output_path


def _year_grid_for_region(region: str) -> dict:
    if region not in _YEAR_GRIDS:
        raise ValueError(f"no year-mode super-observation grid defined for region {region!r}")
    return _YEAR_GRIDS[region]


def _grid_cells(latitude: numpy.ndarray, longitude: numpy.ndarray, grid: dict) -> tuple[numpy.ndarray, numpy.ndarray]:
    longitude_span = grid["nlon"] * grid["dlon"]
    wrapped_longitude = ((longitude - grid["lon0"]) % longitude_span) + grid["lon0"]
    latitude_bin = numpy.floor((latitude - grid["lat0"]) / grid["dlat"]).astype(numpy.int64)
    longitude_bin = numpy.floor((wrapped_longitude - grid["lon0"]) / grid["dlon"]).astype(numpy.int64)
    valid = (latitude_bin >= 0) & (latitude_bin < grid["nlat"]) & (longitude_bin >= 0) & (longitude_bin < grid["nlon"])
    return latitude_bin * grid["nlon"] + longitude_bin, valid


def _cell_bias_standard_error(square_sum: float, signed_sum: float, count: int, decimals: int) -> float | None:
    """Per-cell analytic standard error of the bias: std(model − obs) / sqrt(n), None below n = 2."""
    if count < 2:
        return None
    mean = signed_sum / count
    variance = max((square_sum / count) - mean * mean, 0.0)
    return round(float(numpy.sqrt(variance / count)), decimals)


def _bootstrap_rmsd_ci(
    squared_error: numpy.ndarray, generator: numpy.random.Generator
) -> tuple[float | None, float | None]:
    """95% percentile-bootstrap CI for one start's pooled RMSD, resampling its match-ups.

    Same method and confidence as the lead-curve bootstrap (seeded percentile interval); the
    resampled unit here is the individual match-up of that start, matching the pooled per-start
    reduction ``sqrt(mean(error ** 2))``. Small starts resample match-ups exactly; large starts
    (above ``_YEAR_RMSD_CI_EXACT_MAXIMUM``) use an equal-count quantile-binned multinomial
    bootstrap of the mean squared error — the resample weight of each of the
    ``_YEAR_RMSD_CI_BINS`` sorted equal-count bins is drawn from Multinomial(n, count_bin / n),
    which reproduces the bootstrap distribution of the mean up to the (negligible at that n)
    within-bin variance, at a cost independent of n.
    """
    n = squared_error.size
    if n < 2:
        return (None, None)
    if n <= _YEAR_RMSD_CI_EXACT_MAXIMUM:
        draws = generator.integers(0, n, size=(_YEAR_RMSD_CI_BOOTSTRAP_DRAWS, n))
        bootstrap_mean = squared_error[draws].mean(axis=1)
    else:
        sorted_squares = numpy.sort(squared_error)
        bins = numpy.array_split(sorted_squares, _YEAR_RMSD_CI_BINS)
        bin_means = numpy.array([bin_values.mean() for bin_values in bins])
        bin_counts = numpy.array([bin_values.size for bin_values in bins])
        counts = generator.multinomial(n, bin_counts / n, size=_YEAR_RMSD_CI_BOOTSTRAP_DRAWS)
        bootstrap_mean = counts @ bin_means / n
    bootstrap_rmsd = numpy.sqrt(numpy.maximum(bootstrap_mean, 0.0))
    low, high = _confidence_interval(bootstrap_rmsd, _YEAR_CI_CONFIDENCE)
    if not (numpy.isfinite(low) and numpy.isfinite(high)):
        return (None, None)
    return (round(float(low), 6), round(float(high), 6))


def _analytic_bias_ci(signed_error: numpy.ndarray, bias: float) -> tuple[float | None, float | None]:
    """Analytic normal CI for one start's pooled bias: mean ± 1.96 · std(model − obs) / sqrt(n)."""
    n = signed_error.size
    if n < 2:
        return (None, None)
    half_width = _BIAS_CI_Z * float(signed_error.std()) / numpy.sqrt(n)
    return (round(bias - half_width, 6), round(bias + half_width, 6))


def _write_year_artifacts(
    matchup_parquet_path: str,
    region: str,
    geography_path: str,
    rmsd_path: str,
    source: str,
) -> None:
    grid = _year_grid_for_region(region)
    cell_count = grid["nlat"] * grid["nlon"]
    variable_count = len(_YEAR_TARGETS)
    target_index = {(variable, depth_bin): index for index, (variable, depth_bin, _) in enumerate(_YEAR_TARGETS)}

    error_sum = numpy.zeros((variable_count, _YEAR_LEAD_DAY_COUNT, cell_count))
    error_square_sum = numpy.zeros((variable_count, _YEAR_LEAD_DAY_COUNT, cell_count))
    bias_sum = numpy.zeros((variable_count, _YEAR_LEAD_DAY_COUNT, cell_count))
    error_count = numpy.zeros((variable_count, _YEAR_LEAD_DAY_COUNT, cell_count), dtype=numpy.int64)
    rmsd_rows = {(variable, lead): [] for variable in range(variable_count) for lead in range(_YEAR_LEAD_DAY_COUNT)}
    start_dates: list[str] = []
    current_start = None
    # Per (variable, lead): the current start's raw signed errors (model - obs), pooled at flush
    # into the per-start RMSD/bias/n and their intervals (contiguity in start is guaranteed by the
    # serving sort order of the match-up parquet).
    start_accumulators: dict[tuple[int, int], list] = {}
    bootstrap_generator = numpy.random.default_rng(_YEAR_CI_SEED)

    def flush(start_date):
        for (variable, lead), chunks in start_accumulators.items():
            signed = numpy.concatenate(chunks)
            if signed.size == 0:
                continue
            squared = signed * signed
            root_mean_square = float(numpy.sqrt(squared.mean()))
            bias = float(signed.mean())
            rmsd_low, rmsd_high = _bootstrap_rmsd_ci(squared, bootstrap_generator)
            bias_low, bias_high = _analytic_bias_ci(signed, bias)
            rmsd_rows[(variable, lead)].append(
                (start_date, root_mean_square, int(signed.size), bias, rmsd_low, rmsd_high, bias_low, bias_high)
            )
        start_accumulators.clear()

    parquet_file = pyarrow.parquet.ParquetFile(matchup_parquet_path)
    columns = [
        "variable",
        "depth_bin",
        "lead_day",
        "latitude",
        "longitude",
        "abs_error",
        "observation_value",
        "model_value",
        "start_date",
    ]
    for group_index in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(group_index, columns=columns)
        lead_day = int(batch["lead_day"][0].as_py())
        if lead_day < 1 or lead_day > _YEAR_LEAD_DAY_COUNT:
            continue
        lead = lead_day - 1
        start_date = batch["start_date"][0].as_py()
        if start_date != current_start:
            if current_start is not None:
                flush(current_start)
            current_start = start_date
            start_dates.append(start_date)

        variable_names = numpy.asarray(batch["variable"].to_pylist())
        depth_names = numpy.asarray(batch["depth_bin"].to_pylist())
        all_latitude = batch["latitude"].to_numpy()
        all_longitude = batch["longitude"].to_numpy()
        all_absolute_error = batch["abs_error"].to_numpy()
        all_observation = batch["observation_value"].to_numpy()
        all_model = batch["model_value"].to_numpy()

        # A row group is pure in (start_date, lead_day) but mixes variables: select each target
        # variable/depth_bin explicitly rather than assuming a homogeneous group.
        for (variable_name, depth_name), variable in target_index.items():
            selection = (variable_names == variable_name) & (depth_names == depth_name)
            if not selection.any():
                continue
            latitude = all_latitude[selection]
            longitude = all_longitude[selection]
            absolute_error = all_absolute_error[selection]
            observation = all_observation[selection]
            model = all_model[selection]
            finite = (
                numpy.isfinite(absolute_error)
                & numpy.isfinite(latitude)
                & numpy.isfinite(longitude)
                & numpy.isfinite(observation)
                & numpy.isfinite(model)
            )
            if not finite.all():
                latitude, longitude = latitude[finite], longitude[finite]
                absolute_error, observation, model = absolute_error[finite], observation[finite], model[finite]
            if latitude.size == 0:
                continue
            # Per-start pooled series: every finite match-up of the start counts, exactly like the
            # official per-start reduction (no grid filtering).
            start_accumulators.setdefault((variable, lead), []).append(model - observation)
            cell, valid = _grid_cells(latitude, longitude, grid)
            cell = cell[valid]
            if cell.size == 0:
                continue
            valid_absolute_error = absolute_error[valid]
            numpy.add.at(error_sum[variable, lead], cell, valid_absolute_error)
            numpy.add.at(error_square_sum[variable, lead], cell, valid_absolute_error * valid_absolute_error)
            numpy.add.at(bias_sum[variable, lead], cell, (model[valid] - observation[valid]))
            numpy.add.at(error_count[variable, lead], cell, 1)
    if current_start is not None:
        flush(current_start)

    geography = {
        "grid": grid,
        "variables": {},
        "meta": {
            "n_starts": len(start_dates),
            "generated_from": source,
            "depth_bin": {short: depth_bin for _, depth_bin, short in _YEAR_TARGETS},
            "aggregation": "time-mean of |obs-model| per cell",
            "bias_standard_error": (
                "per cell std(model-obs)/sqrt(n) under bias_se; n is the shared per-cell match-up "
                "count under n (identical sampling for the |error| and bias rasters)"
            ),
        },
        PROVENANCE_KEY: provenance_block(source=source, parameters={"grid": grid, "region": region}),
    }
    for variable, (_, _, short) in enumerate(_YEAR_TARGETS):
        decimals = _YEAR_GEOGRAPHY_DECIMALS[short]
        leads = {}
        bias_leads = {}
        count_leads = {}
        bias_se_leads = {}
        for lead in range(_YEAR_LEAD_DAY_COUNT):
            count = error_count[variable, lead]
            summed = error_sum[variable, lead]
            square_summed = error_square_sum[variable, lead]
            signed_summed = bias_sum[variable, lead]
            leads[str(lead + 1)] = [
                None if cell_count_value == 0 else round(float(summed[cell] / cell_count_value), decimals)
                for cell, cell_count_value in enumerate(count)
            ]
            bias_leads[str(lead + 1)] = [
                None if cell_count_value == 0 else round(float(signed_summed[cell] / cell_count_value), decimals)
                for cell, cell_count_value in enumerate(count)
            ]
            count_leads[str(lead + 1)] = [int(cell_count_value) for cell_count_value in count]
            bias_se_leads[str(lead + 1)] = [
                _cell_bias_standard_error(square_summed[cell], signed_summed[cell], cell_count_value, decimals)
                for cell, cell_count_value in enumerate(count)
            ]
        # n is shared verbatim by the |error| and bias rasters (identical per-cell sampling); it is
        # emitted once rather than duplicated. bias_se is the per-cell analytic standard error.
        geography["variables"][short] = {
            "leads": leads,
            "bias": bias_leads,
            "n": count_leads,
            "bias_se": bias_se_leads,
        }
    Path(os.path.dirname(geography_path) or ".").mkdir(parents=True, exist_ok=True)
    Path(geography_path).write_text(json.dumps(geography), encoding="utf-8")

    rmsd = {
        "variables": {},
        "meta": {
            "method": "pooled RMSD over all class-4 match-ups per start date (same reduction as official scores)",
            "bias_method": "pooled mean(model - obs) per start date",
            "grid": grid,
            "generated_from": source,
            "ci_method": (
                f"rmsd_ci: seeded percentile bootstrap over the start's match-ups "
                f"({_YEAR_RMSD_CI_BOOTSTRAP_DRAWS} resamples, seed {_YEAR_CI_SEED}, "
                f"{int(_YEAR_CI_CONFIDENCE * 100)}% percentile interval — same method/seed as the "
                f"lead-curve bootstrap, resampling match-ups within the start rather than starts; "
                f"above {_YEAR_RMSD_CI_EXACT_MAXIMUM} match-ups an equal-count quantile-binned "
                f"({_YEAR_RMSD_CI_BINS} bins) multinomial bootstrap of the mean squared error is "
                f"used, statistically equivalent at that n). "
                f"bias_ci: analytic normal interval mean +/- {_BIAS_CI_Z} * std(model - obs) / sqrt(n)"
            ),
        },
        PROVENANCE_KEY: provenance_block(source=source, parameters={"grid": grid, "region": region}),
    }
    for variable, (_, depth_bin, short) in enumerate(_YEAR_TARGETS):
        leads = {}
        for lead in range(_YEAR_LEAD_DAY_COUNT):
            rows = sorted(rmsd_rows[(variable, lead)], key=lambda row: row[0])
            leads[str(lead + 1)] = {
                "dates": [row[0] for row in rows],
                "rmsd": [round(row[1], 6) for row in rows],
                "n": [row[2] for row in rows],
                "bias": [round(row[3], 6) for row in rows],
                "rmsd_ci_low": [row[4] for row in rows],
                "rmsd_ci_high": [row[5] for row in rows],
                "bias_ci_low": [row[6] for row in rows],
                "bias_ci_high": [row[7] for row in rows],
            }
        rmsd["variables"][short] = {"depth_bin": depth_bin, "leads": leads}
    Path(rmsd_path).write_text(json.dumps(rmsd), encoding="utf-8")


def _depth_bin_sort_key(label: str) -> tuple[int, float, str]:
    """Order depth-bin labels surface -> deep.

    Known labels use the canonical ``DEPTH_BIN_DISPLAY_ORDER`` ranking; any other label falls back
    to the numeric value that begins it (so ``"100-300m"`` sorts on 100), and finally on the raw
    string so the order is always total and deterministic.
    """
    from oceanbench.core.dataset_utils import DEPTH_BIN_DISPLAY_ORDER

    if label in DEPTH_BIN_DISPLAY_ORDER:
        return (0, float(DEPTH_BIN_DISPLAY_ORDER[label]), label)
    leading = ""
    for character in label:
        if character.isdigit() or character == ".":
            leading += character
        else:
            break
    return (1, float(leading) if leading else float("inf"), label)


def rmsd_by_depth(matchup_parquet_path: str) -> dict:
    """Per ``(variable, depth_bin, lead_day)`` pooled RMSD, bias and obs count over the whole year.

    Streams the Class-4 match-up parquet once, accumulating over *all* match-ups of the year (every
    forecast start pooled together) the squared error ``(model - obs) ** 2``, the signed error
    ``model - obs`` and the finite-observation count for each ``(variable, depth_bin, lead_day)``
    cell. The pooled RMSD is ``sqrt(mean(squared error))`` — the same observation-pooled reduction
    the official Class-4 score uses — the bias is ``mean(model - obs)`` and ``n`` is the pooled obs
    count. Only 3D variables carrying more than one depth bin are returned (surface-only variables
    such as SSH or the 15 m currents are skipped); depth bins are ordered surface -> deep with the
    exact labels found in the parquet.
    """
    square_sum: dict = {}
    signed_sum: dict = {}
    count: dict = {}
    depth_bins_by_variable: dict[str, set] = {}
    leads_by_variable: dict[str, set] = {}

    parquet_file = pyarrow.parquet.ParquetFile(matchup_parquet_path)
    columns = ["variable", "depth_bin", "lead_day", "observation_value", "model_value", "abs_error"]
    for group_index in range(parquet_file.num_row_groups):
        batch = parquet_file.read_row_group(group_index, columns=columns)
        variable_names = numpy.asarray(batch["variable"].to_pylist())
        depth_names = numpy.asarray(batch["depth_bin"].to_pylist())
        lead_days = batch["lead_day"].to_numpy().astype("int64")
        observation = batch["observation_value"].to_numpy().astype("float64")
        model = batch["model_value"].to_numpy().astype("float64")
        absolute_error = batch["abs_error"].to_numpy().astype("float64")
        finite = numpy.isfinite(absolute_error) & numpy.isfinite(observation) & numpy.isfinite(model)
        if not finite.any():
            continue
        variable_names = variable_names[finite]
        depth_names = depth_names[finite]
        lead_days = lead_days[finite]
        signed = model[finite] - observation[finite]
        squared = absolute_error[finite] ** 2

        frame = pandas.DataFrame(
            {
                "variable": variable_names,
                "depth_bin": depth_names,
                "lead_day": lead_days,
                "squared": squared,
                "signed": signed,
            }
        )
        for (variable_name, depth_name, lead_value), group in frame.groupby(
            ["variable", "depth_bin", "lead_day"], sort=False
        ):
            key = (str(variable_name), str(depth_name), int(lead_value))
            square_sum[key] = square_sum.get(key, 0.0) + float(group["squared"].sum())
            signed_sum[key] = signed_sum.get(key, 0.0) + float(group["signed"].sum())
            count[key] = count.get(key, 0) + int(len(group))
            depth_bins_by_variable.setdefault(str(variable_name), set()).add(str(depth_name))
            leads_by_variable.setdefault(str(variable_name), set()).add(int(lead_value))

    variables: dict = {}
    for variable_name, depth_bin_set in depth_bins_by_variable.items():
        if len(depth_bin_set) < 2:
            continue
        depth_bins = sorted(depth_bin_set, key=_depth_bin_sort_key)
        leads = sorted(leads_by_variable[variable_name])
        rmsd_matrix = []
        bias_matrix = []
        count_matrix = []
        for depth_bin in depth_bins:
            rmsd_row = []
            bias_row = []
            count_row = []
            for lead in leads:
                key = (variable_name, depth_bin, lead)
                cell_count = count.get(key, 0)
                if cell_count == 0:
                    rmsd_row.append(None)
                    bias_row.append(None)
                    count_row.append(0)
                    continue
                rmsd_row.append(float(numpy.sqrt(square_sum[key] / cell_count)))
                bias_row.append(float(signed_sum[key] / cell_count))
                count_row.append(int(cell_count))
            rmsd_matrix.append(rmsd_row)
            bias_matrix.append(bias_row)
            count_matrix.append(count_row)
        variables[variable_name] = {
            "depth_bins": depth_bins,
            "leads": leads,
            "rmsd": rmsd_matrix,
            "bias": bias_matrix,
            "n": count_matrix,
        }
    return variables


def write_rmsd_by_depth(
    matchup_parquet_path: str, output_path: str, *, challenger: str, region: str, source: str | None = None
) -> str | None:
    """Write the ``rmsd-by-depth.json`` viewer artifact, returning its path (``None`` if empty).

    Computes :func:`rmsd_by_depth` from the match-up parquet and serializes it to ``output_path``
    under the frozen viewer schema (``schema_version`` 1). When no 3D multi-depth variable is
    present nothing is written and ``None`` is returned.
    """
    variables = rmsd_by_depth(matchup_parquet_path)
    if not variables:
        return None
    payload = {
        "schema_version": _RMSD_BY_DEPTH_SCHEMA_VERSION,
        "challenger": challenger,
        "region": region,
        "variables": variables,
        PROVENANCE_KEY: provenance_block(
            source=source if source is not None else os.path.basename(matchup_parquet_path),
            parameters={"region": region},
        ),
    }
    Path(os.path.dirname(output_path) or ".").mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(payload), encoding="utf-8")
    return output_path


def write_viewer_artifacts(
    *,
    forecast_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
    region: str,
    dataset_slug: str,
    output_directory: str,
    year: int | None = None,
    matchups_context,
    matchup_variables=None,
    matchup_max_workers: int | None = None,
    enable_column_store: bool = False,
) -> ViewerArtifactsResult:
    """Produce every viewer serving artifact for one evaluated dataset into ``output_directory``.

    Writes the Class-4 match-up parquet, the eddy detection census and the year-mode error
    geography / per-start RMSD under ``insights/<dataset_slug>/<region>/``, and the field pyramid
    with its manifest under ``viewer/data/``. Any artifact that cannot be produced (for example
    the year artifacts for a region without a super-observation grid) is skipped and recorded as a
    flag rather than aborting the others.
    """
    from oceanbench.runner import matchups as matchups_module
    from oceanbench.runner.run import _CLASS4_VARIABLES

    output_path = Path(output_directory)
    insights_directory = output_path / "insights" / dataset_slug / region
    insights_directory.mkdir(parents=True, exist_ok=True)
    flags: list[str] = []

    variables = matchup_variables if matchup_variables is not None else list(_CLASS4_VARIABLES)
    matchup_parquet_path = str(insights_directory / MATCHUP_PARQUET_FILENAME)
    # Compute the match-ups per forecast start in parallel and stream them straight to the serving
    # parquet, accumulating the per-start bias records from the same frames, so the whole year is
    # never held in memory at once.
    class4_bias_records: list[dict] = []
    start_partitions = matchups_module.iter_class4_matchups_by_start(
        forecast_dataset, observation_dataset, variables, context=matchups_context, max_workers=matchup_max_workers
    )
    write_matchup_parquet_streamed(
        start_partitions,
        matchup_parquet_path,
        source=f"insights/{dataset_slug}/{region}/{MATCHUP_PARQUET_FILENAME}",
        on_partition=lambda partition: class4_bias_records.extend(
            class4_bias_per_start_records(partition, context=matchups_context)
        ),
    )

    eddy_census_path = str(insights_directory / EDDY_CENSUS_FILENAME)
    try:
        write_eddy_census(forecast_dataset, eddy_census_path, dataset_slug=dataset_slug)
    except Exception as error:  # noqa: BLE001 - one artifact must not abort the others
        flags.append(f"eddy census skipped: {error}")
        eddy_census_path = None

    pyramid_zarr_path = None
    pyramid_manifest_path = None
    try:
        layers, specs = viewer_layers(forecast_dataset)
        pyramid = build_pyramid(
            layers,
            specs,
            output_path=str(output_path / "viewer" / "data" / f"{dataset_slug}.zarr"),
            dataset_slug=dataset_slug,
            year=year,
        )
        pyramid_zarr_path = pyramid.zarr_path
        pyramid_manifest_path = pyramid.manifest_path
    except Exception as error:  # noqa: BLE001
        flags.append(f"pyramid skipped: {error}")

    column_store_zarr_path = None
    if enable_column_store:
        from oceanbench.publish.column_store import COLUMN_STORE_SUFFIX, build_column_store

        try:
            column_store = build_column_store(
                forecast_dataset,
                output_path=str(output_path / "viewer" / "data" / f"{dataset_slug}{COLUMN_STORE_SUFFIX}"),
                dataset_slug=dataset_slug,
                year=year,
            )
            column_store_zarr_path = column_store.zarr_path
        except Exception as error:  # noqa: BLE001
            flags.append(f"column store skipped: {error}")

    year_error_geography_path = str(insights_directory / YEAR_ERROR_GEOGRAPHY_FILENAME)
    year_rmsd_by_start_path = str(insights_directory / YEAR_RMSD_BY_START_FILENAME)
    try:
        _write_year_artifacts(
            matchup_parquet_path,
            region,
            year_error_geography_path,
            year_rmsd_by_start_path,
            source=f"insights/{dataset_slug}/{region}/{MATCHUP_PARQUET_FILENAME}",
        )
    except Exception as error:  # noqa: BLE001
        flags.append(f"year artifacts skipped: {error}")
        year_error_geography_path = None
        year_rmsd_by_start_path = None

    rmsd_by_depth_path = str(insights_directory / RMSD_BY_DEPTH_FILENAME)
    try:
        rmsd_by_depth_path = write_rmsd_by_depth(
            matchup_parquet_path,
            rmsd_by_depth_path,
            challenger=dataset_slug,
            region=region,
            source=f"insights/{dataset_slug}/{region}/{MATCHUP_PARQUET_FILENAME}",
        )
    except Exception as error:  # noqa: BLE001
        flags.append(f"rmsd-by-depth skipped: {error}")
        rmsd_by_depth_path = None

    return ViewerArtifactsResult(
        matchup_parquet_path=matchup_parquet_path,
        eddy_census_path=eddy_census_path,
        pyramid_zarr_path=pyramid_zarr_path,
        pyramid_manifest_path=pyramid_manifest_path,
        column_store_zarr_path=column_store_zarr_path,
        year_error_geography_path=year_error_geography_path,
        year_rmsd_by_start_path=year_rmsd_by_start_path,
        rmsd_by_depth_path=rmsd_by_depth_path,
        class4_bias_records=class4_bias_records,
        flags=flags,
    )
