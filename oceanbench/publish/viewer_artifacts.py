# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Write the files the viewer streams for one evaluated challenger and region.

The Class IV files are derived from the match-ups the scores were computed from, so the viewer
shows the same population as the scores file. Pooled RMSD values come from
``compute_class4_rmsd_table``, the function behind the official Class IV scores. Bias, the error
geography and the confidence intervals have no official counterpart and are computed here.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy
import pandas
import pyarrow
import pyarrow.compute
import pyarrow.parquet
import xarray

from oceanbench.core import eddies as eddies_core
from oceanbench.core.classIV_support import compute_class4_rmsd_table
from oceanbench.core.dataset_utils import DEPTH_BIN_DISPLAY_ORDER, Dimension, Variable
from oceanbench.core.regions import GLOBAL_REGION_NAME, RegionSpec, padded_region, subset_dataset_to_region
from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.core.version import __version__
from oceanbench.publish.column_store import COLUMN_STORE_SUFFIX, build_column_store
from oceanbench.pyramids import build_pyramid, viewer_layers

MATCHUP_PARQUET_FILENAME = "class4-matchups.parquet"
EDDY_CENSUS_FILENAME = "eddies.json"
YEAR_ERROR_GEOGRAPHY_FILENAME = "year-error-geography.json"
YEAR_RMSD_BY_START_FILENAME = "year-rmsd-by-start.json"
RMSD_BY_DEPTH_FILENAME = "rmsd-by-depth.json"
PYRAMID_SUFFIX = ".zarr"
PYRAMID_MANIFEST_SUFFIX = ".viewer-manifest.json"

# The viewer reads the match-ups by row group. A row group holds whole (variable, depth_bin) blocks of a
# single (start_date, lead_day) pair, packed up to TARGET_ROW_GROUP_ROWS rows to keep the footer small.
TARGET_ROW_GROUP_ROWS = 400_000
MAXIMUM_ROW_GROUP_ROWS = 1_000_000
MATCHUP_SCHEMA = pyarrow.schema(
    [
        ("variable", pyarrow.string()),
        ("depth_bin", pyarrow.string()),
        ("lead_day", pyarrow.int16()),
        ("start_date", pyarrow.string()),
        ("latitude", pyarrow.float32()),
        ("longitude", pyarrow.float32()),
        ("observation_value", pyarrow.float32()),
        ("model_value", pyarrow.float32()),
    ]
)
_MATCHUP_GROUP_COLUMNS = ["start_date", "lead_day", "variable", "depth_bin"]
_MATCHUP_PAIR_COLUMNS = ["start_date", "lead_day"]

_EDDY_CENSUS_SCHEMA_VERSION = "2"
_MAXIMUM_CONTOUR_POINT_COUNT = 64
_COORDINATE_ROUNDING_DECIMALS = 4

# Year-mode super-observation grids, global at 2 degrees and IBI at a quarter degree.
YEAR_GRIDS = {
    "global": {"lat0": -90, "dlat": 2, "nlat": 90, "lon0": -180, "dlon": 2, "nlon": 180},
    "ibi": {"lat0": 26.0, "dlat": 0.25, "nlat": 121, "lon0": -19.25, "dlon": 0.25, "nlon": 98},
}
_YEAR_TARGETS = [
    (Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(), "surface", "SSH"),
    (Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(), "surface", "T"),
    (Variable.SEA_WATER_SALINITY.key(), "0-5m", "S"),
    (Variable.EASTWARD_SEA_WATER_VELOCITY.key(), "15m", "u"),
    (Variable.NORTHWARD_SEA_WATER_VELOCITY.key(), "15m", "v"),
]
_YEAR_GEOGRAPHY_DECIMALS = {"SSH": 4, "T": 3, "S": 4, "u": 4, "v": 4}

# The per-start RMSD interval is a seeded percentile bootstrap over the match-ups of that start. Above
# _YEAR_RMSD_CI_EXACT_MAXIMUM match-ups it resamples equal-count quantile bins with a multinomial draw,
# equivalent at that size and independent of it. The bias interval is mean +/- 1.96 std / sqrt(n).
BOOTSTRAP_SEED = 20240703
_YEAR_RMSD_CI_BOOTSTRAP_DRAWS = 200
_YEAR_RMSD_CI_EXACT_MAXIMUM = 10_000
_YEAR_RMSD_CI_BINS = 1_000
_CONFIDENCE = 0.95
_BIAS_CI_Z = 1.96


@dataclass(frozen=True)
class ViewerArtifacts:
    insights: dict[str, str | None]
    has_maps: bool


def confidence_interval(bootstrap_values: numpy.ndarray, confidence: float) -> tuple[float, float]:
    finite = bootstrap_values[numpy.isfinite(bootstrap_values)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    low, high = numpy.percentile(finite, [(1.0 - confidence) / 2.0 * 100.0, (1.0 + confidence) / 2.0 * 100.0])
    return (float(low), float(high))


def class4_rmsd_rows(matchups: pandas.DataFrame) -> pandas.DataFrame:
    """Class IV RMSD, count and missing count per variable, depth bin and lead day (0-based)."""
    return pandas.concat(
        [compute_class4_rmsd_table(group, variable) for variable, group in matchups.groupby("variable", sort=False)],
        ignore_index=True,
    )


def class4_rmsd_rows_by_start(matchups: pandas.DataFrame) -> pandas.DataFrame:
    return pandas.concat(
        [
            class4_rmsd_rows(start_matchups).assign(first_day=first_day)
            for first_day, start_matchups in matchups.groupby("first_day", sort=True)
        ],
        ignore_index=True,
    )


def _scored(matchups: pandas.DataFrame) -> pandas.DataFrame:
    scored = matchups.dropna(subset=["observation_value", "model_value"])
    return scored.assign(signed_error=scored["model_value"] - scored["observation_value"])


def _json_float(value) -> float | None:
    return None if value is None or numpy.isnan(value) else float(value)


def _serving_matchups(matchups: pandas.DataFrame) -> pyarrow.Table:
    table = pyarrow.table(
        {
            "variable": matchups["variable"].astype(str).to_numpy(dtype=object),
            "depth_bin": matchups["depth_bin"].astype(str).to_numpy(dtype=object),
            "lead_day": matchups["lead_day"].to_numpy() + 1,
            "start_date": pandas.to_datetime(matchups["first_day"]).dt.strftime("%Y-%m-%d").to_numpy(dtype=object),
            "latitude": matchups[Dimension.LATITUDE.key()].to_numpy(),
            "longitude": matchups[Dimension.LONGITUDE.key()].to_numpy(),
            "observation_value": matchups["observation_value"].to_numpy(),
            "model_value": matchups["model_value"].to_numpy(),
        }
    ).cast(MATCHUP_SCHEMA)
    order = pyarrow.compute.sort_indices(table, sort_keys=[(column, "ascending") for column in _MATCHUP_GROUP_COLUMNS])
    return table.take(order)


def _group_boundaries(columns: list[numpy.ndarray]) -> numpy.ndarray:
    changed = numpy.zeros(len(columns[0]), dtype=bool)
    changed[0] = True
    for column in columns:
        changed[1:] |= column[1:] != column[:-1]
    return numpy.flatnonzero(changed)


def _packed_row_group_spans(table: pyarrow.Table) -> list[tuple[int, int]]:
    """``(offset, length)`` of each row group: whole (variable, depth_bin) blocks, packed per pair."""
    row_count = table.num_rows
    if not row_count:
        return []
    block_starts = _group_boundaries(
        [table.column(column).to_numpy(zero_copy_only=False) for column in _MATCHUP_GROUP_COLUMNS]
    )
    pair_starts = set(
        _group_boundaries(
            [table.column(column).to_numpy(zero_copy_only=False) for column in _MATCHUP_PAIR_COLUMNS]
        ).tolist()
    )
    block_ends = numpy.append(block_starts[1:], row_count)
    spans: list[tuple[int, int]] = []
    open_start = None
    for block_start, block_end in zip(block_starts.tolist(), block_ends.tolist()):
        if open_start is not None and block_start in pair_starts:
            spans.append((open_start, block_start - open_start))
            open_start = None
        if open_start is None:
            open_start = block_start
        while block_end - open_start > MAXIMUM_ROW_GROUP_ROWS:
            spans.append((open_start, MAXIMUM_ROW_GROUP_ROWS))
            open_start += MAXIMUM_ROW_GROUP_ROWS
        if block_end - open_start >= TARGET_ROW_GROUP_ROWS:
            spans.append((open_start, block_end - open_start))
            open_start = None
    if open_start is not None:
        spans.append((open_start, row_count - open_start))
    return spans


def write_matchup_parquet(matchups: pandas.DataFrame, output_path: Path) -> None:
    """Write the match-ups in the viewer serving layout, one forecast start at a time."""
    with pyarrow.parquet.ParquetWriter(
        output_path,
        MATCHUP_SCHEMA,
        compression="zstd",
        compression_level=3,
        write_statistics=_MATCHUP_GROUP_COLUMNS,
    ) as writer:
        for _, start_matchups in matchups.groupby("first_day", sort=True):
            table = _serving_matchups(start_matchups)
            for offset, length in _packed_row_group_spans(table):
                writer.write_table(table.slice(offset, length))
    verify_matchup_parquet(output_path)


def verify_matchup_parquet(output_path: Path) -> None:
    """Check that every row group holds one (start_date, lead_day) pair, in order, with statistics."""
    metadata = pyarrow.parquet.ParquetFile(output_path).metadata
    group_indices = [MATCHUP_SCHEMA.names.index(column) for column in _MATCHUP_GROUP_COLUMNS]
    previous_key = None
    for group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(group_index)
        statistics = [row_group.column(column_index).statistics for column_index in group_indices]
        if any(column_statistics is None for column_statistics in statistics):
            raise ValueError(f"Match-up row group {group_index} has no statistics.")
        if any(column_statistics.min != column_statistics.max for column_statistics in statistics[:2]):
            raise ValueError(f"Match-up row group {group_index} mixes several (start_date, lead_day) pairs.")
        if row_group.num_rows > MAXIMUM_ROW_GROUP_ROWS:
            raise ValueError(f"Match-up row group {group_index} has {row_group.num_rows} rows.")
        key = tuple(column_statistics.min for column_statistics in statistics)
        if previous_key is not None and key < previous_key:
            raise ValueError(f"Match-up row group {group_index} is out of order.")
        previous_key = key


def rmsd_by_depth(matchups: pandas.DataFrame) -> dict:
    """Pooled RMSD, bias and scored count per depth bin and lead day, for variables with several depth bins."""
    rows = class4_rmsd_rows(matchups).set_index(["variable", "depth_bin", "lead_day"])
    scored = _scored(matchups)
    bias = scored.groupby(["variable", "depth_bin", "lead_day"])["signed_error"].mean()
    variables = {}
    for variable in rows.index.get_level_values("variable").unique():
        variable_rows = rows.loc[variable]
        depth_bins = sorted(variable_rows.index.get_level_values("depth_bin").unique(), key=DEPTH_BIN_DISPLAY_ORDER.get)
        if len(depth_bins) < 2:
            continue
        lead_days = sorted(variable_rows.index.get_level_values("lead_day").unique())
        rmsd_matrix, bias_matrix, count_matrix = [], [], []
        for depth_bin in depth_bins:
            rmsd_row, bias_row, count_row = [], [], []
            for lead_day in lead_days:
                key = (depth_bin, lead_day)
                count = (
                    int(variable_rows.at[key, "count"] - variable_rows.at[key, "missing"])
                    if key in variable_rows.index
                    else 0
                )
                rmsd_row.append(_json_float(variable_rows.at[key, "rmsd"]) if count else None)
                bias_row.append(_json_float(bias.get((variable, depth_bin, lead_day))) if count else None)
                count_row.append(count)
            rmsd_matrix.append(rmsd_row)
            bias_matrix.append(bias_row)
            count_matrix.append(count_row)
        variables[str(variable)] = {
            "depth_bins": [str(depth_bin) for depth_bin in depth_bins],
            "leads": [int(lead_day) + 1 for lead_day in lead_days],
            "rmsd": rmsd_matrix,
            "bias": bias_matrix,
            "n": count_matrix,
        }
    return variables


def _grid_cells(latitude: numpy.ndarray, longitude: numpy.ndarray, grid: dict) -> tuple[numpy.ndarray, numpy.ndarray]:
    longitude_span = grid["nlon"] * grid["dlon"]
    wrapped_longitude = ((longitude - grid["lon0"]) % longitude_span) + grid["lon0"]
    latitude_bin = numpy.floor((latitude - grid["lat0"]) / grid["dlat"]).astype(numpy.int64)
    longitude_bin = numpy.floor((wrapped_longitude - grid["lon0"]) / grid["dlon"]).astype(numpy.int64)
    valid = (latitude_bin >= 0) & (latitude_bin < grid["nlat"]) & (longitude_bin >= 0) & (longitude_bin < grid["nlon"])
    return latitude_bin * grid["nlon"] + longitude_bin, valid


def _cell_bias_standard_error(square_sum: float, signed_sum: float, count: int, decimals: int) -> float | None:
    if count < 2:
        return None
    mean = signed_sum / count
    variance = max((square_sum / count) - mean * mean, 0.0)
    return round(float(numpy.sqrt(variance / count)), decimals)


def year_error_geography(matchups: pandas.DataFrame, grid: dict, lead_days_count: int) -> dict:
    """Mean |error|, bias, count and bias standard error per grid cell, over the whole year."""
    cell_count = grid["nlat"] * grid["nlon"]
    scored = _scored(matchups)
    geography = {
        "grid": grid,
        "variables": {},
        "meta": {
            "n_starts": int(matchups["first_day"].nunique()),
            "generated_from": MATCHUP_PARQUET_FILENAME,
            "depth_bin": {short: depth_bin for _, depth_bin, short in _YEAR_TARGETS},
            "aggregation": "time-mean of |obs-model| per cell",
            "bias_standard_error": (
                "per cell std(model-obs)/sqrt(n) under bias_se; n is the shared per-cell match-up "
                "count under n (identical sampling for the |error| and bias rasters)"
            ),
        },
    }
    for variable, depth_bin, short in _YEAR_TARGETS:
        decimals = _YEAR_GEOGRAPHY_DECIMALS[short]
        target = scored[(scored["variable"] == variable) & (scored["depth_bin"] == depth_bin)]
        leads, bias_leads, count_leads, bias_se_leads = {}, {}, {}, {}
        for lead_day in range(lead_days_count):
            lead_target = target[target["lead_day"] == lead_day]
            cell, valid = _grid_cells(
                lead_target[Dimension.LATITUDE.key()].to_numpy(),
                lead_target[Dimension.LONGITUDE.key()].to_numpy(),
                grid,
            )
            signed_error = lead_target["signed_error"].to_numpy()[valid]
            cell = cell[valid]
            count = numpy.bincount(cell, minlength=cell_count)
            absolute_sum = numpy.bincount(cell, numpy.abs(signed_error), cell_count)
            signed_sum = numpy.bincount(cell, signed_error, cell_count)
            square_sum = numpy.bincount(cell, signed_error * signed_error, cell_count)
            key = str(lead_day + 1)
            leads[key] = [
                None if cell_n == 0 else round(float(absolute_sum[index] / cell_n), decimals)
                for index, cell_n in enumerate(count)
            ]
            bias_leads[key] = [
                None if cell_n == 0 else round(float(signed_sum[index] / cell_n), decimals)
                for index, cell_n in enumerate(count)
            ]
            count_leads[key] = [int(cell_n) for cell_n in count]
            bias_se_leads[key] = [
                _cell_bias_standard_error(square_sum[index], signed_sum[index], int(cell_n), decimals)
                for index, cell_n in enumerate(count)
            ]
        geography["variables"][short] = {"leads": leads, "bias": bias_leads, "n": count_leads, "bias_se": bias_se_leads}
    return geography


def _bootstrap_rmsd_ci(
    squared_error: numpy.ndarray, generator: numpy.random.Generator
) -> tuple[float | None, float | None]:
    n = squared_error.size
    if n < 2:
        return (None, None)
    if n <= _YEAR_RMSD_CI_EXACT_MAXIMUM:
        draws = generator.integers(0, n, size=(_YEAR_RMSD_CI_BOOTSTRAP_DRAWS, n))
        bootstrap_mean = squared_error[draws].mean(axis=1)
    else:
        bins = numpy.array_split(numpy.sort(squared_error), _YEAR_RMSD_CI_BINS)
        bin_means = numpy.array([bin_values.mean() for bin_values in bins])
        bin_counts = numpy.array([bin_values.size for bin_values in bins])
        counts = generator.multinomial(n, bin_counts / n, size=_YEAR_RMSD_CI_BOOTSTRAP_DRAWS)
        bootstrap_mean = counts @ bin_means / n
    low, high = confidence_interval(numpy.sqrt(numpy.maximum(bootstrap_mean, 0.0)), _CONFIDENCE)
    if not (numpy.isfinite(low) and numpy.isfinite(high)):
        return (None, None)
    return (round(low, 6), round(high, 6))


def _analytic_bias_ci(signed_error: numpy.ndarray, bias: float) -> tuple[float | None, float | None]:
    n = signed_error.size
    if n < 2:
        return (None, None)
    half_width = _BIAS_CI_Z * float(signed_error.std()) / numpy.sqrt(n)
    return (round(bias - half_width, 6), round(bias + half_width, 6))


def year_rmsd_by_start(
    matchups: pandas.DataFrame, rows_by_start: pandas.DataFrame, grid: dict, lead_days_count: int
) -> dict:
    """Pooled RMSD, bias and their intervals per start date, for the year chart."""
    generator = numpy.random.default_rng(BOOTSTRAP_SEED)
    scored = _scored(matchups)
    signed_errors = {
        key: group["signed_error"].to_numpy()
        for key, group in scored.groupby(["variable", "depth_bin", "lead_day", "first_day"], sort=False)
    }
    rows = rows_by_start.set_index(["variable", "depth_bin", "lead_day", "first_day"])
    start_dates = sorted({key[3] for key in signed_errors})
    payload = {
        "variables": {},
        "meta": {
            "method": "pooled RMSD over all class-4 match-ups per start date (same reduction as official scores)",
            "bias_method": "pooled mean(model - obs) per start date",
            "grid": grid,
            "generated_from": MATCHUP_PARQUET_FILENAME,
            "ci_method": (
                f"rmsd_ci: seeded percentile bootstrap over the start's match-ups "
                f"({_YEAR_RMSD_CI_BOOTSTRAP_DRAWS} resamples, seed {BOOTSTRAP_SEED}, "
                f"{int(_CONFIDENCE * 100)}% percentile interval; above {_YEAR_RMSD_CI_EXACT_MAXIMUM} match-ups an "
                f"equal-count quantile-binned ({_YEAR_RMSD_CI_BINS} bins) multinomial bootstrap of the mean squared "
                f"error). bias_ci: analytic normal interval mean +/- {_BIAS_CI_Z} * std(model - obs) / sqrt(n)"
            ),
        },
    }
    for variable, depth_bin, short in _YEAR_TARGETS:
        leads = {}
        for lead_day in range(lead_days_count):
            series = {
                name: []
                for name in ["dates", "rmsd", "n", "bias", "rmsd_ci_low", "rmsd_ci_high", "bias_ci_low", "bias_ci_high"]
            }
            for first_day in start_dates:
                signed_error = signed_errors.get((variable, depth_bin, lead_day, first_day))
                if signed_error is None:
                    continue
                bias = float(signed_error.mean())
                rmsd_low, rmsd_high = _bootstrap_rmsd_ci(signed_error * signed_error, generator)
                bias_low, bias_high = _analytic_bias_ci(signed_error, bias)
                series["dates"].append(pandas.Timestamp(first_day).date().isoformat())
                series["rmsd"].append(round(float(rows.at[(variable, depth_bin, lead_day, first_day), "rmsd"]), 6))
                series["n"].append(int(signed_error.size))
                series["bias"].append(round(bias, 6))
                series["rmsd_ci_low"].append(rmsd_low)
                series["rmsd_ci_high"].append(rmsd_high)
                series["bias_ci_low"].append(bias_low)
                series["bias_ci_high"].append(bias_high)
            leads[str(lead_day + 1)] = series
        payload["variables"][short] = {"depth_bin": depth_bin, "leads": leads}
    return payload


def _decimated_contour(latitudes: numpy.ndarray, longitudes: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    if latitudes.size <= _MAXIMUM_CONTOUR_POINT_COUNT:
        return latitudes, longitudes
    stride = int(numpy.ceil(latitudes.size / _MAXIMUM_CONTOUR_POINT_COUNT))
    return latitudes[::stride], longitudes[::stride]


def _clamp(value, low: float, high: float):
    if isinstance(value, list):
        return [min(high, max(low, element)) for element in value]
    return min(high, max(low, value))


def _eddy(detection_index: int, detection_row: pandas.Series, contour_row: pandas.Series | None) -> dict:
    contour_latitudes, contour_longitudes = [], []
    if contour_row is not None:
        latitudes, longitudes = _decimated_contour(
            numpy.asarray(contour_row[eddies_core.CONTOUR_LATITUDES_COLUMN], dtype=float),
            numpy.asarray(contour_row[eddies_core.CONTOUR_LONGITUDES_COLUMN], dtype=float),
        )
        contour_latitudes = [round(float(value), _COORDINATE_ROUNDING_DECIMALS) for value in latitudes]
        contour_longitudes = [round(float(value), _COORDINATE_ROUNDING_DECIMALS) for value in longitudes]
    latitude = round(float(detection_row[eddies_core.LATITUDE_COLUMN]), _COORDINATE_ROUNDING_DECIMALS)
    longitude = round(float(detection_row[eddies_core.LONGITUDE_COLUMN]), _COORDINATE_ROUNDING_DECIMALS)
    eddy = {
        "id": int(detection_index),
        "latitude": _clamp(latitude, -90.0, 90.0),
        "longitude": _clamp(longitude, -180.0, 180.0),
        "polarity": str(detection_row[eddies_core.POLARITY_COLUMN]),
        "contour_latitude": _clamp(contour_latitudes, -90.0, 90.0),
        "contour_longitude": _clamp(contour_longitudes, -180.0, 180.0),
    }
    validate_against_schema(eddy, "eddy")
    return eddy


def _centres_inside(detections: pandas.DataFrame, region: RegionSpec) -> numpy.ndarray:
    bounds = region.bounds
    latitudes = detections[eddies_core.LATITUDE_COLUMN].to_numpy(dtype=float)
    longitudes = (detections[eddies_core.LONGITUDE_COLUMN].to_numpy(dtype=float) + 180.0) % 360.0 - 180.0
    return (
        (latitudes >= bounds.minimum_latitude)
        & (latitudes <= bounds.maximum_latitude)
        & (longitudes >= bounds.minimum_longitude)
        & (longitudes <= bounds.maximum_longitude)
    )


def eddy_census_frames(dataset: xarray.Dataset, start_index: int, centre_region: RegionSpec | None) -> list[dict]:
    """
    One frame per lead day with the eddies detected in one forecast start.

    Contours are computed once on every detection and then subset, so a kept eddy keeps the contour
    whose level defined its amplitude. ``centre_region`` keeps only the eddies centred in a region
    when ``dataset`` is a crop padded around it.
    """
    lead_day_indices = list(range(dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]))
    detections = eddies_core.detect_mesoscale_eddies(
        dataset, first_day_index=start_index, lead_day_indices=lead_day_indices
    )
    contours = (
        detections.iloc[0:0]
        if detections.empty
        else eddies_core.mesoscale_eddy_contours_from_detections(detections, dataset, first_day_index=start_index)
    )
    detections = eddies_core.filter_mesoscale_eddy_detections_by_contours(detections, contours)
    if centre_region is not None:
        detections = detections.loc[_centres_inside(detections, centre_region)]
    contour_rows = (
        {}
        if contours.empty
        else {int(row["detection_index"]): row for _, row in contours.drop_duplicates("detection_index").iterrows()}
    )
    return [
        {
            "lead_day": lead_day_index + 1,
            "detections": [
                _eddy(detection_index, detections.loc[detection_index], contour_rows.get(int(detection_index)))
                for detection_index in detections.index[detections[eddies_core.LEAD_DAY_COLUMN] == lead_day_index]
            ],
        }
        for lead_day_index in lead_day_indices
    ]


def write_eddy_census(
    dataset: xarray.Dataset, output_path: Path, dataset_slug: str, centre_region: RegionSpec | None
) -> None:
    """Write one ``eddies-start-<date>-lead-<N>.json`` per start and lead day, and the index at ``output_path``."""
    metadata = {
        "kind": "eddy-census",
        "schema_version": _EDDY_CENSUS_SCHEMA_VERSION,
        "variable": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        "dataset": dataset_slug,
        "parameters": {**eddies_core.default_eddy_detection_parameters(), "oceanbench_version": __version__},
    }
    start_entries = []
    for start_index, first_day in enumerate(dataset[Dimension.FIRST_DAY_DATETIME.key()].values):
        start_date = pandas.Timestamp(first_day).date().isoformat()
        lead_entries = []
        for frame in eddy_census_frames(dataset, start_index, centre_region):
            file_name = f"eddies-start-{start_date}-lead-{frame['lead_day']}.json"
            payload = {**metadata, "start_index": start_index, "start_date": start_date, "frame": frame}
            (output_path.parent / file_name).write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
            )
            lead_entries.append({"lead_day": frame["lead_day"], "file": file_name})
        start_entries.append({"start_index": start_index, "start_date": start_date, "leads": lead_entries})
    # "leads" repeats the first start for readers of the previous index shape.
    index = {**metadata, "starts": start_entries, "leads": start_entries[0]["leads"] if start_entries else []}
    output_path.write_text(json.dumps(index, sort_keys=True, indent=2), encoding="utf-8")


def _write_json(document: dict, path: Path) -> None:
    path.write_text(json.dumps(document), encoding="utf-8")


def write_viewer_artifacts(
    directory: Path,
    dataset_slug: str,
    challenger_dataset: xarray.Dataset,
    region: RegionSpec,
    matchups: pandas.DataFrame | None,
    rows_by_start: pandas.DataFrame | None,
) -> ViewerArtifacts:
    """
    Write the viewer files of one challenger and region under ``directory``.

    The maps (pyramid and water-column store) are written for the global region only, as
    ``<slug>.zarr``, ``<slug>.viewer-manifest.json`` and ``<slug>.columns.zarr``. The insights are
    written under ``insights/<slug>/<region>/``; the year files need a super-observation grid, so
    they are written for the official regions only.
    """
    insights_directory = directory / "insights" / dataset_slug / region.id
    insights_directory.mkdir(parents=True, exist_ok=True)
    insights: dict[str, str | None] = {
        "class4_matchups": None,
        "eddies": None,
        "rmsd_by_depth": None,
        "year_error_geography": None,
        "year_rmsd_by_start": None,
    }
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]

    if matchups is not None and len(matchups):
        write_matchup_parquet(matchups, insights_directory / MATCHUP_PARQUET_FILENAME)
        insights["class4_matchups"] = MATCHUP_PARQUET_FILENAME
        variables = rmsd_by_depth(matchups)
        if variables:
            document = {"schema_version": 1, "challenger": dataset_slug, "region": region.id, "variables": variables}
            _write_json(document, insights_directory / RMSD_BY_DEPTH_FILENAME)
            insights["rmsd_by_depth"] = RMSD_BY_DEPTH_FILENAME
        if region.id in YEAR_GRIDS:
            grid = YEAR_GRIDS[region.id]
            _write_json(
                year_error_geography(matchups, grid, lead_days_count),
                insights_directory / YEAR_ERROR_GEOGRAPHY_FILENAME,
            )
            _write_json(
                year_rmsd_by_start(matchups, rows_by_start, grid, lead_days_count),
                insights_directory / YEAR_RMSD_BY_START_FILENAME,
            )
            insights["year_error_geography"] = YEAR_ERROR_GEOGRAPHY_FILENAME
            insights["year_rmsd_by_start"] = YEAR_RMSD_BY_START_FILENAME

    # A regional census detects on a padded crop so the filter and contours see past the region edge.
    is_global = region.bounds is None
    eddy_dataset = subset_dataset_to_region(
        challenger_dataset, padded_region(region, eddies_core.REGIONAL_CENSUS_MARGIN_DEGREES)
    )
    write_eddy_census(
        eddy_dataset, insights_directory / EDDY_CENSUS_FILENAME, dataset_slug, None if is_global else region
    )
    insights["eddies"] = EDDY_CENSUS_FILENAME

    has_maps = region.id == GLOBAL_REGION_NAME
    if has_maps:
        year = pandas.Timestamp(challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values[0]).year
        build_column_store(
            challenger_dataset,
            output_path=str(directory / f"{dataset_slug}{COLUMN_STORE_SUFFIX}"),
            dataset_slug=dataset_slug,
            year=year,
        )
        layers, specs = viewer_layers(challenger_dataset)
        build_pyramid(
            layers,
            specs,
            output_path=str(directory / f"{dataset_slug}{PYRAMID_SUFFIX}"),
            dataset_slug=dataset_slug,
            year=year,
            has_columns=True,
        )
    return ViewerArtifacts(
        insights={
            key: None if file_name is None else f"./data/insights/{dataset_slug}/{region.id}/{file_name}"
            for key, file_name in insights.items()
        },
        has_maps=has_maps,
    )
