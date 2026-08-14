# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Score an ensemble challenger on the gridded axis against a staged gridded reference.

The scoring itself lives in :mod:`oceanbench.core.ensemble_gridded` and the curvilinear
sampling in :mod:`oceanbench.core.curvilinear_grid`; this script is the thin runner that
finds the data, drives one forecast start at a time and writes the records out.

One invocation scores one forecast start against one reference, so a campaign is a sequence
of short resumable jobs rather than one long one. An existing output parquet is left alone
unless ``--force`` is passed.

    python score_ensemble_gridded.py score --start-date 2024-01-04 --reference glorys
    python score_ensemble_gridded.py aggregate

References follow the deterministic gridded tables, which score every challenger against both
the GLORYS reanalysis (the primary reference) and the GLO12 analysis. Both are read from the
weekly reference stage the deterministic runs already populate, so no reference is downloaded
twice and the ensemble axis scores exactly the fields the deterministic axis scored.
"""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time

import numpy
import pandas
import s3fs
import xarray

from oceanbench.core.curvilinear_grid import NearestNeighbourMapping, nearest_neighbour_mapping, sample_onto_target_grid
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.ensemble_gridded import (
    ENSEMBLE_DIMENSION,
    METRIC_ENSEMBLE_MEAN_RMSD,
    METRIC_ENSEMBLE_SPREAD,
    METRIC_SPREAD_ERROR_RATIO,
    EnsembleFieldStatistics,
    continuous_ranked_probability_score,
    ensemble_field_statistics,
    ensemble_gridded_records,
    ensemble_spread,
)
from oceanbench.core.score_records import RunContext, records_to_dataframe
from oceanbench.core.version import __version__ as OCEANBENCH_VERSION

CLOUDFERRO_ENDPOINT = "https://s3.waw3-1.cloudferro.com"
ENSEMBLE_BUCKET = "MOISICEEF"
ENSEMBLE_STORE_PATTERN = "glo4-ens50_ng_1d-m_*_2DT-oce_fcst_R{start_date:%Y%m%d}.zarr"

CHALLENGER_NAME = "gloens"
CHALLENGER_VERSION = "glo4-ens50_ng"
REGION_NAME = "global"
SURFACE_DEPTH_LABEL = "surface"

DEFAULT_STAGE_ROOT = Path("/scratch/jseillade/probax/stage-extract")
DEFAULT_OUTPUT_ROOT = Path("/scratch/jseillade/probax/gridded/output")

REFERENCE_STAGE_DIRECTORIES = {
    "glorys": "reference-glorys-quarter_degree-10d",
    "glo12": "reference-glo12-quarter_degree-10d",
}
REFERENCE_WEEK_LEAD_DAYS = 10

# The ensemble stores carry no land mask: every value is finite and land holds a constant
# fill. A land cell holds the fill at every time and every member, so the mask is the
# intersection over several independent samples; a single sample would misclassify the ocean
# cells that coincide with the fill value. Three samples make that coincidence impossible.
LAND_FILL_VALUE = 17.5
LAND_MASK_SAMPLES = ((0, 0), (13, 25), (27, 49))

# The two grids are both about a quarter degree, so a target cell more than two cells from any
# source cell is outside the source domain rather than merely offset.
MAXIMUM_NEIGHBOUR_KILOMETRES = 55.0


# Sea surface height carries a per-system datum, so a raw zos difference between two systems
# holds a constant offset that is an artefact of the datum rather than a forecast error. The
# two shifts below are the SSH-to-SLA shifts each system was calibrated with against altimeter
# SLA, so subtracting each from its own field puts both on the common altimeter datum.
#
# The ensemble shift was measured on the (zos - ssh_ib) basis, not on plain zos
# (artifacts/sla-shift-glo4ens.json: "Model SLA at scoring time is (zos - ssh_ib) - MDT -
# shift"), so the datum-aligned comparison must remove the inverse-barometer field from the
# ensemble side too. The reference shift is the deterministic path's
# REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT, applied to plain zos and shared by both
# references. The MDT term is common to both sides and cancels in a field-vs-field difference.
#
# Plain zos is scored as well, unaligned, under the "surface" depth label: it is what the
# validation slice scored and it is the basis to read if the inverse-barometer removal turns
# out not to be like-for-like against the references.
ENSEMBLE_DATUM_SHIFT = -0.160262
REFERENCE_DATUM_SHIFT = -0.1148

SURFACE_ALIGNED_DEPTH_LABEL = "surface-datum-aligned"


@dataclass(frozen=True)
class EnsembleVariable:
    """One scored variable: its name in the ensemble store and in the reference stage."""

    ensemble_name: str
    standard_variable: Variable
    reference_name: str
    reference_depth_index: int | None
    depth_label: str = SURFACE_DEPTH_LABEL
    # Ensemble field subtracted from ``ensemble_name`` before scoring, if any.
    ensemble_minus_name: str | None = None
    ensemble_datum_shift: float = 0.0
    reference_datum_shift: float = 0.0


SCORED_VARIABLES = (
    EnsembleVariable(
        ensemble_name="tos",
        standard_variable=Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
        reference_name="thetao",
        reference_depth_index=0,
    ),
    EnsembleVariable(
        ensemble_name="zos",
        standard_variable=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
        reference_name="zos",
        reference_depth_index=None,
    ),
    EnsembleVariable(
        ensemble_name="zos",
        standard_variable=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
        reference_name="zos",
        reference_depth_index=None,
        depth_label=SURFACE_ALIGNED_DEPTH_LABEL,
        ensemble_minus_name="ssh_ib",
        ensemble_datum_shift=ENSEMBLE_DATUM_SHIFT,
        reference_datum_shift=REFERENCE_DATUM_SHIFT,
    ),
)


def _open_ensemble_dataset(start_date: pandas.Timestamp) -> tuple[xarray.Dataset, str]:
    filesystem = s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": CLOUDFERRO_ENDPOINT})
    pattern = f"{ENSEMBLE_BUCKET}/{ENSEMBLE_STORE_PATTERN.format(start_date=start_date)}"
    matches = sorted(filesystem.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one ensemble store for {start_date:%Y-%m-%d}, found {matches}")
    dataset = xarray.open_zarr(s3fs.S3Map(root=matches[0], s3=filesystem, check=False), consolidated=True)
    return dataset, matches[0]


def _ensemble_ocean_mask(dataset: xarray.Dataset) -> numpy.ndarray:
    land = numpy.ones(dataset["latitude"].shape, dtype=bool)
    for time_index, member_index in LAND_MASK_SAMPLES:
        land &= dataset["tos"].isel(time=time_index, ens=member_index).values == LAND_FILL_VALUE
    return ~land


def _reference_week_store(stage_root: Path, reference: str, valid_day: pandas.Timestamp) -> tuple[Path, int] | None:
    """The staged weekly reference store covering ``valid_day``, and the index of that day.

    The stage is keyed on the forecast start dates of the deterministic runs and each store
    holds ten consecutive valid days from that start, so several stores can cover the same
    day. Overlapping stores hold bit-identical fields (both references are analyses of the
    day, not forecasts of it), so the earliest covering store is taken.

    ``None`` when no staged week covers the day. The 2024 stage ends on 2025-01-03, so the
    last forecast start of the year runs two days past it; that start still contributes its
    covered lead days rather than failing outright.
    """
    directory = stage_root / REFERENCE_STAGE_DIRECTORIES[reference]
    covering = sorted(
        (path, (valid_day - pandas.Timestamp(path.stem)).days)
        for path in directory.glob("*.zarr")
        if 0 <= (valid_day - pandas.Timestamp(path.stem)).days < REFERENCE_WEEK_LEAD_DAYS
    )
    return covering[0] if covering else None


def _reference_field(
    stage_root: Path,
    reference: str,
    valid_day: pandas.Timestamp,
    variable: EnsembleVariable,
    open_stores: dict[Path, xarray.Dataset],
) -> numpy.ndarray | None:
    covering = _reference_week_store(stage_root, reference, valid_day)
    if covering is None:
        return None
    store_path, lead_day_index = covering
    if store_path not in open_stores:
        open_stores[store_path] = xarray.open_zarr(store_path)
    dataset = open_stores[store_path]
    field = dataset[variable.reference_name].isel(lead_day_index=lead_day_index)
    if variable.reference_depth_index is not None:
        field = field.isel(depth=variable.reference_depth_index)
    return field.values.astype("float64") - variable.reference_datum_shift


def _reference_grid(stage_root: Path, reference: str) -> tuple[numpy.ndarray, numpy.ndarray]:
    directory = stage_root / REFERENCE_STAGE_DIRECTORIES[reference]
    dataset = xarray.open_zarr(sorted(directory.glob("*.zarr"))[0])
    return dataset["lat"].values, dataset["lon"].values


def _member_fields(
    dataset: xarray.Dataset,
    variable: EnsembleVariable,
    time_index: int,
    member_count: int,
) -> numpy.ndarray:
    """The fifty member fields of one variable at one time, on the ensemble's own grid.

    The datum shift and the inverse-barometer removal are applied here so that everything
    downstream of this function sees one field per member and does not need to know which
    basis it is on.
    """
    selection = {"time": time_index, "ens": slice(0, member_count)}
    values = dataset[variable.ensemble_name].isel(**selection).values
    if variable.ensemble_minus_name is not None:
        values = values - dataset[variable.ensemble_minus_name].isel(**selection).values
    return values - variable.ensemble_datum_shift


def _score_start_date(
    start_date: pandas.Timestamp,
    *,
    references: list[str],
    stage_root: Path,
    lead_days: list[int],
    member_count: int,
    write_maps: bool,
) -> tuple[dict[str, dict[tuple[object, int, str], EnsembleFieldStatistics]], dict[str, xarray.Dataset], dict]:
    """Score one forecast start against every requested reference in a single pass.

    The ensemble field of one (lead day, variable) costs a full-globe read of all fifty
    members, which is the dominant cost of the whole run, so it is read once and compared
    against each reference in turn rather than once per reference. All references are staged
    on the same quarter-degree grid, so one sampling mapping serves them all.
    """
    started = time.time()
    ensemble_dataset, store_root = _open_ensemble_dataset(start_date)
    print(f"ensemble store: {store_root}", flush=True)

    ocean_mask = _ensemble_ocean_mask(ensemble_dataset)
    print(f"ensemble land fraction {1 - ocean_mask.mean():.4f}", flush=True)

    mapping = _sampling_mapping(ensemble_dataset, ocean_mask, stage_root, references)
    print(mapping.describe(), flush=True)

    member_coordinate = ensemble_dataset["ens"].values[:member_count]
    open_reference_stores: dict[Path, xarray.Dataset] = {}
    statistics: dict[str, dict[str, dict[tuple[object, int, str], EnsembleFieldStatistics]]] = {
        reference: {} for reference in references
    }
    map_fields: dict[tuple[str, str, str, str], list[tuple[int, xarray.DataArray]]] = {}
    uncovered: list[str] = []

    for variable in SCORED_VARIABLES:
        for lead_day in lead_days:
            lead_started = time.time()
            valid_day = start_date + pandas.Timedelta(days=lead_day)
            reference_fields = {
                reference: _reference_field(stage_root, reference, valid_day, variable, open_reference_stores)
                for reference in references
            }
            covered = [reference for reference, field in reference_fields.items() if field is not None]
            for reference in references:
                if reference not in covered:
                    uncovered.append(f"{reference}/{variable.ensemble_name}/{variable.depth_label}/lead{lead_day}")
                    print(
                        f"  {reference} {variable.ensemble_name} lead {lead_day}: "
                        f"no staged reference covers {valid_day:%Y-%m-%d}, skipped",
                        flush=True,
                    )
            if not covered:
                # Nothing to score against, so the expensive member read is not worth doing.
                continue

            members = sample_onto_target_grid(
                _member_fields(ensemble_dataset, variable, lead_day, member_count),
                mapping,
                leading_dimension=ENSEMBLE_DIMENSION,
                leading_coordinate=member_coordinate,
            )
            for reference in covered:
                reference_field = _reference_grid_field(reference_fields[reference], mapping)
                field_statistics = ensemble_field_statistics(members, reference_field)
                variable_key = variable.standard_variable.key()
                statistics[reference].setdefault(variable.depth_label, {})[
                    (start_date.date(), lead_day, variable_key)
                ] = field_statistics
                if write_maps:
                    map_fields.setdefault(
                        (reference, variable_key, variable.depth_label, "ensemble_spread"), []
                    ).append((lead_day, ensemble_spread(members)))
                    map_fields.setdefault((reference, variable_key, variable.depth_label, "crps_fair"), []).append(
                        (lead_day, continuous_ranked_probability_score(members, reference_field))
                    )
                print(
                    f"  {reference} {variable.ensemble_name} [{variable.depth_label}] lead {lead_day}: "
                    f"crps {field_statistics.crps_fair:.4f} "
                    f"rmsd {numpy.sqrt(field_statistics.ensemble_mean_squared_error):.4f} "
                    f"cells {field_statistics.scored_cell_count} "
                    f"({time.time() - lead_started:.1f}s)",
                    flush=True,
                )

    timing = {
        "uncovered_lead_days": uncovered,
        "start_date": f"{start_date:%Y-%m-%d}",
        "references": references,
        "member_count": member_count,
        "lead_days": lead_days,
        "ensemble_datum_shift": ENSEMBLE_DATUM_SHIFT,
        "reference_datum_shift": REFERENCE_DATUM_SHIFT,
        "datum_aligned_basis": "(zos - ssh_ib) - ensemble_shift  vs  zos - reference_shift",
        "usable_target_fraction": round(mapping.usable_fraction, 6),
        "neighbour_km_median": round(float(numpy.median(mapping.distance_kilometres[mapping.usable])), 2),
        "neighbour_km_max": round(float(mapping.distance_kilometres[mapping.usable].max()), 2),
        "total_seconds": round(time.time() - started, 1),
    }
    return statistics, _maps_datasets(map_fields), timing


def _sampling_mapping(
    ensemble_dataset: xarray.Dataset,
    ocean_mask: numpy.ndarray,
    stage_root: Path,
    references: list[str],
) -> NearestNeighbourMapping:
    grids = {reference: _reference_grid(stage_root, reference) for reference in references}
    first_latitude, first_longitude = grids[references[0]]
    for reference, (latitude, longitude) in grids.items():
        if not (numpy.array_equal(latitude, first_latitude) and numpy.array_equal(longitude, first_longitude)):
            raise RuntimeError(f"reference {reference} is staged on a different grid from {references[0]}")
    return nearest_neighbour_mapping(
        ensemble_dataset["latitude"].values,
        ensemble_dataset["longitude"].values,
        ocean_mask,
        first_latitude,
        first_longitude,
        maximum_distance_kilometres=MAXIMUM_NEIGHBOUR_KILOMETRES,
    )


def _maps_datasets(
    map_fields: dict[tuple[str, str, str, str], list[tuple[int, xarray.DataArray]]],
) -> dict[str, xarray.Dataset]:
    """Stack the per-lead-day maps into one ``(lead_day, latitude, longitude)`` array per field.

    Each field carries its own lead-day coordinate, built from the lead days actually scored,
    so a field that lost a lead day to reference coverage still stacks.
    """
    references = {reference for reference, _variable_key, _depth, _name in map_fields}
    return {
        reference: xarray.Dataset(
            {
                f"{variable_key}_{depth}_{name}".replace("-", "_"): xarray.concat(
                    [field for _lead_day, field in maps],
                    dim=pandas.Index([lead_day for lead_day, _field in maps], name="lead_day"),
                ).astype("float32")
                for (map_reference, variable_key, depth, name), maps in map_fields.items()
                if map_reference == reference
            }
        )
        for reference in references
    }


def _reference_grid_field(values: numpy.ndarray, mapping: NearestNeighbourMapping) -> xarray.DataArray:
    """Wrap a reference field already on the target grid, masking the cells the mapping drops."""
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    return xarray.DataArray(
        numpy.where(mapping.usable, values, numpy.nan),
        dims=[latitude_key, longitude_key],
        coords={latitude_key: mapping.target_latitude, longitude_key: mapping.target_longitude},
    )


def _output_stem(reference: str, start_date: pandas.Timestamp, member_count: int) -> str:
    suffix = "" if member_count == 50 else f"-m{member_count}"
    return f"{CHALLENGER_NAME}-{reference}-{start_date:%Y%m%d}{suffix}"


def _run_context() -> RunContext:
    return RunContext(
        challenger=CHALLENGER_NAME,
        challenger_version=CHALLENGER_VERSION,
        year=2024,
        region=REGION_NAME,
        oceanbench_version=OCEANBENCH_VERSION,
    )


def _score_command(arguments: argparse.Namespace) -> None:
    start_date = pandas.Timestamp(arguments.start_date)
    output_root = Path(arguments.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stems = {reference: _output_stem(reference, start_date, arguments.members) for reference in arguments.references}
    if all((output_root / f"scores-{stem}.parquet").exists() for stem in stems.values()) and not arguments.force:
        print(f"every score file for {start_date:%Y-%m-%d} already exists, skipping (use --force to rebuild)")
        return

    statistics, maps, timing = _score_start_date(
        start_date,
        references=arguments.references,
        stage_root=Path(arguments.stage_root),
        lead_days=list(range(arguments.first_lead_day, arguments.last_lead_day + 1)),
        member_count=arguments.members,
        write_maps=arguments.write_maps,
    )

    for reference, stem in stems.items():
        # One call per depth label, because the label is a property of the whole call and the
        # two sea-surface-height bases share a variable key and are told apart by it.
        records = [
            record
            for depth_label, depth_statistics in sorted(statistics[reference].items())
            for record in ensemble_gridded_records(
                depth_statistics,
                context=_run_context(),
                reference=reference,
                depth=depth_label,
            )
        ]
        if not records:
            print(f"no covered lead day for {reference} at {start_date:%Y-%m-%d}, no score file written")
            continue
        records_to_dataframe(records).to_parquet(
            output_root / f"scores-{stem}.parquet", index=False, compression="zstd"
        )
        if reference in maps:
            maps[reference].to_netcdf(output_root / f"maps-{stem}.nc")
        print(f"wrote {output_root / f'scores-{stem}.parquet'} ({len(records)} records)")
    (output_root / f"timing-{_output_stem('all', start_date, arguments.members)}.json").write_text(
        json.dumps(timing, indent=2) + "\n"
    )
    print("TIMING " + json.dumps(timing))


def _aggregate_command(arguments: argparse.Namespace) -> None:
    output_root = Path(arguments.output_root)
    score_files = sorted(output_root.glob("scores-*.parquet"))
    score_files = [path for path in score_files if path.name != "scores-per-start.parquet"]
    if not score_files:
        raise RuntimeError(f"no per-start score files under {output_root}")

    per_start = pandas.concat([pandas.read_parquet(path) for path in score_files], ignore_index=True)
    per_start = per_start[per_start["start_date"].notna()]
    aggregated = _aggregate_over_start_dates(per_start)

    per_start.to_parquet(output_root / "scores-per-start.parquet", index=False, compression="zstd")
    aggregated.to_parquet(output_root / "scores.parquet", index=False, compression="zstd")
    aggregated.to_csv(output_root / "scores.csv", index=False)
    print(f"wrote {output_root / 'scores.parquet'} ({len(aggregated)} rows) from {len(score_files)} per-start files")


AGGREGATION_KEYS = ["challenger", "challenger_version", "region", "reference", "variable", "depth", "lead_day"]


def _aggregate_over_start_dates(per_start: pandas.DataFrame) -> pandas.DataFrame:
    """Mean over forecast starts, with the spread-error ratio rebuilt from its two averages.

    Averaging the ratio itself would let one start with a near-zero error dominate it, so it
    is recomputed from the averaged spread and the averaged ensemble-mean RMSD, which is what
    ``ensemble_gridded_records`` already does within a single forecast start.
    """
    values = per_start.pivot_table(index=AGGREGATION_KEYS, columns="metric", values="value", aggfunc="mean")
    values[METRIC_SPREAD_ERROR_RATIO] = values[METRIC_ENSEMBLE_SPREAD] / values[METRIC_ENSEMBLE_MEAN_RMSD]
    counts = per_start.groupby(AGGREGATION_KEYS, dropna=False).agg(
        start_count=("start_date", "nunique"), scored_cells=("n", "max")
    )
    return values.join(counts).reset_index().sort_values(AGGREGATION_KEYS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="score one forecast start against one reference")
    score_parser.add_argument("--start-date", required=True)
    score_parser.add_argument(
        "--references",
        nargs="+",
        default=["glorys", "glo12"],
        choices=sorted(REFERENCE_STAGE_DIRECTORIES),
        help="gridded references to score against, all in one pass over the ensemble stream",
    )
    score_parser.add_argument("--stage-root", default=str(DEFAULT_STAGE_ROOT))
    score_parser.add_argument("--members", type=int, default=50)
    score_parser.add_argument("--first-lead-day", type=int, default=1)
    score_parser.add_argument("--last-lead-day", type=int, default=10)
    score_parser.add_argument("--write-maps", action="store_true")
    score_parser.add_argument("--force", action="store_true")
    score_parser.set_defaults(function=_score_command)

    aggregate_parser = subparsers.add_parser("aggregate", help="average the per-start score files")
    aggregate_parser.set_defaults(function=_aggregate_command)

    arguments = parser.parse_args()
    arguments.function(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
