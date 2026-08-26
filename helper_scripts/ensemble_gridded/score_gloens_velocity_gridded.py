# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Score the GloEns velocity on the gridded axis, through the rotation the library performs.

``score_ensemble_gridded_multi.py`` samples a curvilinear challenger with a nearest-neighbour
gather and deliberately refuses to run velocity through it: the components of a tripolar model
are carried along the model axes, and a gather that does not turn them onto east and north
would score the model axes against an eastward and a northward reference. That refusal is the
guard in its ``_score_command`` and it stays exactly as it is.

This script is the route the guard names. It reads the two GloEns velocity stores of one
initialisation together, hands the pair to
:func:`oceanbench.core.curvilinear_staging.regridded_curvilinear_dataset`, and scores what comes
back. That function is the validated staging path and is used here unchanged: each component is
sampled through the positions and the mask of its own C-grid face, the angle of the grid is
sampled through a mapping that masks nothing, the pair is turned onto east and north after both
components are on the common target grid, and target cells whose two faces come from differently
oriented native cells are dropped. Nothing of that is reimplemented here.

The salinity of the same fill is not velocity and needs no rotation, so it is delegated to
``score_ensemble_gridded_multi.py`` and takes the nearest-neighbour path every other GloEns
tracer field took. Both halves land in one score file per forecast start, with the schema and
the record layout of every other gridded run, so the aggregation is the generic
``aggregate`` command of that same script and is not repeated here.

    python score_gloens_velocity_gridded.py --output-root DIR score --start-date 2024-01-04
    python score_ensemble_gridded_multi.py --output-root DIR aggregate
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy
import pandas
import s3fs
import xarray

sys.path.insert(0, str(Path(__file__).resolve().parent))

import score_ensemble_gridded_multi as multi  # noqa: E402

from oceanbench.core.curvilinear_staging import (  # noqa: E402
    FOLD_DISAGREEMENT_DEGREES,
    GLOENS_SOURCE_DIMENSIONS,
    STANDARD_QUARTER_DEGREE_LATITUDE,
    STANDARD_QUARTER_DEGREE_LONGITUDE,
    gloens_tracer_grid,
    gloens_tracer_ocean_mask,
    regridded_curvilinear_dataset,
)
from oceanbench.core.dataset_utils import Dimension  # noqa: E402
from oceanbench.core.ensemble_gridded import ensemble_field_statistics  # noqa: E402
from oceanbench.core.score_records import records_to_dataframe  # noqa: E402

#: The fill entry of the multi-challenger script, which declares what this run scores.
CHALLENGER_KEY = "gloens-depth-fill"

#: The velocity components, under the names the GloEns stores publish them with.
VELOCITY_COMPONENT_NAMES = ("uo", "vo")

#: The two-dimensional GloEns store pattern of one initialisation.
#:
#: It is the store that describes the grid and the land mask: the three-dimensional stores ship
#: coordinate arrays that are missing values from end to end, and they carry no field with a
#: known land value.
GLOENS_COMPANION_PATTERN = "{bucket}/glo4-ens50_ng_1d-m_*_2DT-oce_fcst_R{start_date:%Y%m%d}.zarr"

#: Why the reference is not masked again by hand on this route.
#:
#: The nearest-neighbour route of the multi-challenger script masks the reference with the
#: usable cells of its one mapping. Here a target cell carries a velocity only where its zonal
#: face is usable, its meridional face is usable and the two faces agree on where the grid
#: points, and the staging path has already written a missing value on every cell that fails
#: any of the three. The ensemble-mean error, and every area-weighted mean taken from it, is
#: therefore missing on exactly the same cells the explicit mask would have produced.
REFERENCE_MASK_NOTE = (
    "the reference is not masked separately: the staging path already writes a missing value on "
    "every target cell whose velocity faces are unusable or disagree, so the scored cells are the "
    "same set the explicit mask of the nearest-neighbour route produces"
)

#: The ensemble variance is taken over the members alone, as the frozen scorer takes it.
#:
#: :func:`oceanbench.core.ensemble_gridded.ensemble_field_statistics` computes the member
#: variance without masking it by the reference, so a cell where the challenger is wet and the
#: reference is not still contributes to the spread while contributing to no error term. This
#: run keeps that behaviour deliberately, because its rows are merged with aggregates produced
#: before the masking question was settled and a fill must not be scored on different terms from
#: the rows it fills. The masked variant is applied to every system at once in the later rescore.
SPREAD_MASKING_NOTE = (
    "ensemble variance unmasked by the reference, as in the frozen scorer, so this fill is on the "
    "same terms as the aggregates it completes; the masked variant lands for every system at once"
)


def _velocity_variables(specification) -> tuple[multi.ScoredVariable, ...]:
    return tuple(
        variable for variable in specification.variables if variable.challenger_name in VELOCITY_COMPONENT_NAMES
    )


def _tracer_variables(specification) -> tuple[multi.ScoredVariable, ...]:
    return tuple(
        variable for variable in specification.variables if variable.challenger_name not in VELOCITY_COMPONENT_NAMES
    )


def _check_target_grid(latitude: numpy.ndarray, longitude: numpy.ndarray) -> None:
    """The staged grid must be the grid the staging path was validated on, cell for cell.

    The rotation and the face mapping are built onto the target grid the library declares, so a
    staged reference on any other grid would be scored against fields sampled somewhere else.
    """
    if not (
        numpy.array_equal(latitude, STANDARD_QUARTER_DEGREE_LATITUDE)
        and numpy.array_equal(longitude, STANDARD_QUARTER_DEGREE_LONGITUDE)
    ):
        raise RuntimeError(
            "the staged reference grid is not the standard quarter-degree grid the curvilinear "
            "staging path targets, so the rotation route cannot score against it"
        )


def _open_companion_store(start_date: pandas.Timestamp) -> xarray.Dataset:
    filesystem = multi._filesystem(anonymous=True)
    pattern = GLOENS_COMPANION_PATTERN.format(bucket=multi.GLOENS_BUCKET, start_date=start_date)
    matches = sorted(filesystem.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one two-dimensional GloEns store for {start_date:%Y-%m-%d}, found {matches}")
    return xarray.open_zarr(s3fs.S3Map(root=matches[0], s3=filesystem, check=False), consolidated=True)


def _component_depth_values(dataset: xarray.Dataset, component: str) -> numpy.ndarray:
    name = [str(dimension) for dimension in dataset[component].dims if str(dimension).startswith("depth")]
    return numpy.asarray(dataset[name[0]].values, dtype="float64")


def _component_depth_dimension(dataset: xarray.Dataset, component: str) -> str:
    return [str(dimension) for dimension in dataset[component].dims if str(dimension).startswith("depth")][0]


def _bare_component(dataset: xarray.Dataset, component: str, time_index: int, level_indices: list[int]):
    """One component at one time and the wanted levels, with the native grid description dropped.

    The two stores each carry their own all-missing ``latitude`` and ``longitude`` arrays, and
    putting the two components in one dataset with those attached would ask xarray to reconcile
    two arrays of missing values against each other. They describe the native grid and this
    route replaces the native grid, so they go before the merge rather than after it.
    """
    depth_dimension = _component_depth_dimension(dataset, component)
    field = dataset[component].isel({"time": time_index}).isel({depth_dimension: level_indices})
    described = [
        name
        for name in field.coords
        if str(name) in (Dimension.LATITUDE.key(), Dimension.LONGITUDE.key(), "time", "time_centered", "time_counter")
    ]
    return field.drop_vars(described)


def _merged_velocity_dataset(
    zonal_dataset: xarray.Dataset,
    meridional_dataset: xarray.Dataset,
    time_index: int,
    level_indices: list[int],
) -> xarray.Dataset:
    """The two components of one lead day in one dataset, which is what the rotation needs.

    The components are published one store per component, and neither can be turned onto east
    and north on its own: the rotation mixes them. They are put together here, still lazy, and
    the staging path reads the pair.
    """
    return xarray.Dataset(
        {
            component: _bare_component(dataset, component, time_index, level_indices)
            for component, dataset in (
                (VELOCITY_COMPONENT_NAMES[0], zonal_dataset),
                (VELOCITY_COMPONENT_NAMES[1], meridional_dataset),
            )
        }
    )


def _members_by_component(
    regridded: xarray.Dataset, member_dimension: str, member_count: int
) -> dict[str, numpy.ndarray]:
    """Both regridded and rotated components, member by member, as ``(member, level, y, x)``.

    The two are computed together for each member because the rotation makes each of them a
    combination of both, so asking for one alone would read the other store anyway and asking
    for them separately would read both stores twice. One member of one component is a whole
    seventy-five level column in these stores, so the members are walked rather than taken at
    once, and each is kept at single precision exactly as the nearest-neighbour route keeps it.
    """
    gathered: dict[str, list[numpy.ndarray]] = {component: [] for component in VELOCITY_COMPONENT_NAMES}
    for member_index in range(member_count):
        pair = regridded[list(VELOCITY_COMPONENT_NAMES)].isel({member_dimension: member_index}).compute()
        for component in VELOCITY_COMPONENT_NAMES:
            gathered[component].append(pair[component].values.astype("float32"))
        del pair
    return {component: numpy.stack(fields) for component, fields in gathered.items()}


def _score_velocity_start_date(
    specification,
    start_date: pandas.Timestamp,
    *,
    references: list[str],
    stage_root: Path,
    lead_days: list[int],
) -> tuple[dict, dict, list[str]]:
    variables = _velocity_variables(specification)
    latitude, longitude, reference_depths = multi._reference_grid(stage_root, references[0])
    _check_target_grid(latitude, longitude)

    companion = _open_companion_store(start_date)
    tracer_latitude, tracer_longitude = gloens_tracer_grid(companion)
    tracer_ocean_mask = gloens_tracer_ocean_mask(companion)

    datasets = {}
    for component in VELOCITY_COMPONENT_NAMES:
        dataset, root = multi._open_challenger(specification, start_date, component)
        datasets[component] = dataset
        print(f"challenger store: {root}", flush=True)

    depth_values = _component_depth_values(datasets[VELOCITY_COMPONENT_NAMES[0]], VELOCITY_COMPONENT_NAMES[0])
    other_depth_values = _component_depth_values(datasets[VELOCITY_COMPONENT_NAMES[1]], VELOCITY_COMPONENT_NAMES[1])
    if not numpy.array_equal(depth_values, other_depth_values):
        raise RuntimeError(
            "the two velocity stores publish different vertical levels, so their components do not "
            "describe the same water and cannot be paired into a vector"
        )

    depth_labels = sorted({(variable.depth_label, variable.nominal_depth) for variable in variables})
    level_indices = [int(numpy.abs(depth_values - nominal).argmin()) for _label, nominal in depth_labels]
    depth_mapping = {
        label: {
            "nominal_depth_m": nominal,
            "challenger_level_index": level_index,
            "challenger_depth_m": round(float(depth_values[level_index]), 4),
            "reference_level_index": multi._reference_depth_index(reference_depths, nominal),
            "reference_depth_m": round(
                float(reference_depths[multi._reference_depth_index(reference_depths, nominal)]), 4
            ),
        }
        for (label, nominal), level_index in zip(depth_labels, level_indices)
    }

    member_coordinate = datasets[VELOCITY_COMPONENT_NAMES[0]][specification.member_dimension].values[
        : specification.member_count
    ]
    statistics: dict[str, dict[str, dict[tuple[object, int, str], object]]] = {
        reference: {} for reference in references
    }
    open_reference_stores: dict[Path, xarray.Dataset] = {}
    uncovered: list[str] = []

    for lead_day in lead_days:
        time_index = lead_day - specification.lead_day_to_time_index
        valid_day = start_date + pandas.Timedelta(days=time_index)
        lead_started = time.time()

        reference_fields = {
            (variable.challenger_name, variable.depth_label): {
                reference: multi._reference_field(
                    stage_root, reference, valid_day, variable, reference_depths, open_reference_stores
                )
                for reference in references
            }
            for variable in variables
        }
        if all(field is None for per_variable in reference_fields.values() for field in per_variable.values()):
            for variable in variables:
                uncovered.append(f"{variable.challenger_name}/{variable.depth_label}/lead{lead_day}")
            print(f"  velocity lead {lead_day}: no staged reference covers {valid_day:%Y-%m-%d}, skipped", flush=True)
            continue

        merged = _merged_velocity_dataset(
            datasets[VELOCITY_COMPONENT_NAMES[0]], datasets[VELOCITY_COMPONENT_NAMES[1]], time_index, level_indices
        )
        regrid_started = time.time()
        regridded = regridded_curvilinear_dataset(
            merged,
            tracer_latitude,
            tracer_longitude,
            tracer_ocean_mask,
            source_dimensions=GLOENS_SOURCE_DIMENSIONS,
            target_latitude=latitude,
            target_longitude=longitude,
            depth_values=depth_values[level_indices],
        )
        members = _members_by_component(regridded, specification.member_dimension, specification.member_count)
        read_seconds = time.time() - regrid_started

        for variable in variables:
            level_position = [label for label, _nominal in depth_labels].index(variable.depth_label)
            component_members = members[variable.challenger_name][:, level_position].astype("float64")
            member_field = multi._member_dataarray(component_members, member_coordinate, latitude, longitude)
            for reference in references:
                reference_values = reference_fields[(variable.challenger_name, variable.depth_label)][reference]
                if reference_values is None:
                    uncovered.append(f"{reference}/{variable.challenger_name}/{variable.depth_label}/lead{lead_day}")
                    continue
                reference_field = multi._target_dataarray(reference_values, latitude, longitude)
                field_statistics = ensemble_field_statistics(member_field, reference_field)
                statistics[reference].setdefault(variable.depth_label, {})[
                    (start_date.date(), lead_day, variable.standard_variable.key())
                ] = field_statistics
                print(
                    f"  {reference} {variable.challenger_name} [{variable.depth_label}] lead {lead_day}: "
                    f"crps {field_statistics.crps_fair:.4f} "
                    f"rmsd {numpy.sqrt(field_statistics.ensemble_mean_squared_error):.4f} "
                    f"cells {field_statistics.scored_cell_count}",
                    flush=True,
                )
        print(
            f"  velocity lead {lead_day} done (read and regrid {read_seconds:.1f}s, "
            f"total {time.time() - lead_started:.1f}s)",
            flush=True,
        )
        del members

    return statistics, depth_mapping, uncovered


def _merged_statistics(first: dict, second: dict) -> dict:
    merged = {reference: dict(per_depth) for reference, per_depth in first.items()}
    for reference, per_depth in second.items():
        for depth_label, per_key in per_depth.items():
            merged.setdefault(reference, {}).setdefault(depth_label, {}).update(per_key)
    return merged


def _score_command(arguments: argparse.Namespace) -> None:
    specification = multi.CHALLENGERS[CHALLENGER_KEY]
    start_date = pandas.Timestamp(arguments.start_date)
    output_root = Path(arguments.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stems = {reference: multi._output_stem(CHALLENGER_KEY, reference, start_date) for reference in arguments.references}
    if all((output_root / f"scores-{stem}.parquet").exists() for stem in stems.values()) and not arguments.force:
        print(f"every score file for {start_date:%Y-%m-%d} already exists, skipping (use --force to rebuild)")
        return

    first_lead = arguments.first_lead_day if arguments.first_lead_day is not None else specification.first_lead_day
    last_lead = arguments.last_lead_day if arguments.last_lead_day is not None else specification.last_lead_day
    lead_days = list(range(first_lead, last_lead + 1))
    stage_root = Path(arguments.stage_root)
    started = time.time()

    print("=== velocity, through the rotation route ===", flush=True)
    velocity_statistics, depth_mapping, uncovered = _score_velocity_start_date(
        specification,
        start_date,
        references=arguments.references,
        stage_root=stage_root,
        lead_days=lead_days,
    )

    tracer_variables = _tracer_variables(specification)
    tracer_statistics: dict = {reference: {} for reference in arguments.references}
    tracer_timing: dict = {}
    if tracer_variables:
        print("=== salinity, through the nearest-neighbour route of the multi-challenger script ===", flush=True)
        tracer_statistics, _maps, tracer_timing = multi._score_start_date(
            replace(specification, variables=tracer_variables),
            start_date,
            references=arguments.references,
            stage_root=stage_root,
            lead_days=lead_days,
            write_maps=False,
        )
        uncovered = uncovered + list(tracer_timing["uncovered_lead_days"])
        depth_mapping = {**tracer_timing["depth_mapping"], **depth_mapping}

    statistics = _merged_statistics(velocity_statistics, tracer_statistics)

    for reference, stem in stems.items():
        records = [
            record
            for depth_label, depth_statistics in sorted(statistics.get(reference, {}).items())
            for record in multi.ensemble_gridded_records(
                depth_statistics, context=multi._run_context(specification), reference=reference, depth=depth_label
            )
        ]
        if not records:
            print(f"no covered lead day for {reference} at {start_date:%Y-%m-%d}, no score file written")
            continue
        frame = records_to_dataframe(records)
        frame["challenger_version"] = specification.version
        frame.to_parquet(output_root / f"scores-{stem}.parquet", index=False, compression="zstd")
        print(f"wrote {output_root / f'scores-{stem}.parquet'} ({len(records)} records)")

    timing = {
        "challenger": specification.name,
        "challenger_variant": CHALLENGER_KEY,
        "route": "curvilinear staging rotation for velocity, nearest neighbour for salinity",
        "start_date": f"{start_date:%Y-%m-%d}",
        "references": arguments.references,
        "member_count": specification.member_count,
        "lead_days": lead_days,
        "lead_day_to_time_index_offset": specification.lead_day_to_time_index,
        "depth_cap_m": multi.DEPTH_CAP_METRES,
        "depth_cap_reason": multi.DEPTH_CAP_REASON,
        "nominal_subsurface_depths_m": list(multi.NOMINAL_SUBSURFACE_DEPTHS),
        "depth_mapping": depth_mapping,
        "fold_disagreement_degrees": FOLD_DISAGREEMENT_DEGREES,
        "reference_mask_note": REFERENCE_MASK_NOTE,
        "spread_masking_note": SPREAD_MASKING_NOTE,
        "uncovered_lead_days": uncovered,
        "total_seconds": round(time.time() - started, 1),
    }
    (output_root / f"timing-{multi._output_stem(CHALLENGER_KEY, 'all', start_date)}.json").write_text(
        json.dumps(timing, indent=2) + "\n"
    )
    print("TIMING " + json.dumps({key: value for key, value in timing.items() if key != "depth_mapping"}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="score one forecast start of the GloEns fill")
    score_parser.add_argument("--start-date", required=True)
    score_parser.add_argument(
        "--references", nargs="+", default=["glorys"], choices=sorted(multi.REFERENCE_STAGE_DIRECTORIES)
    )
    score_parser.add_argument("--stage-root", default=str(multi.DEFAULT_STAGE_ROOT))
    score_parser.add_argument("--first-lead-day", type=int, default=None)
    score_parser.add_argument("--last-lead-day", type=int, default=None)
    score_parser.add_argument("--force", action="store_true")
    score_parser.set_defaults(function=_score_command)

    arguments = parser.parse_args()
    arguments.function(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
