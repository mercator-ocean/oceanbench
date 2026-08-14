# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Score ensemble challengers on the gridded axis against the staged GLORYS reference.

This is the multi-challenger sibling of ``score_ensemble_gridded.py``. That script scores the
GloEns surface fields and is left untouched so the frozen surface record cannot move; this one
adds the challengers and the depth axis it does not cover:

  glonet2-ens       8 members, regular quarter-degree grid, 20 levels to 763.33 m
  glonet2-ens-icp   the initial-condition-perturbed variant of the same
  gloens-depth      the GloEns 3D stores, 50 members, tripolar grid, 75 levels

The metrics, the records and the aggregation all come from ``oceanbench.core.ensemble_gridded``
unchanged, so a row written here is the same kind of row the surface run wrote.

    python score_ensemble_gridded_multi.py --output-root DIR score --challenger glonet2-ens \\
        --start-date 2024-01-03
    python score_ensemble_gridded_multi.py --output-root DIR aggregate
"""

import argparse
from dataclasses import dataclass
import json
import os
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

DEFAULT_STAGE_ROOT = Path("/scratch/jseillade/probax/stage-extract")

REFERENCE_STAGE_DIRECTORIES = {
    "glorys": "reference-glorys-quarter_degree-10d",
    "glo12": "reference-glo12-quarter_degree-10d",
}
REFERENCE_WEEK_LEAD_DAYS = 10

REGION_NAME = "global"
SURFACE_DEPTH_LABEL = "surface"

# The nominal depth grid shared by every challenger scored here, so a row from one model joins
# a row from another on the depth label alone.
#
# These are the native levels of the staged quarter-degree GLORYS reference, capped at the
# glonet2 family ceiling of 763.33 m by design: the deeper GloEns levels are deliberately out
# of scope so that the three models are compared on the range all of them cover. The cap drops
# stage levels 902.34 m and below.
#
# The glonet2 stores carry these ten values exactly (their levels 10..19), so on that side the
# "nearest native" mapping is an exact match and contributes no vertical interpolation error.
# The GloEns 3D stores are on a different 75-level grid and take the nearest level to each.
NOMINAL_SUBSURFACE_DEPTHS = (
    47.374,
    92.326,
    155.851,
    222.475,
    318.127,
    380.213,
    453.938,
    541.089,
    643.567,
    763.333,
)
DEPTH_CAP_METRES = 763.333
DEPTH_CAP_REASON = "capped at the glonet2 family ceiling 763.33 m for like-for-like comparability"


def _depth_label(nominal_depth: float) -> str:
    return f"{nominal_depth:.3f}m"


# Sea surface height carries a per-system datum. The glonet2 family was calibrated against
# altimeter SLA with the same shift the reference path applies to its own zos
# (REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT), and unlike GloEns it ships no inverse-barometer
# field, so there is no basis change either. Subtracting an identical constant from both sides
# of a difference is a no-op, so glonet2 zos is scored once, on the plain basis, and the
# datum-aligned duplicate the GloEns run needed is not written. See the metadata note.
GLONET2_DATUM_SHIFT = -0.1148
REFERENCE_DATUM_SHIFT = -0.1148

# The staged quarter-degree reference has no level between 0.494 m and 47.374 m, so the 15 m
# velocity field the observation-space campaign scores has no like-for-like partner here and
# uo/vo are not scored. This is a stated omission, not a silent drop.
VELOCITY_OMISSION_REASON = (
    "uo/vo not scored: the staged quarter-degree reference has no level near 15 m "
    "(nearest are 0.494 m and 47.374 m), so a 15 m challenger field has no like-for-like partner"
)


@dataclass(frozen=True)
class ScoredVariable:
    """One scored field: where it lives on each side, and under which depth label."""

    challenger_name: str
    standard_variable: Variable
    reference_name: str
    depth_label: str
    # Nominal depth in metres, or None for a genuinely two-dimensional field.
    nominal_depth: float | None
    challenger_datum_shift: float = 0.0
    reference_datum_shift: float = 0.0


@dataclass(frozen=True)
class ChallengerSpec:
    """Everything that differs between the ensemble challengers scored here."""

    name: str
    version: str
    member_dimension: str
    member_count: int
    variables: tuple[ScoredVariable, ...]
    first_lead_day: int
    last_lead_day: int
    # Store time index holding the field valid at ``start_date + lead_day - time_index_offset``.
    # glonet2 filenames carry the first valid day, so lead 1 sits at time index 0; the GloEns
    # stores carry the initialisation day at time index 0, so lead 1 sits at time index 1.
    lead_day_to_time_index: int
    curvilinear: bool
    # One store per init holding every variable, or one store per variable per init.
    store_per_variable: bool
    anonymous: bool


def _glonet2_variables() -> tuple[ScoredVariable, ...]:
    variables = [
        ScoredVariable(
            challenger_name="thetao",
            standard_variable=Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
            reference_name="thetao",
            depth_label=SURFACE_DEPTH_LABEL,
            nominal_depth=0.494,
        ),
        ScoredVariable(
            challenger_name="zos",
            standard_variable=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
            reference_name="zos",
            depth_label=SURFACE_DEPTH_LABEL,
            nominal_depth=None,
            challenger_datum_shift=GLONET2_DATUM_SHIFT,
            reference_datum_shift=REFERENCE_DATUM_SHIFT,
        ),
    ]
    for depth in NOMINAL_SUBSURFACE_DEPTHS:
        variables.append(
            ScoredVariable(
                challenger_name="thetao",
                standard_variable=Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
                reference_name="thetao",
                depth_label=_depth_label(depth),
                nominal_depth=depth,
            )
        )
        variables.append(
            ScoredVariable(
                challenger_name="so",
                standard_variable=Variable.SEA_WATER_SALINITY,
                reference_name="so",
                depth_label=_depth_label(depth),
                nominal_depth=depth,
            )
        )
    return tuple(variables)


def _gloens_depth_variables() -> tuple[ScoredVariable, ...]:
    """Subsurface only: the GloEns surface record already exists and is not regenerated."""
    variables = []
    for depth in NOMINAL_SUBSURFACE_DEPTHS:
        variables.append(
            ScoredVariable(
                challenger_name="thetao",
                standard_variable=Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
                reference_name="thetao",
                depth_label=_depth_label(depth),
                nominal_depth=depth,
            )
        )
        variables.append(
            ScoredVariable(
                challenger_name="so",
                standard_variable=Variable.SEA_WATER_SALINITY,
                reference_name="so",
                depth_label=_depth_label(depth),
                nominal_depth=depth,
            )
        )
    return tuple(variables)


CHALLENGERS = {
    "glonet2-ens": ChallengerSpec(
        name="glonet2-ens",
        version="glonet2-ens",
        member_dimension="member",
        member_count=8,
        variables=_glonet2_variables(),
        first_lead_day=1,
        last_lead_day=9,
        lead_day_to_time_index=1,
        curvilinear=False,
        store_per_variable=False,
        anonymous=False,
    ),
    "glonet2-ens-icp": ChallengerSpec(
        name="glonet2-ens-icp",
        version="glonet2-ens-icp",
        member_dimension="member",
        member_count=8,
        variables=_glonet2_variables(),
        first_lead_day=1,
        last_lead_day=9,
        lead_day_to_time_index=1,
        curvilinear=False,
        store_per_variable=False,
        anonymous=False,
    ),
    "gloens-depth": ChallengerSpec(
        name="gloens",
        version="glo4-ens50_ng",
        member_dimension="ens",
        member_count=50,
        variables=_gloens_depth_variables(),
        first_lead_day=1,
        last_lead_day=10,
        lead_day_to_time_index=0,
        curvilinear=True,
        store_per_variable=True,
        anonymous=True,
    ),
}

GLONET2_BUCKET = "oceanbench-bucket"
GLONET2_PREFIX = "dev/ml-forecast-outputs"
GLOENS_BUCKET = "MOISICEEF"
GLOENS_3D_PATTERN = "glo4-ens50_ng_1d-m_*_3D{grid}-{variable}_fcst_R{start_date:%Y%m%d}.zarr"
GLOENS_3D_GRID_CODE = {"thetao": "T", "so": "T", "uo": "U", "vo": "V"}

MAXIMUM_NEIGHBOUR_KILOMETRES = 55.0


def _filesystem(anonymous: bool) -> s3fs.S3FileSystem:
    if anonymous:
        return s3fs.S3FileSystem(anon=True, client_kwargs={"endpoint_url": CLOUDFERRO_ENDPOINT})
    key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret = os.environ.get("AWS_SECRET_ACCESS_KEY")
    if not key or not secret:
        raise RuntimeError(
            "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY must be set for the dev prefix; "
            "source the campaign credentials file"
        )
    return s3fs.S3FileSystem(key=key, secret=secret, client_kwargs={"endpoint_url": CLOUDFERRO_ENDPOINT})


def _open_challenger(
    specification: ChallengerSpec, start_date: pandas.Timestamp, variable_name: str | None
) -> tuple[xarray.Dataset, str]:
    filesystem = _filesystem(specification.anonymous)
    if specification.store_per_variable:
        grid_code = GLOENS_3D_GRID_CODE[variable_name]
        store_name = GLOENS_3D_PATTERN.format(grid=grid_code, variable=variable_name, start_date=start_date)
        pattern = f"{GLOENS_BUCKET}/{store_name}"
        matches = sorted(filesystem.glob(pattern))
        if len(matches) != 1:
            raise RuntimeError(f"expected one store for {variable_name} at {start_date:%Y-%m-%d}, found {matches}")
        root = matches[0]
    else:
        root = f"{GLONET2_BUCKET}/{GLONET2_PREFIX}/{specification.name}/{start_date:%Y%m%d}.zarr"
    dataset = xarray.open_zarr(s3fs.S3Map(root=root, s3=filesystem, check=False), consolidated=True)
    return dataset, root


def _challenger_depth_index(dataset: xarray.Dataset, nominal_depth: float) -> tuple[int, float]:
    """Index of the challenger level nearest ``nominal_depth``, and that level's own depth."""
    name = "depth" if "depth" in dataset.coords else "deptht"
    depths = dataset[name].values.astype("float64")
    index = int(numpy.abs(depths - nominal_depth).argmin())
    return index, float(depths[index])


def _reference_depth_index(reference_depths: numpy.ndarray, nominal_depth: float) -> int:
    index = int(numpy.abs(reference_depths - nominal_depth).argmin())
    if abs(reference_depths[index] - nominal_depth) > 0.01:
        raise RuntimeError(
            f"nominal depth {nominal_depth} is not a staged reference level; " f"nearest is {reference_depths[index]}"
        )
    return index


def _reference_week_store(stage_root: Path, reference: str, valid_day: pandas.Timestamp) -> tuple[Path, int] | None:
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
    variable: ScoredVariable,
    reference_depths: numpy.ndarray,
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
    if variable.nominal_depth is not None:
        field = field.isel(depth=_reference_depth_index(reference_depths, variable.nominal_depth))
    return field.values.astype("float64") - variable.reference_datum_shift


def _reference_grid(stage_root: Path, reference: str) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    directory = stage_root / REFERENCE_STAGE_DIRECTORIES[reference]
    dataset = xarray.open_zarr(sorted(directory.glob("*.zarr"))[0])
    return dataset["lat"].values, dataset["lon"].values, dataset["depth"].values.astype("float64")


def _target_dataarray(values: numpy.ndarray, latitude: numpy.ndarray, longitude: numpy.ndarray) -> xarray.DataArray:
    return xarray.DataArray(
        values,
        dims=[Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={Dimension.LATITUDE.key(): latitude, Dimension.LONGITUDE.key(): longitude},
    )


def _member_dataarray(
    values: numpy.ndarray,
    member_coordinate: numpy.ndarray,
    latitude: numpy.ndarray,
    longitude: numpy.ndarray,
) -> xarray.DataArray:
    """Wrap already-on-target-grid member fields, using the dimension name the metrics expect."""
    return xarray.DataArray(
        values,
        dims=["member", Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            "member": member_coordinate,
            Dimension.LATITUDE.key(): latitude,
            Dimension.LONGITUDE.key(): longitude,
        },
    )


def _identity_grid_check(dataset: xarray.Dataset, latitude: numpy.ndarray, longitude: numpy.ndarray) -> None:
    challenger_latitude = dataset["latitude"].values.astype("float64")
    challenger_longitude = dataset["longitude"].values.astype("float64")
    if not (
        challenger_latitude.shape == latitude.shape
        and challenger_longitude.shape == longitude.shape
        and numpy.array_equal(challenger_latitude, latitude)
        and numpy.array_equal(challenger_longitude, longitude)
    ):
        raise RuntimeError(
            "challenger grid is not identical to the staged reference grid; "
            "this challenger was configured as a same-grid challenger"
        )


def _gloens_source_grid(start_date: pandas.Timestamp) -> tuple[numpy.ndarray, numpy.ndarray]:
    """The tripolar grid of the GloEns stores, read from the two-dimensional store.

    The 3D stores ship latitude and longitude arrays that are entirely missing values, so the
    grid is taken from the 2D store of the same initialisation, which carries it intact on the
    same 1049 x 1440 shape. The two stores were checked to hold the same fields on the same
    cells: the 3D surface level and the 2D surface temperature agree to the 0.006 K
    quantisation step of the stores' integer encoding, and their land masks agree on 99.994%
    of cells, so the borrowed grid describes the 3D fields as well.
    """
    filesystem = _filesystem(anonymous=True)
    pattern = f"{GLOENS_BUCKET}/glo4-ens50_ng_1d-m_*_2DT-oce_fcst_R{start_date:%Y%m%d}.zarr"
    matches = sorted(filesystem.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one 2D GloEns store for {start_date:%Y-%m-%d}, found {matches}")
    dataset = xarray.open_zarr(s3fs.S3Map(root=matches[0], s3=filesystem, check=False), consolidated=True)
    return dataset["latitude"].values, dataset["longitude"].values


def _curvilinear_mapping(
    dataset: xarray.Dataset,
    variable_name: str,
    source_latitude: numpy.ndarray,
    source_longitude: numpy.ndarray,
    latitude,
    longitude,
) -> NearestNeighbourMapping:
    """Nearest-neighbour mapping from the tripolar GloEns grid onto the staged grid.

    The 3D stores carry a genuine missing value on land, so the ocean mask is read from a
    single level rather than reconstructed from a fill constant as the 2D stores require.
    """
    sample = dataset[variable_name].isel({"time": 0, "ens": 0, "deptht": 0}).values
    ocean_mask = numpy.isfinite(sample)
    return nearest_neighbour_mapping(
        source_latitude,
        source_longitude,
        ocean_mask,
        latitude,
        longitude,
        maximum_distance_kilometres=MAXIMUM_NEIGHBOUR_KILOMETRES,
    )


def _score_start_date(
    specification: ChallengerSpec,
    start_date: pandas.Timestamp,
    *,
    references: list[str],
    stage_root: Path,
    lead_days: list[int],
    write_maps: bool,
) -> tuple[dict, dict[str, xarray.Dataset], dict]:
    started = time.time()
    latitude, longitude, reference_depths = _reference_grid(stage_root, references[0])
    for reference in references[1:]:
        other_latitude, other_longitude, _ = _reference_grid(stage_root, reference)
        if not (numpy.array_equal(latitude, other_latitude) and numpy.array_equal(longitude, other_longitude)):
            raise RuntimeError(f"reference {reference} is staged on a different grid from {references[0]}")

    statistics: dict[str, dict[str, dict[tuple[object, int, str], EnsembleFieldStatistics]]] = {
        reference: {} for reference in references
    }
    map_fields: dict[tuple[str, str, str, str], list[tuple[int, xarray.DataArray]]] = {}
    open_reference_stores: dict[Path, xarray.Dataset] = {}
    open_challenger_stores: dict[str | None, tuple[xarray.Dataset, str]] = {}
    mappings: dict[str | None, NearestNeighbourMapping] = {}
    depth_mapping: dict[str, dict] = {}
    uncovered: list[str] = []

    # Group by challenger variable so a per-variable store is opened once, and so the members
    # of one (variable, lead day) read serve every depth taken from it.
    challenger_variables = sorted({variable.challenger_name for variable in specification.variables})

    for challenger_variable in challenger_variables:
        store_key = challenger_variable if specification.store_per_variable else None
        if store_key not in open_challenger_stores:
            open_challenger_stores[store_key] = _open_challenger(
                specification, start_date, challenger_variable if specification.store_per_variable else None
            )
            dataset, root = open_challenger_stores[store_key]
            print(f"challenger store: {root}", flush=True)
            if specification.curvilinear:
                mapping_started = time.time()
                source_latitude, source_longitude = _gloens_source_grid(start_date)
                mappings[store_key] = _curvilinear_mapping(
                    dataset, challenger_variable, source_latitude, source_longitude, latitude, longitude
                )
                print(
                    f"{mappings[store_key].describe()} ({time.time() - mapping_started:.1f}s)",
                    flush=True,
                )
            else:
                _identity_grid_check(dataset, latitude, longitude)
                mappings[store_key] = None
        dataset, _root = open_challenger_stores[store_key]
        mapping = mappings[store_key]
        member_coordinate = dataset[specification.member_dimension].values[: specification.member_count]

        variables_here = [v for v in specification.variables if v.challenger_name == challenger_variable]
        # Resolve every depth this variable needs once, so one chunk read serves all of them.
        depth_indices: dict[str, tuple[int | None, float | None]] = {}
        for variable in variables_here:
            if variable.nominal_depth is None:
                depth_indices[variable.depth_label] = (None, None)
                continue
            index, native = _challenger_depth_index(dataset, variable.nominal_depth)
            depth_indices[variable.depth_label] = (index, native)
            depth_mapping.setdefault(
                variable.depth_label,
                {
                    "nominal_depth_m": variable.nominal_depth,
                    "challenger_level_index": index,
                    "challenger_depth_m": round(native, 4),
                    "reference_level_index": _reference_depth_index(reference_depths, variable.nominal_depth),
                    "reference_depth_m": round(
                        float(reference_depths[_reference_depth_index(reference_depths, variable.nominal_depth)]), 4
                    ),
                },
            )

        for lead_day in lead_days:
            time_index = lead_day - specification.lead_day_to_time_index
            valid_day = start_date + pandas.Timedelta(days=time_index)
            lead_started = time.time()

            reference_fields = {
                variable.depth_label: {
                    reference: _reference_field(
                        stage_root, reference, valid_day, variable, reference_depths, open_reference_stores
                    )
                    for reference in references
                }
                for variable in variables_here
            }
            if all(field is None for per_variable in reference_fields.values() for field in per_variable.values()):
                for variable in variables_here:
                    uncovered.append(f"{variable.challenger_name}/{variable.depth_label}/lead{lead_day}")
                print(
                    f"  {challenger_variable} lead {lead_day}: no staged reference covers "
                    f"{valid_day:%Y-%m-%d}, skipped",
                    flush=True,
                )
                continue

            members_by_depth = _read_members(
                dataset,
                specification,
                challenger_variable,
                time_index,
                sorted({index for index, _native in depth_indices.values() if index is not None}),
                has_depth=any(index is not None for index, _n in depth_indices.values()),
            )
            read_seconds = time.time() - lead_started

            for variable in variables_here:
                index, _native = depth_indices[variable.depth_label]
                raw_members = members_by_depth[index]
                shifted = raw_members - variable.challenger_datum_shift
                if mapping is None:
                    members = _member_dataarray(shifted, member_coordinate, latitude, longitude)
                else:
                    members = sample_onto_target_grid(
                        shifted, mapping, leading_dimension="member", leading_coordinate=member_coordinate
                    )
                for reference in references:
                    reference_values = reference_fields[variable.depth_label][reference]
                    if reference_values is None:
                        uncovered.append(
                            f"{reference}/{variable.challenger_name}/{variable.depth_label}/lead{lead_day}"
                        )
                        continue
                    if mapping is not None:
                        reference_values = numpy.where(mapping.usable, reference_values, numpy.nan)
                    reference_field = _target_dataarray(reference_values, latitude, longitude)
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
                        f"  {reference} {variable.challenger_name} [{variable.depth_label}] lead {lead_day}: "
                        f"crps {field_statistics.crps_fair:.4f} "
                        f"rmsd {numpy.sqrt(field_statistics.ensemble_mean_squared_error):.4f} "
                        f"cells {field_statistics.scored_cell_count}",
                        flush=True,
                    )
            print(
                f"  {challenger_variable} lead {lead_day} done "
                f"(read {read_seconds:.1f}s, total {time.time() - lead_started:.1f}s)",
                flush=True,
            )
            del members_by_depth

    timing = {
        "challenger": specification.name,
        "challenger_variant": None,
        "start_date": f"{start_date:%Y-%m-%d}",
        "references": references,
        "member_count": specification.member_count,
        "lead_days": lead_days,
        "lead_day_to_time_index_offset": specification.lead_day_to_time_index,
        "depth_cap_m": DEPTH_CAP_METRES,
        "depth_cap_reason": DEPTH_CAP_REASON,
        "nominal_subsurface_depths_m": list(NOMINAL_SUBSURFACE_DEPTHS),
        "depth_mapping": depth_mapping,
        "velocity_note": VELOCITY_OMISSION_REASON,
        "uncovered_lead_days": uncovered,
        "total_seconds": round(time.time() - started, 1),
    }
    return statistics, _maps_datasets(map_fields), timing


def _read_members(
    dataset: xarray.Dataset,
    specification: ChallengerSpec,
    variable_name: str,
    time_index: int,
    depth_indices: list[int],
    *,
    has_depth: bool,
) -> dict[int | None, numpy.ndarray]:
    """The member fields of one variable at one time, for every depth needed, at once.

    The GloEns 3D stores chunk the whole 75-level column together, so one member costs one
    column read whichever level is wanted; every needed level is taken out of that read before
    it is dropped, and the column itself is never held beyond the member being read. The
    glonet2 stores chunk one level at a time and carry every member in the same chunk, so
    there the read is per level and already covers the ensemble.
    """
    member_slice = slice(0, specification.member_count)
    if not has_depth:
        values = dataset[variable_name].isel({"time": time_index, specification.member_dimension: member_slice}).values
        return {None: values.astype("float64")}

    depth_name = "deptht" if "deptht" in dataset[variable_name].dims else "depth"
    if specification.curvilinear:
        # Column-chunked: read one member at a time and keep only the wanted levels.
        per_depth = {index: [] for index in depth_indices}
        for member_index in range(specification.member_count):
            column = dataset[variable_name].isel({"time": time_index, specification.member_dimension: member_index})
            column = column.isel({depth_name: depth_indices}).values
            for position, index in enumerate(depth_indices):
                per_depth[index].append(column[position].astype("float32"))
            del column
        return {index: numpy.stack(fields).astype("float64") for index, fields in per_depth.items()}

    # Level-chunked and member-complete: one read per level covers the whole ensemble.
    return {
        index: dataset[variable_name]
        .isel({"time": time_index, specification.member_dimension: member_slice, depth_name: index})
        .values.astype("float64")
        for index in depth_indices
    }


def _maps_datasets(
    map_fields: dict[tuple[str, str, str, str], list[tuple[int, xarray.DataArray]]],
) -> dict[str, xarray.Dataset]:
    references = {reference for reference, _variable_key, _depth, _name in map_fields}
    return {
        reference: xarray.Dataset(
            {
                f"{variable_key}_{depth}_{name}".replace("-", "_")
                .replace(".", "p"): xarray.concat(
                    [field for _lead_day, field in maps],
                    dim=pandas.Index([lead_day for lead_day, _field in maps], name="lead_day"),
                )
                .astype("float32")
                for (map_reference, variable_key, depth, name), maps in map_fields.items()
                if map_reference == reference
            }
        )
        for reference in references
    }


def _output_stem(challenger_key: str, reference: str, start_date: pandas.Timestamp) -> str:
    return f"{challenger_key}-{reference}-{start_date:%Y%m%d}"


def _run_context(specification: ChallengerSpec) -> RunContext:
    return RunContext(
        challenger=specification.name,
        challenger_version=specification.version,
        year=2024,
        region=REGION_NAME,
        oceanbench_version=OCEANBENCH_VERSION,
    )


def _score_command(arguments: argparse.Namespace) -> None:
    specification = CHALLENGERS[arguments.challenger]
    start_date = pandas.Timestamp(arguments.start_date)
    output_root = Path(arguments.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    stems = {reference: _output_stem(arguments.challenger, reference, start_date) for reference in arguments.references}
    if all((output_root / f"scores-{stem}.parquet").exists() for stem in stems.values()) and not arguments.force:
        print(f"every score file for {start_date:%Y-%m-%d} already exists, skipping (use --force to rebuild)")
        return

    first_lead = arguments.first_lead_day if arguments.first_lead_day is not None else specification.first_lead_day
    last_lead = arguments.last_lead_day if arguments.last_lead_day is not None else specification.last_lead_day

    statistics, maps, timing = _score_start_date(
        specification,
        start_date,
        references=arguments.references,
        stage_root=Path(arguments.stage_root),
        lead_days=list(range(first_lead, last_lead + 1)),
        write_maps=arguments.write_maps,
    )
    timing["challenger_variant"] = arguments.challenger

    for reference, stem in stems.items():
        records = [
            record
            for depth_label, depth_statistics in sorted(statistics[reference].items())
            for record in ensemble_gridded_records(
                depth_statistics, context=_run_context(specification), reference=reference, depth=depth_label
            )
        ]
        if not records:
            print(f"no covered lead day for {reference} at {start_date:%Y-%m-%d}, no score file written")
            continue
        frame = records_to_dataframe(records)
        # The record context carries the model family name, so the variant is kept explicitly.
        frame["challenger_version"] = specification.version
        frame.to_parquet(output_root / f"scores-{stem}.parquet", index=False, compression="zstd")
        if reference in maps:
            maps[reference].to_netcdf(output_root / f"maps-{stem}.nc")
        print(f"wrote {output_root / f'scores-{stem}.parquet'} ({len(records)} records)")

    (output_root / f"timing-{_output_stem(arguments.challenger, 'all', start_date)}.json").write_text(
        json.dumps(timing, indent=2) + "\n"
    )
    print("TIMING " + json.dumps({k: v for k, v in timing.items() if k != "depth_mapping"}))


AGGREGATION_KEYS = ["challenger", "challenger_version", "region", "reference", "variable", "depth", "lead_day"]


def _aggregate_over_start_dates(per_start: pandas.DataFrame) -> pandas.DataFrame:
    values = per_start.pivot_table(index=AGGREGATION_KEYS, columns="metric", values="value", aggfunc="mean")
    values[METRIC_SPREAD_ERROR_RATIO] = values[METRIC_ENSEMBLE_SPREAD] / values[METRIC_ENSEMBLE_MEAN_RMSD]
    counts = per_start.groupby(AGGREGATION_KEYS, dropna=False).agg(
        start_count=("start_date", "nunique"), scored_cells=("n", "max")
    )
    return values.join(counts).reset_index().sort_values(AGGREGATION_KEYS)


def _aggregate_command(arguments: argparse.Namespace) -> None:
    output_root = Path(arguments.output_root)
    score_files = sorted(output_root.glob("scores-*.parquet"))
    score_files = [path for path in score_files if path.name != "scores-per-start.parquet"]
    if not score_files:
        raise RuntimeError(f"no per-start score files under {output_root}")
    if arguments.expect_starts is not None:
        starts = {path.stem.rsplit("-", 1)[-1] for path in score_files}
        if len(starts) < arguments.expect_starts:
            raise RuntimeError(
                f"completeness guard: {len(starts)} distinct forecast starts under {output_root}, "
                f"expected {arguments.expect_starts}; not aggregating a partial campaign"
            )

    per_start = pandas.concat([pandas.read_parquet(path) for path in score_files], ignore_index=True)
    per_start = per_start[per_start["start_date"].notna()]
    aggregated = _aggregate_over_start_dates(per_start)

    per_start.to_parquet(output_root / "scores-per-start.parquet", index=False, compression="zstd")
    aggregated.to_parquet(output_root / "scores.parquet", index=False, compression="zstd")
    aggregated.to_csv(output_root / "scores.csv", index=False)
    print(f"wrote {output_root / 'scores.parquet'} ({len(aggregated)} rows) from {len(score_files)} per-start files")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)

    score_parser = subparsers.add_parser("score", help="score one forecast start of one challenger")
    score_parser.add_argument("--challenger", required=True, choices=sorted(CHALLENGERS))
    score_parser.add_argument("--start-date", required=True)
    score_parser.add_argument(
        "--references", nargs="+", default=["glorys"], choices=sorted(REFERENCE_STAGE_DIRECTORIES)
    )
    score_parser.add_argument("--stage-root", default=str(DEFAULT_STAGE_ROOT))
    score_parser.add_argument("--first-lead-day", type=int, default=None)
    score_parser.add_argument("--last-lead-day", type=int, default=None)
    score_parser.add_argument("--write-maps", action="store_true")
    score_parser.add_argument("--force", action="store_true")
    score_parser.set_defaults(function=_score_command)

    aggregate_parser = subparsers.add_parser("aggregate", help="average the per-start score files")
    aggregate_parser.add_argument(
        "--expect-starts",
        type=int,
        default=None,
        help="refuse to aggregate unless this many distinct forecast starts are present",
    )
    aggregate_parser.set_defaults(function=_aggregate_command)

    arguments = parser.parse_args()
    arguments.function(arguments)
    return 0


if __name__ == "__main__":
    sys.exit(main())
