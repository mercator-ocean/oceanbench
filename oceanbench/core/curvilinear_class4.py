# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Class IV matchup of a challenger left on its native curvilinear grid.

The Class IV matchup of :mod:`oceanbench.core.classIV_support` places an observation in the
model with a bilinear interpolation along the latitude and longitude axes, which a tripolar
grid does not have. :mod:`oceanbench.core.curvilinear_staging` answers that by putting the
challenger on the regular scoring grid before anything reads it, and that is the path the
gridded metrics need. Observation space does not need it: an observation is a point, not a
cell, so it can take its value from the native cell it falls in without the whole field being
resampled first.

This module is that alternative, beside the regular-grid horizontal interpolation rather than
instead of it. Nothing of the deterministic Class IV path changes: the observation dataframe,
the depth bins, the lead day grouping and the vertical descent are the same code, through
:func:`oceanbench.core.classIV_support.vertically_interpolate_class4_profiles`. Only the
horizontal step differs, from an interpolation between four axis neighbours to the nearest
native cell within the neighbour cutoff, and beyond that cutoff the observation is dropped
rather than reached for.

Taking the nearest cell rather than interpolating is the honest operation on this grid. A
bilinear stencil needs four cells whose positions bracket the point along two axes, which on
a tripolar grid means solving for the curvilinear index of the point first; the native cell
is about a quarter degree, the observation error is not much smaller than the field variation
across one, and the campaign scored its gridded metrics on the same nearest-neighbour basis.
Whether the two treatments agree is a property of the challenger and not of the code, which
is why the two entry points are siblings and a run can be scored through either.

Three things are handled here that the regular-grid path never sees:

The velocity components come from three grids at once
    ``uo`` sits on the eastern face, ``vo`` on the northern one, and neither is eastward or
    northward. Both are gathered through their own faces and turned onto east and north
    together, with the grid angle taken at the tracer cell of the observation, so a request
    for one component reads both.

The vertical axis is staggered
    ``deptht``, ``depthu`` and ``depthv`` are the same levels published under three names, so
    the axis is renamed to the common one and the tracer values are used.

Sea level cannot be scored here
    Turning a model sea surface height into an altimeter anomaly subtracts a mean dynamic
    topography resolved by grid resolution, which is not defined on a curvilinear grid. That
    conversion belongs after the regrid, so it is refused here with an error rather than
    approximated.
"""

from dataclasses import dataclass
import hashlib

import numpy
import pandas
import xarray

from oceanbench.core.classIV_support import vertically_interpolate_class4_profiles
from oceanbench.core.curvilinear_c_grid import (
    GRID_TYPE_MERIDIONAL_VELOCITY,
    GRID_TYPE_TRACER,
    GRID_TYPE_ZONAL_VELOCITY,
    c_grid_ocean_mask,
    c_grid_positions,
    grid_type_of_variable,
    i_axis_angle_to_east,
    rotated_to_east_north,
)
from oceanbench.core.curvilinear_grid import (
    MAXIMUM_NEIGHBOUR_KILOMETRES,
    ScatteredNearestNeighbourMapping,
    nearest_neighbour_at_points,
)
from oceanbench.core.curvilinear_staging import NEMO_DEPTH_DIMENSIONS, CurvilinearChallenger, curvilinear_challenger
from oceanbench.core.dataset_source import get_dataset_source
from oceanbench.core.dataset_utils import Dimension, Variable

OBSERVATION_DIMENSION = "observation"

VELOCITY_VARIABLE_KEYS = (
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
)

_OBSERVATION_MAPPING_CACHE: dict[str, ScatteredNearestNeighbourMapping] = {}


@dataclass(frozen=True)
class NativeGrid:
    """The tracer grid and land mask a native-grid matchup reads, taken once per dataset."""

    latitude: numpy.ndarray
    longitude: numpy.ndarray
    ocean_mask: numpy.ndarray
    source_dimensions: tuple[str, str]


def native_grid_of_dataset(dataset: xarray.Dataset) -> NativeGrid | None:
    """The native grid of ``dataset``, or nothing when it is not on one.

    A challenger is on its native grid when it is declared curvilinear in
    :data:`oceanbench.core.curvilinear_staging.CURVILINEAR_CHALLENGERS` and its data still
    carries the native dimensions. The second half is what makes the two paths agree: a
    declared challenger that went through the staging regrid holds ordinary axes by then and
    is matched up as any regular-grid challenger is, and only a dataset read straight from the
    native store comes through here.
    """
    source = get_dataset_source(dataset)
    declaration = None if source is None else curvilinear_challenger(source.name)
    if declaration is None or not set(declaration.source_dimensions).issubset(set(dataset.dims)):
        return None
    return native_grid_of_declaration(dataset, declaration)


def native_grid_of_declaration(dataset: xarray.Dataset, declaration: CurvilinearChallenger) -> NativeGrid:
    latitude, longitude = declaration.tracer_grid(dataset)
    return NativeGrid(
        latitude=numpy.asarray(latitude),
        longitude=numpy.asarray(longitude),
        ocean_mask=numpy.asarray(declaration.tracer_ocean_mask(dataset), dtype=bool),
        source_dimensions=declaration.source_dimensions,
    )


def _mapping_fingerprint(*arrays: numpy.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = numpy.ascontiguousarray(array)
        digest.update(f"{values.shape}{values.dtype}".encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def observation_mapping(
    native_grid: NativeGrid,
    grid_type: str,
    observation_latitude: numpy.ndarray,
    observation_longitude: numpy.ndarray,
) -> ScatteredNearestNeighbourMapping:
    """Which native cell each observation takes, built once per grid point and kept.

    The observations do not change between members, so the tree of a fifty-member ensemble is
    built once per C-grid point rather than fifty times.
    """
    key = _mapping_fingerprint(
        native_grid.latitude,
        native_grid.longitude,
        native_grid.ocean_mask,
        numpy.asarray(observation_latitude, dtype="float64"),
        numpy.asarray(observation_longitude, dtype="float64"),
        numpy.frombuffer(grid_type.encode("utf-8"), dtype="uint8"),
    )
    if key not in _OBSERVATION_MAPPING_CACHE:
        latitude, longitude = c_grid_positions(native_grid.latitude, native_grid.longitude, grid_type)
        _OBSERVATION_MAPPING_CACHE[key] = nearest_neighbour_at_points(
            latitude,
            longitude,
            c_grid_ocean_mask(native_grid.ocean_mask, grid_type),
            numpy.asarray(observation_latitude, dtype="float64"),
            numpy.asarray(observation_longitude, dtype="float64"),
            maximum_distance_kilometres=MAXIMUM_NEIGHBOUR_KILOMETRES,
        )
    return _OBSERVATION_MAPPING_CACHE[key]


def velocity_component_names(dataset: xarray.Dataset) -> tuple[str, str]:
    """The two velocity components of a native-grid dataset, along the i-axis then the j-axis.

    They are found by the C-grid point they are carried on rather than by one fixed name,
    because a native store names them ``uo`` and ``vo`` and the library renames them to their
    standard names on the way in. Both are needed to answer for either: a NEMO model runs its
    components along its own axes, so a request for the eastward current reads the northward
    one too.
    """
    zonal = [str(name) for name in dataset.data_vars if grid_type_of_variable(str(name)) == GRID_TYPE_ZONAL_VELOCITY]
    meridional = [
        str(name) for name in dataset.data_vars if grid_type_of_variable(str(name)) == GRID_TYPE_MERIDIONAL_VELOCITY
    ]
    if len(zonal) != 1 or len(meridional) != 1:
        raise ValueError(
            "matching up a current on the native grid needs exactly one velocity component per "
            f"axis, and the challenger holds {zonal} along the i-axis and {meridional} along the "
            "j-axis: neither component can be turned onto east and north on its own"
        )
    return zonal[0], meridional[0]


def _with_common_depth_dimension(model_data: xarray.DataArray, depth_values: numpy.ndarray) -> xarray.DataArray:
    depth_key = Dimension.DEPTH.key()
    staggered = [name for name in NEMO_DEPTH_DIMENSIONS if name in model_data.dims]
    renamed = model_data.rename({staggered[0]: depth_key}) if staggered else model_data
    if depth_key not in renamed.dims:
        return renamed.expand_dims({depth_key: [0.0]})
    return renamed.assign_coords({depth_key: numpy.asarray(depth_values)})


def _tracer_depth_values(dataset: xarray.Dataset) -> numpy.ndarray:
    for name in (*NEMO_DEPTH_DIMENSIONS, Dimension.DEPTH.key()):
        if name in dataset.coords:
            return numpy.asarray(dataset[name].values, dtype="float64")
    return numpy.array([0.0])


def _gathered_profiles(
    time_slice: xarray.DataArray,
    mapping: ScatteredNearestNeighbourMapping,
    row_indices: numpy.ndarray,
    source_dimensions: tuple[str, str],
) -> numpy.ndarray:
    """``(depth, observation)`` values of the native cell of each observation in the group."""
    row_dimension, column_dimension = source_dimensions
    flat = mapping.source_flat_indices[row_indices]
    rows, columns = numpy.unravel_index(flat, (time_slice.sizes[row_dimension], time_slice.sizes[column_dimension]))
    gathered = time_slice.isel(
        {
            row_dimension: xarray.DataArray(rows, dims=OBSERVATION_DIMENSION),
            column_dimension: xarray.DataArray(columns, dims=OBSERVATION_DIMENSION),
        }
    )
    values = numpy.asarray(gathered.transpose(Dimension.DEPTH.key(), OBSERVATION_DIMENSION).values, dtype="float64")
    return numpy.where(mapping.usable[row_indices][numpy.newaxis, :], values, numpy.nan)


def _observation_angle(native_grid: NativeGrid, observations_dataframe: pandas.DataFrame) -> numpy.ndarray:
    """The angle from east to the grid i-axis at the tracer cell of each observation.

    The angle describes the grid rather than the ocean, so it is taken with every tracer cell
    in the search: an observation just inside the coast has a wet velocity face and must still
    be turned.
    """
    geometry_grid = NativeGrid(
        latitude=native_grid.latitude,
        longitude=native_grid.longitude,
        ocean_mask=numpy.ones(native_grid.latitude.shape, dtype=bool),
        source_dimensions=native_grid.source_dimensions,
    )
    mapping = observation_mapping(
        geometry_grid,
        GRID_TYPE_TRACER,
        observations_dataframe[Dimension.LATITUDE.key()].to_numpy("float64"),
        observations_dataframe[Dimension.LONGITUDE.key()].to_numpy("float64"),
    )
    angle = i_axis_angle_to_east(native_grid.latitude, native_grid.longitude).ravel()[mapping.source_flat_indices]
    return numpy.where(mapping.usable, angle, numpy.nan)


def _model_values_on_one_grid_point(
    model_data: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
    native_grid: NativeGrid,
    grid_type: str,
    variable_key: str,
) -> numpy.ndarray:
    mapping = observation_mapping(
        native_grid,
        grid_type,
        observations_dataframe[Dimension.LATITUDE.key()].to_numpy("float64"),
        observations_dataframe[Dimension.LONGITUDE.key()].to_numpy("float64"),
    )
    model_depths = numpy.asarray(model_data[Dimension.DEPTH.key()].values, dtype="float64")
    first_day_to_index = {
        first_day: index for index, first_day in enumerate(model_data[Dimension.FIRST_DAY_DATETIME.key()].values)
    }
    lead_day_to_index = {
        lead_day: index for index, lead_day in enumerate(model_data[Dimension.LEAD_DAY_INDEX.key()].values)
    }
    model_values = numpy.full(len(observations_dataframe), numpy.nan)
    for (first_day, lead_day), group in observations_dataframe.groupby(["first_day", "lead_day"], sort=False):
        time_slice = model_data.isel(
            {
                Dimension.FIRST_DAY_DATETIME.key(): first_day_to_index[first_day],
                Dimension.LEAD_DAY_INDEX.key(): lead_day_to_index[lead_day],
            }
        ).compute()
        row_indices = group.index.to_numpy()
        model_values[row_indices] = vertically_interpolate_class4_profiles(
            _gathered_profiles(time_slice, mapping, row_indices, native_grid.source_dimensions),
            model_depths,
            group[Dimension.DEPTH.key()].to_numpy("float64"),
            variable_key,
        )
    return model_values


def interpolate_class4_native_model_to_observations(
    challenger_dataset: xarray.Dataset,
    variable_key: str,
    observations_dataframe: pandas.DataFrame,
    native_grid: NativeGrid,
) -> numpy.ndarray:
    """One member's model values at every observation, taken from the native cells.

    ``challenger_dataset`` is one member of the challenger, on its native grid, holding the
    requested variable and, for a velocity component, the other component too. The returned
    array is aligned with ``observations_dataframe`` row for row and holds a missing value
    wherever the nearest native cell is land, is further away than the neighbour cutoff, or
    the vertical descent could not reach the observation depth.
    """
    if variable_key == Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key():
        raise ValueError(
            "sea level cannot be matched up on a native curvilinear grid: converting a model "
            "sea surface height to an altimeter anomaly subtracts a mean dynamic topography "
            "resolved by grid resolution, which a curvilinear grid does not have. Score it "
            "through the regridded staging path of oceanbench.core.curvilinear_staging."
        )
    if variable_key not in VELOCITY_VARIABLE_KEYS and variable_key not in challenger_dataset.data_vars:
        raise ValueError(f"the challenger holds {sorted(challenger_dataset.data_vars)} and not {variable_key}")
    observations_dataframe = observations_dataframe.reset_index(drop=True)
    depth_values = _tracer_depth_values(challenger_dataset)

    def values_of(name: str) -> numpy.ndarray:
        return _model_values_on_one_grid_point(
            _with_common_depth_dimension(challenger_dataset[name], depth_values),
            observations_dataframe,
            native_grid,
            grid_type_of_variable(name),
            variable_key,
        )

    if variable_key not in VELOCITY_VARIABLE_KEYS:
        return values_of(variable_key)

    zonal_name, meridional_name = velocity_component_names(challenger_dataset)
    eastward, northward = rotated_to_east_north(
        values_of(zonal_name),
        values_of(meridional_name),
        _observation_angle(native_grid, observations_dataframe),
    )
    return eastward if variable_key == VELOCITY_VARIABLE_KEYS[0] else northward


def interpolate_class4_native_ensemble_to_observations(
    challenger_dataset: xarray.Dataset,
    variable_key: str,
    observations_dataframe: pandas.DataFrame,
    native_grid: NativeGrid,
    *,
    ensemble_dimension: str,
) -> numpy.ndarray:
    """Native-grid model values at every observation for every member, shape ``(n, M)``.

    The member loop is here for the same reason it is in
    :func:`oceanbench.core.ensemble_class4.interpolate_class4_ensemble_to_observations`: one
    member slice goes in at a time, so the matchup itself never has a member dimension to
    reason about. The observation to cell mapping does not depend on the member, so it is
    built on the first member and read from the cache by all the others.
    """
    if ensemble_dimension not in challenger_dataset.dims:
        raise ValueError(f"the challenger has no {ensemble_dimension} dimension, found {list(challenger_dataset.dims)}")
    return numpy.stack(
        [
            interpolate_class4_native_model_to_observations(
                challenger_dataset.isel({ensemble_dimension: member_index}),
                variable_key,
                observations_dataframe,
                native_grid,
            )
            for member_index in range(challenger_dataset.sizes[ensemble_dimension])
        ],
        axis=1,
    )
