# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Put a curvilinear challenger on the regular scoring grid, once, as it is staged.

A challenger published on a tripolar grid cannot go through the gridded metrics as it is:
the interpolator of :mod:`oceanbench.core.interpolate`, the coordinate snapping of
:mod:`oceanbench.core.rmsd` and the geostrophic derivation of
:mod:`oceanbench.core.geostrophic_currents` all read separable one-dimensional axes, which a
tripolar grid does not have. :mod:`oceanbench.core.curvilinear_grid` answers that with a
nearest-neighbour sampling on the sphere, and this module is where that answer is applied:
at staging, so that everything downstream of the stage sees an ordinary regular-grid
challenger with one-dimensional latitude and longitude axes and standard variable names.

Sampling once at staging rather than at every metric is what makes it affordable. The
mapping from the regular grid to the native cells is a property of the two grids and the land
mask alone, so it is built once per challenger and reused across every forecast start, lead
day, member, depth and variable of a run; applying it is a gather, which xarray runs lazily
on the dask array of the store and which never averages anything, so the ensemble dimension
crosses the regrid untouched and every member is sampled on its own.

Three things about a NEMO store have to be handled here or they arrive as wrong numbers
rather than as a failure:

Land is data, not a missing value
    The GloEns two-dimensional store holds land as a raw zero, which decodes to 17.5 degrees
    for the surface temperature and to 0 metres for the sea surface height, and no cell of it
    is a missing value. Two fifths of the grid is land on that basis. The ocean mask is
    therefore declared by the challenger and passed to the mapping, which drops every target
    cell whose nearest native cell is land instead of importing the sentinel onto the coast.

The velocity faces are staggered and partly dry
    Each component is sampled through the positions of its own face, from
    :mod:`oceanbench.core.curvilinear_c_grid`, and through the face mask of that component,
    which is the tracer mask on both sides of the face: a face against the coast holds an
    exact zero.

The velocity components are grid relative
    They are turned onto east and north through the angle of the grid before they are given
    the standard names that say they are eastward and northward. The rotation is a pointwise
    combination of the two components, so it is applied once both have been sampled onto the
    common target grid: on the native grid they sit on different faces and combining them
    there would need an average across cells that this path deliberately never takes. Where
    the two faces of a target cell come from native cells whose axes point in different
    directions, which happens along the fold, the pair is not a vector at all and both
    components are dropped, see :data:`FOLD_DISAGREEMENT_DEGREES`.

The gather itself is the same fancy indexing
:func:`oceanbench.core.curvilinear_grid.sample_onto_target_grid` performs, expressed through
xarray so that it stays lazy, keeps the dtype of the store and accepts any number of leading
dimensions rather than one. The mapping is the campaign kernel unchanged.

Which challenger is curvilinear is declared in :data:`CURVILINEAR_CHALLENGERS`, keyed on the
challenger source name, the same key the Class IV conventions of
:mod:`oceanbench.core.classIV_support` are declared on. It is empty until a challenger is
registered, so no existing challenger changes path.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib

import numpy
import xarray

from oceanbench.core.climate_forecast_standard_names import StandardVariable
from oceanbench.core.curvilinear_c_grid import (
    GRID_TYPE_MERIDIONAL_VELOCITY,
    GRID_TYPE_ZONAL_VELOCITY,
    c_grid_ocean_mask,
    c_grid_positions,
    grid_type_of_variable,
    i_axis_angle_to_east,
    rotated_to_east_north,
)
from oceanbench.core.curvilinear_grid import (
    MAXIMUM_NEIGHBOUR_KILOMETRES,
    NearestNeighbourMapping,
    nearest_neighbour_mapping,
)
from oceanbench.core.dataset_utils import Dimension

#: The regular quarter-degree grid every curvilinear challenger is sampled onto.
#:
#: These are the axes of the staged quarter-degree GLORYS reference, which is what the
#: gridded metrics score against, so a challenger regridded here lands cell for cell on the
#: reference. Both axes are whole multiples of a quarter degree, which is exact in binary, so
#: building them here gives the same numbers the reference store holds rather than numbers
#: close to them.
STANDARD_QUARTER_DEGREE_LATITUDE = -78.0 + 0.25 * numpy.arange(672, dtype="float64")
STANDARD_QUARTER_DEGREE_LONGITUDE = -180.0 + 0.25 * numpy.arange(1440, dtype="float64")

#: Depth dimension names a NEMO store gives its staggered vertical axes.
#:
#: The three carry the same tracer level depths, bitwise, so they are one axis published
#: under three names and the scoring axis needs the one name.
NEMO_DEPTH_DIMENSIONS = ("deptht", "depthu", "depthv")

#: Grid description variables of a NEMO store, which describe the native grid and mean
#: nothing once the fields are off it.
NEMO_GRID_DESCRIPTION_VARIABLES = ("nav_lat", "nav_lon")

STANDARD_NAME_ATTRIBUTE = "standard_name"

#: How far the two velocity faces of a target cell may disagree on where the i-axis points.
#:
#: A target cell takes its zonal component from the nearest zonal face and its meridional one
#: from the nearest meridional face, and those two faces do not always belong to the same
#: native cell. Near the tripolar fold they belong to cells whose axes point in different
#: directions, and the pair is then not a vector: no single angle turns it onto east and north,
#: and sampling the angle through each component separately does not help either, because the
#: defect is the pairing and not the angle. On the GloEns grid 937 wet target cells disagree by
#: more than 60 degrees, all of them north of 69 N and the worst of them by 180 degrees, which
#: is a turned component of the wrong sign. Both components of such a cell are dropped.
FOLD_DISAGREEMENT_DEGREES = 20.0


@dataclass(frozen=True)
class CurvilinearChallenger:
    """How to put one curvilinear challenger on the regular grid.

    ``tracer_grid`` and ``tracer_ocean_mask`` are called with the dataset being staged. They
    are functions rather than arrays because a store does not always describe its own grid:
    the GloEns three-dimensional stores ship coordinate arrays that are entirely missing
    values, and both the grid and the land mask of a cycle have to be read from the
    two-dimensional store of the same initialisation.

    ``tracer_ocean_mask`` returns true on the ocean cells of the tracer grid. It is required:
    a store that holds land as a value rather than as a missing value would otherwise have
    that value sampled onto every coastal cell of the target grid, where it reads as ocean.
    """

    tracer_grid: Callable[[xarray.Dataset], tuple[numpy.ndarray, numpy.ndarray]]
    tracer_ocean_mask: Callable[[xarray.Dataset], numpy.ndarray]
    source_dimensions: tuple[str, str] = ("y", "x")
    target_latitude: numpy.ndarray = field(default_factory=lambda: STANDARD_QUARTER_DEGREE_LATITUDE)
    target_longitude: numpy.ndarray = field(default_factory=lambda: STANDARD_QUARTER_DEGREE_LONGITUDE)


#: Challenger source name to its curvilinear grid declaration.
#:
#: Empty until a challenger is registered, which leaves every challenger staged as it is
#: published.
CURVILINEAR_CHALLENGERS: dict[str, CurvilinearChallenger] = {}

_MAPPING_CACHE: dict[str, NearestNeighbourMapping] = {}


def curvilinear_challenger(dataset_name: str) -> CurvilinearChallenger | None:
    return CURVILINEAR_CHALLENGERS.get(dataset_name)


def ocean_mask_from_land_sentinel(values: numpy.ndarray, land_sentinel: float) -> numpy.ndarray:
    """Ocean cells of a field whose land cells hold one exact value rather than nothing.

    The comparison is exact on purpose. The sentinel is a raw zero the store decodes with its
    own scale and offset, so it comes back as the same float on every land cell, while an
    ocean cell that happens to sit at that temperature or that sea level lands on a
    neighbouring quantisation step.
    """
    return numpy.asarray(values) != land_sentinel


def _grid_fingerprint(*arrays: numpy.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = numpy.ascontiguousarray(array)
        digest.update(f"{values.shape}{values.dtype}".encode("utf-8"))
        digest.update(values.tobytes())
    return digest.hexdigest()


def curvilinear_mapping(
    source_latitude: numpy.ndarray,
    source_longitude: numpy.ndarray,
    source_ocean_mask: numpy.ndarray,
    target_latitude: numpy.ndarray,
    target_longitude: numpy.ndarray,
) -> NearestNeighbourMapping:
    """The nearest-neighbour mapping of one grid pair and one land mask, built once and kept.

    The mapping depends on nothing that changes with the date, the member, the depth or the
    variable, so one mapping serves a whole run and the cost of building it is paid once.

    Land is given to the mapping rather than removed from the search: the tree is built over
    every native cell and a target cell whose nearest native cell is land is dropped. Removing
    the land cells from the search instead would give every coastal target cell the nearest
    cell of open water, which is a real ocean value from the wrong side of the coast and is
    invisible in every aggregate.
    """
    key = _grid_fingerprint(source_latitude, source_longitude, source_ocean_mask, target_latitude, target_longitude)
    if key not in _MAPPING_CACHE:
        _MAPPING_CACHE[key] = nearest_neighbour_mapping(
            source_latitude,
            source_longitude,
            numpy.asarray(source_ocean_mask, dtype=bool),
            target_latitude,
            target_longitude,
            maximum_distance_kilometres=MAXIMUM_NEIGHBOUR_KILOMETRES,
        )
    return _MAPPING_CACHE[key]


def _variable_on_common_depth_axis(variable: xarray.DataArray, depth_key: str) -> xarray.DataArray:
    staggered = [name for name in NEMO_DEPTH_DIMENSIONS if name in variable.dims]
    if not staggered:
        return variable
    return variable.rename({staggered[0]: depth_key}).drop_vars(depth_key, errors="ignore")


def _common_depth_values(
    dataset: xarray.Dataset,
    staggered: list[str],
    depth_values: numpy.ndarray | None,
) -> numpy.ndarray | None:
    if depth_values is not None:
        return numpy.asarray(depth_values)
    with_values = [name for name in staggered if name in dataset.coords]
    return numpy.asarray(dataset[with_values[0]].values) if with_values else None


def with_common_depth_axis(dataset: xarray.Dataset, depth_values: numpy.ndarray | None = None) -> xarray.Dataset:
    """Put every staggered vertical axis of a NEMO store on the common depth axis.

    The tracer, zonal velocity and meridional velocity fields name the same levels ``deptht``,
    ``depthu`` and ``depthv``, and a store that publishes the three of them carries all three
    names at once: ``uo`` on ``depthu`` and ``vo`` on ``depthv`` in the same dataset. The
    rename is therefore taken per data variable rather than over the dataset, so the three
    names collapse onto the one scoring axis instead of colliding. Passing ``depth_values``
    replaces the values with the tracer ones, which is how a store that publishes its staggered
    axis on its own levels is put back on the tracer levels the scoring depth labels mean.
    """
    depth_key = Dimension.DEPTH.key()
    staggered = [name for name in NEMO_DEPTH_DIMENSIONS if name in dataset.dims]
    if not staggered:
        if depth_values is None or depth_key not in dataset.dims:
            return dataset
        return _with_depth_values(dataset, depth_values)
    on_staggered = [str(name) for name in dataset.data_vars if set(staggered) & set(dataset[name].dims)]
    common_values = _common_depth_values(dataset, staggered, depth_values)
    renamed = dataset.drop_dims(staggered).assign(
        {name: _variable_on_common_depth_axis(dataset[name], depth_key) for name in on_staggered}
    )
    if common_values is None:
        return renamed
    return _with_depth_values(renamed, common_values)


def _with_depth_values(dataset: xarray.Dataset, depth_values: numpy.ndarray) -> xarray.Dataset:
    depth_key = Dimension.DEPTH.key()
    if dataset.sizes[depth_key] != len(depth_values):
        raise ValueError(
            f"the dataset holds {dataset.sizes[depth_key]} levels and the tracer axis holds "
            f"{len(depth_values)}, so they are not the same levels under two names"
        )
    return dataset.assign_coords({depth_key: numpy.asarray(depth_values)})


def _without_native_grid_description(dataset: xarray.Dataset, source_dimensions: tuple[str, str]) -> xarray.Dataset:
    """Drop what describes the native grid, so nothing of it survives the regrid.

    The two-dimensional latitude and longitude the store carries name the same things as the
    axes of the regular grid, and keeping them would leave a dataset holding two answers to
    where its cells are.
    """
    described = [
        name
        for name in (*NEMO_GRID_DESCRIPTION_VARIABLES, Dimension.LATITUDE.key(), Dimension.LONGITUDE.key())
        if name in dataset.variables and set(source_dimensions) & set(dataset[name].dims)
    ]
    return dataset.drop_vars(described)


def _sampled_variable(
    variable: xarray.DataArray,
    mapping: NearestNeighbourMapping,
    source_dimensions: tuple[str, str],
) -> xarray.DataArray:
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    row_dimension, column_dimension = source_dimensions
    rows, columns = numpy.unravel_index(
        mapping.source_flat_indices,
        (variable.sizes[row_dimension], variable.sizes[column_dimension]),
    )
    sampled = variable.isel(
        {
            row_dimension: xarray.DataArray(rows, dims=(latitude_key, longitude_key)),
            column_dimension: xarray.DataArray(columns, dims=(latitude_key, longitude_key)),
        }
    )
    usable = xarray.DataArray(mapping.usable, dims=(latitude_key, longitude_key))
    return sampled.where(usable).assign_coords(
        {latitude_key: mapping.target_latitude, longitude_key: mapping.target_longitude}
    )


def _velocity_component_names(dataset: xarray.Dataset) -> tuple[str | None, str | None]:
    zonal = [str(name) for name in dataset.data_vars if grid_type_of_variable(str(name)) == GRID_TYPE_ZONAL_VELOCITY]
    meridional = [
        str(name) for name in dataset.data_vars if grid_type_of_variable(str(name)) == GRID_TYPE_MERIDIONAL_VELOCITY
    ]
    if len(zonal) > 1 or len(meridional) > 1:
        raise ValueError(f"the dataset holds several velocity components per axis, {zonal} and {meridional}")
    if bool(zonal) != bool(meridional):
        raise ValueError(
            f"the dataset holds {(zonal or meridional)[0]} without the other velocity component: the components "
            "are along the axes of the model grid, so neither can be turned onto east and north on its own"
        )
    return (zonal[0] if zonal else None, meridional[0] if meridional else None)


def _angle_disagreement(first: xarray.DataArray, second: xarray.DataArray) -> xarray.DataArray:
    return numpy.abs((first - second + numpy.pi) % (2.0 * numpy.pi) - numpy.pi)


def _folded_velocity_cells(
    tracer_latitude: numpy.ndarray,
    tracer_longitude: numpy.ndarray,
    zonal_mapping: NearestNeighbourMapping,
    meridional_mapping: NearestNeighbourMapping,
    source_dimensions: tuple[str, str],
) -> xarray.DataArray:
    """Target cells whose two velocity faces come from differently oriented native cells."""
    angle = xarray.DataArray(i_axis_angle_to_east(tracer_latitude, tracer_longitude), dims=source_dimensions)
    zonal_angle = _sampled_variable(angle, zonal_mapping, source_dimensions)
    meridional_angle = _sampled_variable(angle, meridional_mapping, source_dimensions)
    return _angle_disagreement(zonal_angle, meridional_angle) > numpy.radians(FOLD_DISAGREEMENT_DEGREES)


def _with_rotated_velocities(
    regridded: xarray.Dataset,
    zonal_name: str | None,
    meridional_name: str | None,
    angle: xarray.DataArray,
    folded: xarray.DataArray,
) -> xarray.Dataset:
    if zonal_name is None or meridional_name is None:
        return regridded
    eastward, northward = rotated_to_east_north(regridded[zonal_name], regridded[meridional_name], angle)
    eastward = eastward.where(~folded)
    northward = northward.where(~folded)
    eastward_attributes = {
        **regridded[zonal_name].attrs,
        STANDARD_NAME_ATTRIBUTE: StandardVariable.EASTWARD_SEA_WATER_VELOCITY.value,
    }
    northward_attributes = {
        **regridded[meridional_name].attrs,
        STANDARD_NAME_ATTRIBUTE: StandardVariable.NORTHWARD_SEA_WATER_VELOCITY.value,
    }
    return regridded.assign(
        {
            zonal_name: eastward.assign_attrs(eastward_attributes),
            meridional_name: northward.assign_attrs(northward_attributes),
        }
    )


def regridded_curvilinear_dataset(
    dataset: xarray.Dataset,
    tracer_latitude: numpy.ndarray,
    tracer_longitude: numpy.ndarray,
    tracer_ocean_mask: numpy.ndarray,
    *,
    source_dimensions: tuple[str, str] = ("y", "x"),
    target_latitude: numpy.ndarray = STANDARD_QUARTER_DEGREE_LATITUDE,
    target_longitude: numpy.ndarray = STANDARD_QUARTER_DEGREE_LONGITUDE,
    depth_values: numpy.ndarray | None = None,
) -> xarray.Dataset:
    """Sample every native-grid field of ``dataset`` onto the regular target grid.

    Fields that do not live on the native grid are kept as they are, and every other
    dimension of a sampled field, the ensemble one included, crosses unchanged. A field is
    sampled through the positions and the mask of the C-grid point it is carried on, so the
    velocity components come through their own staggered faces and not through the tracer
    cells, and a velocity pair is then turned onto east and north. Target cells whose nearest
    native cell is land, and cells further from the native grid than the neighbour cutoff,
    come back as missing values.
    """
    prepared = _without_native_grid_description(with_common_depth_axis(dataset, depth_values), source_dimensions)
    native = [str(name) for name in prepared.data_vars if set(source_dimensions).issubset(set(prepared[name].dims))]
    zonal_name, meridional_name = _velocity_component_names(prepared[native])

    def mapping_of(grid_type: str) -> NearestNeighbourMapping:
        return curvilinear_mapping(
            *c_grid_positions(tracer_latitude, tracer_longitude, grid_type),
            c_grid_ocean_mask(tracer_ocean_mask, grid_type),
            target_latitude,
            target_longitude,
        )

    mappings = {grid_type: mapping_of(grid_type) for grid_type in sorted({grid_type_of_variable(n) for n in native})}
    regridded = prepared.drop_vars(native).drop_dims(
        [name for name in source_dimensions if name in prepared.dims], errors="ignore"
    )
    regridded = regridded.assign(
        {
            name: _sampled_variable(prepared[name], mappings[grid_type_of_variable(name)], source_dimensions)
            for name in native
        }
    )
    if zonal_name is None:
        return regridded.assign_attrs(dataset.attrs)
    # The angle describes the grid and not the ocean, so it is sampled through a mapping that
    # masks nothing: a coastal target cell whose velocity face is wet must still be turned.
    geometry_mapping = curvilinear_mapping(
        tracer_latitude,
        tracer_longitude,
        numpy.ones(numpy.shape(tracer_latitude), dtype=bool),
        target_latitude,
        target_longitude,
    )
    angle = _sampled_variable(
        xarray.DataArray(i_axis_angle_to_east(tracer_latitude, tracer_longitude), dims=source_dimensions),
        geometry_mapping,
        source_dimensions,
    )
    folded = _folded_velocity_cells(
        tracer_latitude,
        tracer_longitude,
        mappings[GRID_TYPE_ZONAL_VELOCITY],
        mappings[GRID_TYPE_MERIDIONAL_VELOCITY],
        source_dimensions,
    )
    return _with_rotated_velocities(regridded, zonal_name, meridional_name, angle, folded).assign_attrs(dataset.attrs)


def maybe_regridded_curvilinear_dataset(dataset: xarray.Dataset, dataset_name: str) -> xarray.Dataset:
    """Regrid ``dataset`` when its challenger is declared curvilinear, otherwise return it.

    This is the staging seam: it is called on the dataset a week is staged from and on the
    dataset an unstaged run reads directly, so both routes hand the same regular-grid
    challenger to the metrics.
    """
    challenger = curvilinear_challenger(dataset_name)
    if challenger is None:
        return dataset
    tracer_latitude, tracer_longitude = challenger.tracer_grid(dataset)
    return regridded_curvilinear_dataset(
        dataset,
        tracer_latitude,
        tracer_longitude,
        challenger.tracer_ocean_mask(dataset),
        source_dimensions=challenger.source_dimensions,
        target_latitude=challenger.target_latitude,
        target_longitude=challenger.target_longitude,
    )
