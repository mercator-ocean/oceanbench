# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Nearest-neighbour-on-the-sphere sampling of a curvilinear grid onto a regular one.

Some challengers are published on the native tripolar grid, where latitude and longitude are
both functions of ``(y, x)``: GloEns is regular in longitude only between about 67 S and
21 N and folds towards both poles. The regular-grid interpolator in
:mod:`oceanbench.core.interpolate` and the coordinate snapping in
:mod:`oceanbench.core.rmsd` both assume separable one-dimensional axes and are therefore
invalid on such a grid.

This module answers the only question the gridded metrics need: for each cell of the regular
scoring grid, which source cell is it. The answer is the nearest neighbour in three-dimensional
Cartesian space on the unit sphere, which is uniformly correct on the Mercator band and on the
folds alike and needs no curvilinear stencil. Regular-grid challengers keep using the standard
interpolator; this path exists for the curvilinear ones.

The mapping is a property of the two grids alone, so it is built once and reused across every
forecast start, lead day, member and variable of a run.
"""

from dataclasses import dataclass

import numpy
import xarray
from scipy.spatial import cKDTree

from oceanbench.core.dataset_utils import Dimension

EARTH_RADIUS_KILOMETRES = 6371.0

#: How far a target point may reach for its source cell before it is dropped instead.
#:
#: The source and target grids are both about a quarter degree, so this is a little over two
#: cells at the equator. A target point further than that from every source cell sits outside
#: the source grid, and sampling it would carry a value across a gap the model never resolved.
MAXIMUM_NEIGHBOUR_KILOMETRES = 55.0


def _unit_sphere_coordinates(latitudes: numpy.ndarray, longitudes: numpy.ndarray) -> numpy.ndarray:
    latitude_radians = numpy.radians(numpy.asarray(latitudes, dtype="float64").ravel())
    longitude_radians = numpy.radians(numpy.asarray(longitudes, dtype="float64").ravel())
    cosine = numpy.cos(latitude_radians)
    return numpy.column_stack(
        [
            cosine * numpy.cos(longitude_radians),
            cosine * numpy.sin(longitude_radians),
            numpy.sin(latitude_radians),
        ]
    )


def _great_circle_kilometres(chord_lengths: numpy.ndarray) -> numpy.ndarray:
    return 2.0 * numpy.arcsin(numpy.clip(chord_lengths / 2.0, 0.0, 1.0)) * EARTH_RADIUS_KILOMETRES


@dataclass(frozen=True)
class NearestNeighbourMapping:
    """Which source cell each target cell samples, and whether that sample is usable.

    ``source_flat_indices`` indexes the flattened source grid and ``distance_kilometres`` is
    the great-circle distance actually travelled. ``usable`` is false where the nearest source
    cell is land or lies further than the requested cutoff, which happens wherever the target
    grid extends past the source grid: those cells are dropped from the scoring rather than
    filled with a distant value.
    """

    source_flat_indices: numpy.ndarray
    distance_kilometres: numpy.ndarray
    usable: numpy.ndarray
    target_latitude: numpy.ndarray
    target_longitude: numpy.ndarray

    @property
    def usable_fraction(self) -> float:
        return float(self.usable.mean())

    def describe(self) -> str:
        distances = self.distance_kilometres[self.usable]
        median, ninetieth = numpy.percentile(distances, [50, 90])
        return (
            f"nearest neighbour on the sphere: {self.usable_fraction:.2%} of target cells usable, "
            f"displacement median {median:.1f} km, 90th percentile {ninetieth:.1f} km, "
            f"maximum {distances.max():.1f} km"
        )


def nearest_neighbour_mapping(
    source_latitude: numpy.ndarray,
    source_longitude: numpy.ndarray,
    source_ocean_mask: numpy.ndarray,
    target_latitude: numpy.ndarray,
    target_longitude: numpy.ndarray,
    *,
    maximum_distance_kilometres: float,
) -> NearestNeighbourMapping:
    """Map every cell of the regular target grid onto its nearest source cell.

    The tree is built over *every* source cell, not only the ocean ones, and a target cell
    whose nearest source cell is land is marked unusable. Searching ocean cells only would
    instead pull a value from across the coast onto a land target cell, which silently scores
    the wrong water.
    """
    tree = cKDTree(_unit_sphere_coordinates(source_latitude, source_longitude))
    target_latitude_grid, target_longitude_grid = numpy.meshgrid(target_latitude, target_longitude, indexing="ij")
    chord_lengths, flat_indices = tree.query(_unit_sphere_coordinates(target_latitude_grid, target_longitude_grid), k=1)
    distance_kilometres = _great_circle_kilometres(chord_lengths).reshape(target_latitude_grid.shape)
    flat_indices = flat_indices.reshape(target_latitude_grid.shape)
    usable = source_ocean_mask.ravel()[flat_indices] & (distance_kilometres <= maximum_distance_kilometres)
    return NearestNeighbourMapping(
        source_flat_indices=flat_indices,
        distance_kilometres=distance_kilometres,
        usable=usable,
        target_latitude=numpy.asarray(target_latitude),
        target_longitude=numpy.asarray(target_longitude),
    )


@dataclass(frozen=True)
class ScatteredNearestNeighbourMapping:
    """Which source cell each scattered point takes, and whether that cell is usable.

    The observation counterpart of :class:`NearestNeighbourMapping`: the points are the
    Class IV observation positions rather than the cells of a regular grid, so everything is
    one-dimensional and there are no target axes to carry.
    """

    source_flat_indices: numpy.ndarray
    distance_kilometres: numpy.ndarray
    usable: numpy.ndarray

    @property
    def usable_fraction(self) -> float:
        return float(self.usable.mean())


def nearest_neighbour_at_points(
    source_latitude: numpy.ndarray,
    source_longitude: numpy.ndarray,
    source_ocean_mask: numpy.ndarray,
    point_latitude: numpy.ndarray,
    point_longitude: numpy.ndarray,
    *,
    maximum_distance_kilometres: float,
) -> ScatteredNearestNeighbourMapping:
    """Map scattered points onto their nearest source cell, the same way a target grid is.

    Land is kept in the tree and marked unusable afterwards, exactly as in
    :func:`nearest_neighbour_mapping`: an observation whose nearest model cell is land is one
    the model does not simulate, and giving it the nearest wet cell instead would score the
    wrong side of the coast rather than report a gap.
    """
    tree = cKDTree(_unit_sphere_coordinates(source_latitude, source_longitude))
    chord_lengths, flat_indices = tree.query(_unit_sphere_coordinates(point_latitude, point_longitude), k=1)
    distance_kilometres = _great_circle_kilometres(chord_lengths)
    return ScatteredNearestNeighbourMapping(
        source_flat_indices=flat_indices,
        distance_kilometres=distance_kilometres,
        usable=numpy.asarray(source_ocean_mask, dtype=bool).ravel()[flat_indices]
        & (distance_kilometres <= maximum_distance_kilometres),
    )


def sample_onto_target_grid(
    source_fields: numpy.ndarray,
    mapping: NearestNeighbourMapping,
    *,
    leading_dimension: str | None = None,
    leading_coordinate: numpy.ndarray | None = None,
) -> xarray.DataArray:
    """Sample one source field, or a stack of them, onto the target grid.

    ``source_fields`` is either ``(source_y, source_x)`` or ``(n, source_y, source_x)``; in the
    stacked case ``leading_dimension`` names the extra dimension. Unusable target cells come
    back as ``nan`` so that every downstream weighted mean drops them.
    """
    stacked = source_fields.reshape(-1, source_fields.shape[-2] * source_fields.shape[-1])
    sampled = stacked[:, mapping.source_flat_indices].astype("float64")
    sampled[:, ~mapping.usable] = numpy.nan

    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    coordinates = {latitude_key: mapping.target_latitude, longitude_key: mapping.target_longitude}
    if leading_dimension is None:
        return xarray.DataArray(sampled[0], dims=[latitude_key, longitude_key], coords=coordinates)
    return xarray.DataArray(
        sampled.reshape((-1, *mapping.usable.shape)),
        dims=[leading_dimension, latitude_key, longitude_key],
        coords={leading_dimension: leading_coordinate, **coordinates},
    )
