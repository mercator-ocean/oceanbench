# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.geostrophic_currents import (
    EARTH_RADIUS_METERS,
    EARTH_ROTATION_RATE,
    GRAVITATIONAL_ACCELERATION,
    _compute_geostrophic_currents,
)


def _dataset_from_sea_surface_height(
    sea_surface_height: numpy.ndarray,
    longitudes: numpy.ndarray,
    latitudes: numpy.ndarray,
) -> xarray.Dataset:
    return xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                sea_surface_height[numpy.newaxis, numpy.newaxis],
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def _sea_surface_height_dataset(
    longitudes: numpy.ndarray,
    latitudes: numpy.ndarray = numpy.arange(10.0, 31.0, 1.0),
) -> xarray.Dataset:
    # Shifted by 45 degrees so the field is curved at the dateline, where a one-sided difference shows
    sea_surface_height = numpy.broadcast_to(
        numpy.sin(numpy.deg2rad(longitudes + 45)), (latitudes.size, longitudes.size)
    ).copy()
    return _dataset_from_sea_surface_height(sea_surface_height, longitudes, latitudes)


def test_northward_geostrophic_velocity_is_periodic_on_a_global_grid() -> None:
    longitudes = numpy.arange(-179.5, 180.0, 1.0)
    dataset = _sea_surface_height_dataset(longitudes)

    northward_velocity = _compute_geostrophic_currents(dataset)[
        Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key()
    ].values[0, 0]

    latitude_radian = numpy.deg2rad(dataset[Dimension.LATITUDE.key()].values)[:, numpy.newaxis]
    coriolis_parameter = 2 * EARTH_ROTATION_RATE * numpy.sin(latitude_radian)
    zonal_derivative = numpy.cos(numpy.deg2rad(longitudes + 45)) / (EARTH_RADIUS_METERS * numpy.cos(latitude_radian))
    expected_velocity = GRAVITATIONAL_ACCELERATION / coriolis_parameter * zonal_derivative
    numpy.testing.assert_allclose(northward_velocity[:, [0, -1]], expected_velocity[:, [0, -1]], rtol=1e-4)


def test_geostrophic_currents_on_a_regional_grid_keep_one_sided_edges() -> None:
    # A sea surface height linear in longitude and latitude has a constant gradient, which centred and
    # one-sided differences both recover exactly, so every column up to the edges matches the closed form.
    # Wrapping the regional edges around as on a global grid would break the first and last columns.
    longitudes = numpy.arange(-20.0, 20.5, 0.5)
    latitudes = numpy.arange(10.0, 31.0, 1.0)
    zonal_slope_meters_per_degree = 0.01
    meridional_slope_meters_per_degree = 0.02
    sea_surface_height = (
        zonal_slope_meters_per_degree * longitudes[numpy.newaxis, :]
        + meridional_slope_meters_per_degree * latitudes[:, numpy.newaxis]
    )
    dataset = _dataset_from_sea_surface_height(sea_surface_height, longitudes, latitudes)

    geostrophic_currents = _compute_geostrophic_currents(dataset)

    latitude_radian = numpy.deg2rad(latitudes)[:, numpy.newaxis]
    coriolis_parameter = 2 * EARTH_ROTATION_RATE * numpy.sin(latitude_radian)
    meters_per_degree = numpy.deg2rad(1.0) * EARTH_RADIUS_METERS
    expected_northward_velocity = (
        GRAVITATIONAL_ACCELERATION
        / coriolis_parameter
        * zonal_slope_meters_per_degree
        / (meters_per_degree * numpy.cos(latitude_radian))
    )
    expected_eastward_velocity = (
        -GRAVITATIONAL_ACCELERATION / coriolis_parameter * meridional_slope_meters_per_degree / meters_per_degree
    )
    numpy.testing.assert_allclose(
        geostrophic_currents[Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key()].values[0, 0],
        numpy.broadcast_to(expected_northward_velocity, sea_surface_height.shape),
        rtol=1e-10,
    )
    numpy.testing.assert_allclose(
        geostrophic_currents[Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key()].values[0, 0],
        numpy.broadcast_to(expected_eastward_velocity, sea_surface_height.shape),
        rtol=1e-10,
    )


def test_geostrophic_currents_exclude_the_five_degree_equatorial_band() -> None:
    latitudes = numpy.arange(-10.0, 10.5, 0.5)
    dataset = _sea_surface_height_dataset(numpy.arange(-20.0, 20.5, 0.5), latitudes)

    geostrophic_latitudes = _compute_geostrophic_currents(dataset)[Dimension.LATITUDE.key()].values

    assert geostrophic_latitudes.tolist() == latitudes[numpy.abs(latitudes) >= 5.0].tolist()
