# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import dask.array
import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.geostrophic_currents import (
    EARTH_RADIUS_METERS,
    EARTH_ROTATION_RATE,
    GRAVITATIONAL_ACCELERATION,
    _compute_geostrophic_currents,
)


def _sea_surface_height_dataset(
    longitudes: numpy.ndarray,
    latitudes: numpy.ndarray = numpy.arange(10.0, 31.0, 1.0),
) -> xarray.Dataset:
    # Shifted by 45 degrees so the field is curved at the dateline, where a one-sided difference shows
    sea_surface_height = numpy.broadcast_to(
        numpy.sin(numpy.deg2rad(longitudes + 45)), (1, 1, latitudes.size, longitudes.size)
    ).copy()
    return xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                sea_surface_height,
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


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
    longitudes = numpy.arange(-20.0, 20.5, 0.5)
    dataset = _sea_surface_height_dataset(longitudes)

    northward_velocity = _compute_geostrophic_currents(dataset)[
        Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key()
    ].values

    latitude_radian = numpy.deg2rad(dataset[Dimension.LATITUDE.key()].values)
    coriolis_parameter = 2 * EARTH_ROTATION_RATE * numpy.sin(latitude_radian)
    zonal_grid_spacing = (
        numpy.gradient(longitudes)
        * (numpy.pi / 180)
        * EARTH_RADIUS_METERS
        * numpy.cos(latitude_radian[:, numpy.newaxis])
    )
    sea_surface_height = dataset[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()].chunk(
        {Dimension.FIRST_DAY_DATETIME.key(): 2}
    )
    expected_velocity = (
        GRAVITATIONAL_ACCELERATION
        / coriolis_parameter[:, numpy.newaxis]
        * (dask.array.gradient(sea_surface_height, axis=-1) / zonal_grid_spacing)
    )
    numpy.testing.assert_array_equal(northward_velocity, numpy.asarray(expected_velocity))


def test_geostrophic_currents_exclude_the_five_degree_equatorial_band() -> None:
    latitudes = numpy.arange(-10.0, 10.5, 0.5)
    dataset = _sea_surface_height_dataset(numpy.arange(-20.0, 20.5, 0.5), latitudes)

    geostrophic_latitudes = _compute_geostrophic_currents(dataset)[Dimension.LATITUDE.key()].values

    assert geostrophic_latitudes.tolist() == latitudes[numpy.abs(latitudes) >= 5.0].tolist()
