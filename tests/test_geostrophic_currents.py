# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import dask.array
import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.geostrophic_currents import _compute_geostrophic_currents

GRAVITY = 9.81
EARTH_RADIUS = 6371000
OMEGA = 7.2921e-5


def _sea_surface_height_dataset(longitudes: numpy.ndarray) -> xarray.Dataset:
    latitudes = numpy.arange(10.0, 31.0, 1.0)
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
    coriolis = 2 * OMEGA * numpy.sin(latitude_radian)
    zonal_derivative = numpy.cos(numpy.deg2rad(longitudes + 45)) / (EARTH_RADIUS * numpy.cos(latitude_radian))
    expected_velocity = GRAVITY / coriolis * zonal_derivative
    numpy.testing.assert_allclose(northward_velocity[:, [0, -1]], expected_velocity[:, [0, -1]], rtol=1e-4)


def test_geostrophic_currents_on_a_regional_grid_keep_one_sided_edges() -> None:
    longitudes = numpy.arange(-20.0, 20.5, 0.5)
    dataset = _sea_surface_height_dataset(longitudes)

    northward_velocity = _compute_geostrophic_currents(dataset)[
        Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key()
    ].values

    latitude_radian = numpy.deg2rad(dataset[Dimension.LATITUDE.key()].values)
    coriolis = 2 * OMEGA * numpy.sin(latitude_radian)
    dx = numpy.gradient(longitudes) * (numpy.pi / 180) * EARTH_RADIUS * numpy.cos(latitude_radian[:, numpy.newaxis])
    sea_surface_height = dataset[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()].chunk(
        {Dimension.FIRST_DAY_DATETIME.key(): 2}
    )
    expected_velocity = GRAVITY / coriolis[:, numpy.newaxis] * (dask.array.gradient(sea_surface_height, axis=-1) / dx)
    numpy.testing.assert_array_equal(northward_velocity, numpy.asarray(expected_velocity))
