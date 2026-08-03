# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import dask
import numpy
import xarray

from oceanbench.core import geostrophic_currents
from oceanbench.core.dataset_utils import Dimension, Variable

_DIMENSIONS = (
    Dimension.FIRST_DAY_DATETIME.key(),
    Dimension.LEAD_DAY_INDEX.key(),
    Dimension.LATITUDE.key(),
    Dimension.LONGITUDE.key(),
)


def _dataset() -> xarray.Dataset:
    latitudes = numpy.linspace(-30.0, 30.0, 13)
    longitudes = numpy.linspace(0.0, 20.0, 11)
    shape = (3, 4, len(latitudes), len(longitudes))
    generator = numpy.random.default_rng(20260801)
    sea_surface_height = generator.normal(0.0, 0.2, shape).astype("float32")
    return xarray.Dataset(
        data_vars={Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (_DIMENSIONS, sea_surface_height)},
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(
                ["2024-01-03", "2024-01-10", "2024-01-17"], dtype="datetime64[ns]"
            ),
            Dimension.LEAD_DAY_INDEX.key(): [0, 1, 2, 3],
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def _previous_implementation(dataset: xarray.Dataset) -> xarray.Dataset:
    """The pre-refactoring computation, kept verbatim as the numerical reference."""
    sea_surface_height = dataset[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()].chunk(
        {Dimension.FIRST_DAY_DATETIME.key(): 2}
    )
    latitude = dataset[Dimension.LATITUDE.key()].values
    longitude = dataset[Dimension.LONGITUDE.key()].values

    latitude_radian = numpy.deg2rad(latitude)
    omega = 7.2921e-5
    f = 2 * omega * numpy.sin(latitude_radian)
    f_safe = numpy.where(numpy.abs(f) < 1e-10, numpy.nan, f)
    R = 6371000

    dx = numpy.gradient(longitude) * (numpy.pi / 180) * R * numpy.cos(latitude_radian[:, numpy.newaxis])
    dy = numpy.gradient(latitude)[:, numpy.newaxis] * (numpy.pi / 180) * R

    dssh_dx = dask.array.gradient(sea_surface_height, axis=-1) / dx
    dssh_dy = dask.array.gradient(sea_surface_height, axis=-2) / dy

    g = 9.81

    return xarray.Dataset(
        data_vars={
            Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key(): (
                _DIMENSIONS,
                -g / f_safe[:, numpy.newaxis] * dssh_dy,
            ),
            Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key(): (
                _DIMENSIONS,
                g / f_safe[:, numpy.newaxis] * dssh_dx,
            ),
        },
        coords=dataset.coords,
    )


def test_bounded_blocks_do_not_change_the_geostrophic_currents() -> None:
    dataset = _dataset()

    computed = geostrophic_currents.compute_geostrophic_currents(dataset)
    reference = geostrophic_currents._exclude_equator(_previous_implementation(dataset))

    for variable in (
        Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key(),
        Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key(),
    ):
        numpy.testing.assert_array_equal(computed[variable].values, reference[variable].values)


def test_geostrophic_blocks_are_bounded_per_start_and_lead_day() -> None:
    computed = geostrophic_currents.compute_geostrophic_currents(_dataset())

    chunk_sizes = computed[Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key()].chunksizes
    assert chunk_sizes[Dimension.FIRST_DAY_DATETIME.key()] == (1, 1, 1)
    assert chunk_sizes[Dimension.LEAD_DAY_INDEX.key()] == (1, 1, 1, 1)
