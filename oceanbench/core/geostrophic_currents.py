# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray
import dask

from oceanbench.core.climate_forecast_standard_names import (
    remane_dataset_with_standard_names,
)
from oceanbench.core.dataset_utils import (
    Dimension,
    Variable,
)


def compute_geostrophic_currents(dataset: xarray.Dataset) -> xarray.Dataset:
    return _compute_geostrophic_currents(_harmonise_dataset(dataset))


def _harmonise_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    return remane_dataset_with_standard_names(dataset)


def _compute_geostrophic_currents(dataset: xarray.Dataset) -> xarray.Dataset:
    sea_surface_height = dataset[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()].chunk(
        {Dimension.FIRST_DAY_DATETIME.key(): 2}
    )
    latitude = dataset[Dimension.LATITUDE.key()].values
    longitude = dataset[Dimension.LONGITUDE.key()].values

    latitude_radian = numpy.deg2rad(latitude)

    # coriolis
    omega = 7.2921e-5
    f = 2 * omega * numpy.sin(latitude_radian)
    f_safe = numpy.where(numpy.abs(f) < 1e-10, numpy.nan, f)
    R = 6371000

    # Compute grid spacing
    dx = numpy.gradient(longitude) * (numpy.pi / 180) * R * numpy.cos(latitude_radian[:, numpy.newaxis])
    dy = numpy.gradient(latitude)[:, numpy.newaxis] * (numpy.pi / 180) * R

    dssh_dx = dask.array.gradient(sea_surface_height, axis=-1) / dx
    dssh_dy = dask.array.gradient(sea_surface_height, axis=-2) / dy

    g = 9.81  # gravity

    eastward_geostrophic_velocity = -g / f_safe[:, numpy.newaxis] * dssh_dy
    northward_geostrophic_velocity = g / f_safe[:, numpy.newaxis] * dssh_dx

    dimensions = (
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    )

    geostrophic_currents = xarray.Dataset(
        data_vars={
            Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key(): (
                dimensions,
                eastward_geostrophic_velocity,
            ),
            Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key(): (
                dimensions,
                northward_geostrophic_velocity,
            ),
        },
        coords=dataset.coords,
    )

    return _exclude_equator(geostrophic_currents)


def _exclude_equator(dataset: xarray.Dataset) -> xarray.Dataset:
    latitude = dataset[Dimension.LATITUDE.key()]
    not_on_equator = (latitude < -0.5) | (latitude > 0.5)
    return dataset.isel({Dimension.LATITUDE.key(): not_on_equator})
