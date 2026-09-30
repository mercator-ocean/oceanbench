# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray
import dask

from oceanbench.core.climate_forecast_standard_names import (
    rename_dataset_with_standard_names,
)
from oceanbench.core.dataset_utils import (
    Dimension,
    Variable,
    is_global_longitude_grid,
)

EQUATORIAL_BAND_HALF_WIDTH_DEGREES = 5.0
EARTH_ROTATION_RATE = 7.2921e-5
EARTH_RADIUS_METERS = 6371000
GRAVITATIONAL_ACCELERATION = 9.81


def compute_geostrophic_currents(dataset: xarray.Dataset) -> xarray.Dataset:
    return _compute_geostrophic_currents(_harmonise_dataset(dataset))


def _harmonise_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    return rename_dataset_with_standard_names(dataset)


def _compute_geostrophic_currents(dataset: xarray.Dataset) -> xarray.Dataset:
    sea_surface_height = dataset[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()].chunk(
        {Dimension.FIRST_DAY_DATETIME.key(): 2}
    )
    latitude = dataset[Dimension.LATITUDE.key()].values
    longitude = dataset[Dimension.LONGITUDE.key()].values

    latitude_radian = numpy.deg2rad(latitude)

    coriolis_parameter = 2 * EARTH_ROTATION_RATE * numpy.sin(latitude_radian)
    safe_coriolis_parameter = numpy.where(numpy.abs(coriolis_parameter) < 1e-10, numpy.nan, coriolis_parameter)

    zonal_grid_spacing = (
        numpy.gradient(longitude)
        * (numpy.pi / 180)
        * EARTH_RADIUS_METERS
        * numpy.cos(latitude_radian[:, numpy.newaxis])
    )
    meridional_grid_spacing = numpy.gradient(latitude)[:, numpy.newaxis] * (numpy.pi / 180) * EARTH_RADIUS_METERS

    if is_global_longitude_grid(longitude):
        eastern_neighbour = sea_surface_height.roll({Dimension.LONGITUDE.key(): -1})
        western_neighbour = sea_surface_height.roll({Dimension.LONGITUDE.key(): 1})
        dssh_dx = ((eastern_neighbour - western_neighbour) / 2).data / zonal_grid_spacing
    else:
        dssh_dx = dask.array.gradient(sea_surface_height, axis=-1) / zonal_grid_spacing
    dssh_dy = dask.array.gradient(sea_surface_height, axis=-2) / meridional_grid_spacing

    eastward_geostrophic_velocity = -GRAVITATIONAL_ACCELERATION / safe_coriolis_parameter[:, numpy.newaxis] * dssh_dy
    northward_geostrophic_velocity = GRAVITATIONAL_ACCELERATION / safe_coriolis_parameter[:, numpy.newaxis] * dssh_dx

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

    return _exclude_equatorial_band(geostrophic_currents)


def _exclude_equatorial_band(dataset: xarray.Dataset) -> xarray.Dataset:
    latitude = dataset[Dimension.LATITUDE.key()]
    outside_equatorial_band = abs(latitude) >= EQUATORIAL_BAND_HALF_WIDTH_DEGREES
    return dataset.isel({Dimension.LATITUDE.key(): outside_equatorial_band})
