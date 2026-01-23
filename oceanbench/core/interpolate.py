# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import xarray
import numpy
from oceanbench.core.climate_forecast_standard_names import rename_dimensions_with_standard_names, StandardDimension


def interpolate_1deg(data: xarray.Dataset) -> xarray.Dataset:
    data = rename_dimensions_with_standard_names(data)

    latitude_dim = StandardDimension.LATITUDE.value
    longitude_dim = StandardDimension.LONGITUDE.value

    new_lat = numpy.arange(-89.5, 90.0, 1.0)  # 180 points: -89.5, -88.5, ..., 88.5, 89.5
    new_lon = numpy.arange(-179.5, 180.0, 1.0)  # 360 points: -179.5, -178.5, ..., 178.5, 179.5

    data = data.chunk({latitude_dim: -1, longitude_dim: -1})

    return data.interp(**{latitude_dim: new_lat, longitude_dim: new_lon})
