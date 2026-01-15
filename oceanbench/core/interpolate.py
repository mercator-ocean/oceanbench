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

    new_lat = numpy.arange(numpy.floor(data[latitude_dim].min()), numpy.ceil(data[latitude_dim].max()), 1)
    new_lon = numpy.arange(numpy.floor(data[longitude_dim].min()), numpy.ceil(data[longitude_dim].max()), 1)

    data = data.chunk({latitude_dim: -1, longitude_dim: -1})

    return data.interp(**{latitude_dim: new_lat, longitude_dim: new_lon})
