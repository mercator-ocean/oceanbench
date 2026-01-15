# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import xarray
import numpy
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names, StandardDimension


def interpolate_1_degree(data: xarray.Dataset) -> xarray.Dataset:
    data = rename_dataset_with_standard_names(data)

    latitude_dimension = StandardDimension.LATITUDE.value
    longitude_dimension = StandardDimension.LONGITUDE.value
    time_dimension = StandardDimension.TIME.value
    depth_dimension = StandardDimension.DEPTH.value

    latitude_minimum = data[latitude_dimension].min().values
    latitude_maximum = data[latitude_dimension].max().values
    longitude_minimum = data[longitude_dimension].min().values
    longitude_maximum = data[longitude_dimension].max().values

    latitude_start = numpy.ceil(latitude_minimum - 0.5) + 0.5
    latitude_end = numpy.floor(latitude_maximum + 0.5) - 0.5
    longitude_start = numpy.ceil(longitude_minimum - 0.5) + 0.5
    longitude_end = numpy.floor(longitude_maximum + 0.5) - 0.5

    new_latitude = numpy.arange(latitude_start, latitude_end + 1, 1.0)
    new_longitude = numpy.arange(longitude_start, longitude_end + 1, 1.0)

    chunk_dimensions = {latitude_dimension: -1, longitude_dimension: -1}
    if time_dimension in data.dims:
        chunk_dimensions[time_dimension] = 1
    if depth_dimension in data.dims:
        chunk_dimensions[depth_dimension] = 1

    data = data.chunk(chunk_dimensions)

    return data.interp(**{latitude_dimension: new_latitude, longitude_dimension: new_longitude})
