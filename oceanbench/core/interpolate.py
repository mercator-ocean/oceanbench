# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import xarray
import numpy
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names, StandardDimension
from oceanbench.core.dataset_source import get_dataset_source, with_dataset_source
from oceanbench.core.dataset_utils import Dimension


def one_degree_target_grid(data: xarray.Dataset) -> tuple[numpy.ndarray, numpy.ndarray]:
    latitude_dimension = StandardDimension.LATITUDE.value
    longitude_dimension = StandardDimension.LONGITUDE.value

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
    return new_latitude, new_longitude


def apply_one_degree_interpolation(
    data: xarray.Dataset,
    new_latitude: numpy.ndarray,
    new_longitude: numpy.ndarray,
) -> xarray.Dataset:
    data = rename_dataset_with_standard_names(data)

    latitude_dimension = StandardDimension.LATITUDE.value
    longitude_dimension = StandardDimension.LONGITUDE.value
    time_dimension = StandardDimension.TIME.value
    depth_dimension = StandardDimension.DEPTH.value

    chunk_dimensions = {latitude_dimension: -1, longitude_dimension: -1}
    if time_dimension in data.dims:
        chunk_dimensions[time_dimension] = 1
    if depth_dimension in data.dims:
        chunk_dimensions[depth_dimension] = 1
    for forecast_dimension in (Dimension.FIRST_DAY_DATETIME.key(), Dimension.LEAD_DAY_INDEX.key()):
        if forecast_dimension in data.dims:
            chunk_dimensions[forecast_dimension] = 1

    data = data.chunk(chunk_dimensions)
    return data.interp(**{latitude_dimension: new_latitude, longitude_dimension: new_longitude})


def interpolate_1_degree(data: xarray.Dataset) -> xarray.Dataset:
    from oceanbench.core.multistore import (
        MultiStoreConcatRecipe,
        OneDegreeInterpolation,
        attach_multistore_recipe,
        get_multistore_recipe,
    )
    from dataclasses import replace

    standardised = rename_dataset_with_standard_names(data)
    new_latitude, new_longitude = one_degree_target_grid(standardised)
    interpolated = apply_one_degree_interpolation(data, new_latitude, new_longitude)

    base_recipe = get_multistore_recipe(data)
    if base_recipe is not None and base_recipe.interpolation is None:
        interpolated = attach_multistore_recipe(
            interpolated,
            replace(
                base_recipe,
                interpolation=OneDegreeInterpolation(
                    latitude=tuple(float(value) for value in new_latitude),
                    longitude=tuple(float(value) for value in new_longitude),
                ),
            ),
        )

    dataset_source = get_dataset_source(standardised)
    if dataset_source is None:
        return interpolated

    return with_dataset_source(
        interpolated,
        kind=dataset_source.kind,
        name=dataset_source.name,
        resolution="one_degree",
    )
