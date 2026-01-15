# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from enum import Enum
import xarray


class StandardDimension(Enum):
    DEPTH = "depth"
    TIME = "time"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"


def rename_dimensions_with_standard_names(
    dataset: xarray.Dataset,
) -> xarray.Dataset:
    mapping = {
        dim_name: dataset[dim_name].standard_name
        for dim_name in dataset.dims
        if dim_name in dataset.coords and hasattr(dataset[dim_name], "standard_name")
    }
    return dataset.rename(mapping)


class StandardVariable(Enum):
    SEA_SURFACE_HEIGHT_ABOVE_GEOID = "sea_surface_height_above_geoid"
    SEA_WATER_POTENTIAL_TEMPERATURE = "sea_water_potential_temperature"
    SEA_WATER_SALINITY = "sea_water_salinity"
    NORTHWARD_SEA_WATER_VELOCITY = "northward_sea_water_velocity"
    EASTWARD_SEA_WATER_VELOCITY = "eastward_sea_water_velocity"
    MIXED_LAYER_THICKNESS = "ocean_mixed_layer_thickness"
    GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY = "geostrophic_northward_sea_water_velocity"
    GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY = "geostrophic_eastward_sea_water_velocity"


def rename_dataset_with_standard_names(
    dataset: xarray.Dataset,
) -> xarray.Dataset:
    mapping = {
        variable_name: dataset[variable_name].standard_name
        for variable_name in dataset.variables
        if hasattr(dataset[variable_name], "standard_name")
    }

    return dataset.rename(mapping)
