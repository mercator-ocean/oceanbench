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


class StandardVariable(Enum):
    SEA_SURFACE_HEIGHT_ABOVE_GEOID = "sea_surface_height_above_geoid"
    SEA_WATER_POTENTIAL_TEMPERATURE = "sea_water_potential_temperature"
    SEA_WATER_SALINITY = "sea_water_salinity"
    NORTHWARD_SEA_WATER_VELOCITY = "northward_sea_water_velocity"
    EASTWARD_SEA_WATER_VELOCITY = "eastward_sea_water_velocity"
    MIXED_LAYER_THICKNESS = "ocean_mixed_layer_thickness"
    GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY = "geostrophic_northward_sea_water_velocity"
    GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY = "geostrophic_eastward_sea_water_velocity"


def _rename_mapping_from_standard_name_attributes(dataset: xarray.Dataset) -> dict[str, str]:
    return {
        variable_name: dataset[variable_name].standard_name
        for variable_name in dataset.variables
        if hasattr(dataset[variable_name], "standard_name")
    }


def _rename_mapping_from_common_aliases(dataset: xarray.Dataset) -> dict[str, str]:
    common_aliases_to_standard_names = {
        "lat": StandardDimension.LATITUDE.value,
        "lon": StandardDimension.LONGITUDE.value,
        "zos": StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.value,
        "thetao": StandardVariable.SEA_WATER_POTENTIAL_TEMPERATURE.value,
        "so": StandardVariable.SEA_WATER_SALINITY.value,
        "uo": StandardVariable.EASTWARD_SEA_WATER_VELOCITY.value,
        "vo": StandardVariable.NORTHWARD_SEA_WATER_VELOCITY.value,
    }
    return {
        alias_name: standard_name
        for alias_name, standard_name in common_aliases_to_standard_names.items()
        if alias_name in dataset.variables and standard_name not in dataset.variables
    }


def rename_dataset_with_standard_names(
    dataset: xarray.Dataset,
) -> xarray.Dataset:
    mapping = _rename_mapping_from_standard_name_attributes(dataset)
    mapping.update(_rename_mapping_from_common_aliases(dataset))
    return dataset.rename(mapping)
