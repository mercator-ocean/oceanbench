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


#: Standard names of quantities the benchmark already scores under another name.
#:
#: A NEMO store describes its fields in the vocabulary of the model, so a GloEns week calls its
#: salinity ``sea_water_practical_salinity`` and its sea surface height
#: ``dynamic_sea_surface_height_above_geoid``. Those are the quantities scored here as
#: ``sea_water_salinity`` and ``sea_surface_height_above_geoid``, and giving them those names as
#: the dataset is renamed is what lets every path downstream ask for one name and find the field.
#: The velocity components are not listed: they run along the axes of the model grid and only
#: become eastward and northward once they are turned.
STANDARD_NAME_ALIASES: dict[str, str] = {
    "sea_water_practical_salinity": StandardVariable.SEA_WATER_SALINITY.value,
    "dynamic_sea_surface_height_above_geoid": StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.value,
}


def rename_dataset_with_standard_names(
    dataset: xarray.Dataset,
) -> xarray.Dataset:
    mapping = {
        variable_name: STANDARD_NAME_ALIASES.get(
            dataset[variable_name].standard_name, dataset[variable_name].standard_name
        )
        for variable_name in dataset.variables
        if hasattr(dataset[variable_name], "standard_name")
    }
    return dataset.rename(mapping)
