# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from enum import Enum
import xarray
from typing import Optional


class StandardDimension(Enum):
    DEPTH = "depth"
    TIME = "time"
    LATITUDE = "latitude"
    LONGITUDE = "longitude"

    def dimension_name_from_dataset_standard_names(self, dataset: xarray.Dataset) -> Optional[str]:
        return _get_variable_name_from_standard_name(dataset, self.value)


class StandardVariable(Enum):
    HEIGHT = "sea_surface_height_above_geoid"
    TEMPERATURE = "sea_water_potential_temperature"
    SALINITY = "sea_water_salinity"
    NORTHWARD_VELOCITY = "northward_sea_water_velocity"
    EASTWARD_VELOCITY = "eastward_sea_water_velocity"

    def variable_name_from_dataset_standard_names(self, dataset: xarray.Dataset) -> str:
        return _get_variable_name_from_standard_name(dataset, self.value)


def _get_variable_name_from_standard_name(dataset: xarray.Dataset, standard_name: str) -> str:
    for variable_name in dataset.variables:
        if hasattr(dataset[variable_name], "standard_name") and dataset[variable_name].standard_name == standard_name:
            return str(variable_name)
    raise Exception(f"No variable with standard name {standard_name} found in dataset")
