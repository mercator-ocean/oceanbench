# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from enum import Enum

from oceanbench.core.climate_forecast_standard_names import (
    StandardDimension,
    StandardVariable,
)


class Variable(Enum):
    SEA_SURFACE_HEIGHT_ABOVE_GEOID = StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_GEOID
    SEA_WATER_POTENTIAL_TEMPERATURE = StandardVariable.SEA_WATER_POTENTIAL_TEMPERATURE
    SEA_WATER_SALINITY = StandardVariable.SEA_WATER_SALINITY
    NORTHWARD_SEA_WATER_VELOCITY = StandardVariable.NORTHWARD_SEA_WATER_VELOCITY
    EASTWARD_SEA_WATER_VELOCITY = StandardVariable.EASTWARD_SEA_WATER_VELOCITY
    MIXED_LAYER_DEPTH = StandardVariable.MIXED_LAYER_THICKNESS
    GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY = StandardVariable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY
    GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY = StandardVariable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY

    def key(self) -> str:
        return self.value.value


class Dimension(Enum):
    DEPTH = StandardDimension.DEPTH
    LATITUDE = StandardDimension.LATITUDE
    LONGITUDE = StandardDimension.LONGITUDE
    TIME = StandardDimension.TIME
    LEAD_DAY_INDEX = "lead_day_index"
    FIRST_DAY_DATETIME = "first_day_datetime"

    def key(self) -> str:
        return self.value.value if isinstance(self.value, StandardDimension) else self.value


class DepthLevel(Enum):
    SURFACE = 4.940250e-01
    MINUS_50_METERS = 4.737369e01
    MINUS_200_METERS = 2.224752e02
    MINUS_550_METERS = 5.410889e02
