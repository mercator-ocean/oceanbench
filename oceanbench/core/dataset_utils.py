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
    MINUS_100_METERS = 9.232607e01
    MINUS_200_METERS = 2.224752e02
    MINUS_300_METERS = 3.181274e02
    MINUS_500_METERS = 5.410889e02


VARIABLE_LABELS: dict[str, str] = {
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): "surface height",
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): "temperature",
    Variable.SEA_WATER_SALINITY.key(): "salinity",
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): "northward velocity",
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): "eastward velocity",
    Variable.MIXED_LAYER_DEPTH.key(): "mixed layer depth",
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key(): "northward geostrophic velocity",
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key(): "eastward geostrophic velocity",
}

DEPTH_LABELS: dict[DepthLevel, str] = {
    DepthLevel.SURFACE: "surface",
    DepthLevel.MINUS_50_METERS: "50m",
    DepthLevel.MINUS_100_METERS: "100m",
    DepthLevel.MINUS_200_METERS: "200m",
    DepthLevel.MINUS_300_METERS: "300m",
    DepthLevel.MINUS_500_METERS: "500m",
}

LEAD_DAYS_COUNT = 10

# For class IV validation :

DEPTH_BINS_DEFAULT: dict[str, tuple[float, float]] = {
    "surface": (-1, 1),
    "0-5m": (0, 5),
    "5-100m": (5, 100),
    "100-300m": (100, 300),
    "300-600m": (300, 600),
}

DEPTH_BINS_BY_VARIABLE: dict[str, dict[str, tuple[float, float]]] = {
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): {
        "SST": (-1, 1),
        "0-5m": (1, 5),
        "5-100m": (5, 100),
        "100-300m": (100, 300),
        "300-600m": (300, 600),
    },
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): {"15m": (10, 20)},
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): {"15m": (10, 20)},
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): {"surface": (-5, 5)},
}

VARIABLE_DISPLAY_ORDER: dict[str, int] = {
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): 0,
    Variable.SEA_WATER_SALINITY.key(): 1,
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): 2,
    Variable.EASTWARD_SEA_WATER_VELOCITY.key(): 3,
    Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): 4,
}

DEPTH_BIN_DISPLAY_ORDER: dict[str, int] = {
    name: index for index, name in enumerate(["SST", "surface", "0-5m", "5-100m", "100-300m", "300-600m", "15m"])
}
