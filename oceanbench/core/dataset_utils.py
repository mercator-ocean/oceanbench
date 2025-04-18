# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from enum import Enum
from xarray import Dataset
import numpy
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

    def dimension_name_from_dataset(self, dataset):
        """
        Get the actual dimension name in the dataset corresponding to this Dimension enum member.
        """
        mapping = {
            Dimension.DEPTH: "depth",
            Dimension.LATITUDE: "latitude",
            Dimension.LONGITUDE: "longitude",
            Dimension.TIME: "time",
            Dimension.LEAD_DAY_INDEX: "lead_day_index",
            Dimension.FIRST_DAY_DATETIME: "first_day_datetime",
        }
        return mapping[self]


class DepthLevel(Enum):
    SURFACE = 4.940250e-01
    MINUS_50_METERS = 4.737369e01
    MINUS_200_METERS = 2.224752e02
    MINUS_550_METERS = 5.410889e02


def get_dimension(dataset: Dataset, dimension: Dimension):
    """
    Get the dimension name in the dataset corresponding to the given Dimension enum member.
    Handles common naming variations (lat/latitude, lon/longitude).
    """
    dimension_name = dimension.dimension_name_from_dataset(dataset)

    # If the dimension exists directly, return it
    if dimension_name in dataset:
        return dataset[dimension_name]

    # Mapping of standard names to their common variants
    common_variants = {
        "latitude": "lat",
        "longitude": "lon",
        "lat": "latitude",
        "lon": "longitude",
    }

    # Try the common variant
    if dimension_name in common_variants:
        variant = common_variants[dimension_name]
        if variant in dataset:
            return dataset[variant]

    # If no variant was found, raise the original error
    raise KeyError(
        f"No variable named '{dimension_name}' or common variants. "
        f"Variables on the dataset include {list(dataset.variables.keys())}"
    )


def _select_closest_depths(dataset: Dataset, target_depths: list) -> Dataset:
    """
    Selects the depth levels from the dataset that are closest to the target depths.

    Args:
        dataset: Dataset containing a 'depth' dimension
        target_depths: List of target depths to approximate

    Returns:
        Dataset with selected depths
    """
    available_depths = dataset["depth"].values
    selected_indices = []

    for target_depth in target_depths:
        # Find the index of the closest depth
        closest_idx = numpy.argmin(numpy.abs(available_depths - target_depth))
        selected_indices.append(closest_idx)

    # Select the corresponding depths
    selected_dataset = dataset.isel(depth=selected_indices)

    # Optional: rename depths with target values for consistency
    # selected_dataset = selected_dataset.assign_coords({'depth': target_depths})

    return selected_dataset
