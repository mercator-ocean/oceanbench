# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
from xarray import Dataset

from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.climate_forecast_standard_names import (
    rename_dataset_with_standard_names,
)

# Known grid spacings (in degrees)
ONE_DEGREE_SPACING = 1.0
QUARTER_DEGREE_SPACING = 0.25
TWELFTH_DEGREE_SPACING = 1.0 / 12.0  # ~0.0833


def get_dataset_resolution(dataset: Dataset) -> str:
    """
    Determine the resolution of a dataset based on its grid spacing.

    This function detects resolution by measuring the spacing between
    coordinate points, making it robust to different geographic coverages.

    Returns:
        str: 'one_degree', 'quarter_degree', or 'twelfth_degree'

    Raises:
        ValueError: If the resolution is unknown
    """
    standard_dataset = rename_dataset_with_standard_names(dataset)

    latitude_values = standard_dataset[Dimension.LATITUDE.key()].values
    longitude_values = standard_dataset[Dimension.LONGITUDE.key()].values

    latitude_spacing = abs(float(latitude_values[1] - latitude_values[0]))
    longitude_spacing = abs(float(longitude_values[1] - longitude_values[0]))

    if numpy.isclose(latitude_spacing, ONE_DEGREE_SPACING, rtol=0.1) and numpy.isclose(
        longitude_spacing, ONE_DEGREE_SPACING, rtol=0.1
    ):
        return "one_degree"
    if numpy.isclose(latitude_spacing, QUARTER_DEGREE_SPACING, rtol=0.1) and numpy.isclose(
        longitude_spacing, QUARTER_DEGREE_SPACING, rtol=0.1
    ):
        return "quarter_degree"
    if numpy.isclose(latitude_spacing, TWELFTH_DEGREE_SPACING, rtol=0.1) and numpy.isclose(
        longitude_spacing, TWELFTH_DEGREE_SPACING, rtol=0.1
    ):
        return "twelfth_degree"

    raise ValueError(
        f"Unknown resolution: grid spacing latitude={latitude_spacing:.4f}°, longitude={longitude_spacing:.4f}°. "
        f"Expected spacings: {ONE_DEGREE_SPACING}° (one_degree), {QUARTER_DEGREE_SPACING}° (quarter_degree), "
        f"or {TWELFTH_DEGREE_SPACING:.4f}° (twelfth_degree)."
    )
