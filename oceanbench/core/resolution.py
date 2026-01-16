# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from xarray import Dataset


def get_dataset_resolution(dataset: Dataset) -> str:
    """
    Determine the resolution of a dataset based on its dimensions.

    Returns:
        str: 'degree', 'quarter_degree', or 'twelfth_degree'

    Raises:
        ValueError: If the resolution is unknown
    """
    # Handle both possible names
    lat_dim_name = "latitude" if "latitude" in dataset.sizes else "lat"
    lon_dim_name = "longitude" if "longitude" in dataset.sizes else "lon"

    lat_size = dataset.sizes[lat_dim_name]
    lon_size = dataset.sizes[lon_dim_name]

    # Define known resolutions (lat, lon)
    resolutions = {
        # (168, 360): "degree",  # 1 degree
        (180, 360): "degree",  # 1 degree (from 1/12 degree interpolated datasets)
        (672, 1440): "quarter_degree",  # 0.25 degree
        (2041, 4320): "twelfth_degree",  # 1/12 degree (~0.083 degree)
    }

    key = (lat_size, lon_size)
    if key in resolutions:
        return resolutions[key]
    else:
        raise ValueError(
            f"Unknown resolution: dimensions {lat_dim_name}={lat_size}, {lon_dim_name}={lon_size}. "
            f"Expected values: {dict((k, v) for k, v in resolutions.items())}"
        )
