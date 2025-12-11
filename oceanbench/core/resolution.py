# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from xarray import Dataset

QUARTER_DEGREE = 0.25


def is_quarter_degree_dataset(dataset: Dataset) -> bool:
    # Handle both possible names
    lat_dim_name = "latitude" if "latitude" in dataset.sizes else "lat"
    lon_dim_name = "longitude" if "longitude" in dataset.sizes else "lon"

    lat_size = dataset.sizes[lat_dim_name]
    lon_size = dataset.sizes[lon_dim_name]

    # Quarter degree: lat=672, lon=1440
    if lat_size == 672 and lon_size == 1440:
        return True
    # Twelfth degree: lat=2041, lon=4320
    elif lat_size == 2041 and lon_size == 4320:
        return False
    else:
        raise ValueError(
            f"Unknown resolution: dimensions {lat_dim_name}={lat_size}, {lon_dim_name}={lon_size}. "
            f"Expected values: (672, 1440) for quarter degree or (2041, 4320) for twelfth degree."
        )
