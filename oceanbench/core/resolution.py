# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from xarray import Dataset


from oceanbench.core.dataset_utils import Dimension

QUARTER_DEGREE = 0.25

QUARTER_DEGREE_LAT_SIZE = 672
QUARTER_DEGREE_LON_SIZE = 1440
TWELFTH_DEGREE_LAT_SIZE = 2041
TWELFTH_DEGREE_LON_SIZE = 4320


def is_quarter_degree_dataset(dataset: Dataset) -> bool:

    lat_key = Dimension.LATITUDE.key()
    lon_key = Dimension.LONGITUDE.key()

    if lat_key not in dataset.sizes or lon_key not in dataset.sizes:
        raise ValueError(f"Dataset missing required dimensions: {lat_key}, {lon_key}")

    lat_size = dataset.sizes[lat_key]
    lon_size = dataset.sizes[lon_key]

    if lat_size == QUARTER_DEGREE_LAT_SIZE and lon_size == QUARTER_DEGREE_LON_SIZE:
        return True
    elif lat_size == TWELFTH_DEGREE_LAT_SIZE and lon_size == TWELFTH_DEGREE_LON_SIZE:
        return False
    else:
        raise ValueError(
            f"Unknown resolution: dimensions {lat_key}={lat_size}, {lon_key}={lon_size}. "
            f"Expected values: ({QUARTER_DEGREE_LAT_SIZE}, {QUARTER_DEGREE_LON_SIZE}) for quarter degree "
            f"or ({TWELFTH_DEGREE_LAT_SIZE}, {TWELFTH_DEGREE_LON_SIZE}) for twelfth degree."
        )
