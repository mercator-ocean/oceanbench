# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from xarray import Dataset


from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.climate_forecast_standard_names import (
    rename_dataset_with_standard_names,
)

QUARTER_DEGREE = 0.25

QUARTER_DEGREE_LAT_SIZE = 672
QUARTER_DEGREE_LON_SIZE = 1440
TWELFTH_DEGREE_LAT_SIZE = 2041
TWELFTH_DEGREE_LON_SIZE = 4320


def is_quarter_degree_dataset(dataset: Dataset) -> bool:
    standard_dataset = rename_dataset_with_standard_names(dataset)

    latitude_size = standard_dataset.sizes[Dimension.LATITUDE.key()]
    longitude_size = standard_dataset.sizes[Dimension.LONGITUDE.key()]

    if latitude_size == QUARTER_DEGREE_LAT_SIZE and longitude_size == QUARTER_DEGREE_LON_SIZE:
        return True
    if latitude_size == TWELFTH_DEGREE_LAT_SIZE and longitude_size == TWELFTH_DEGREE_LON_SIZE:
        return False
    raise ValueError(
        f"Unknown resolution: dimensions {Dimension.LATITUDE.key()}={latitude_size}, "
        f"{Dimension.LONGITUDE.key()}={longitude_size}. "
        f"Expected values: ({QUARTER_DEGREE_LAT_SIZE}, {QUARTER_DEGREE_LON_SIZE}) for quarter degree "
        f"or ({TWELFTH_DEGREE_LAT_SIZE}, {TWELFTH_DEGREE_LON_SIZE}) for twelfth degree."
    )
