# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

import xarray
from datetime import datetime
from oceanbench.core.datetime_utils import generate_dates


def glonet() -> xarray.Dataset:
    first_day_datetimes: list[datetime] = generate_dates("2024-01-03", "2024-12-25", 7)
    challenger_dataset: xarray.Dataset = xarray.open_mfdataset(
        list(map(_glonet_dataset_path, first_day_datetimes)),
        engine="zarr",
        preprocess=lambda dataset: dataset.rename({"time": "lead_day_index"}).assign({"lead_day_index": range(10)}),
        combine="nested",
        concat_dim="first_day_datetime",
        parallel=True,
    ).assign({"first_day_datetime": first_day_datetimes})
    return challenger_dataset


def _glonet_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glonet_full_2024/{start_datetime_string}.zarr"
