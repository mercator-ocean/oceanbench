# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

import xarray
from datetime import datetime, timedelta
from oceanbench.core.datetime_utils import generate_dates


def glo12() -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(_glo12_dataset_path)


def _glo12_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/GLO12/{start_datetime_string}.zarr"


def glo36v1() -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(_glo36v1_dataset_path, preprocess=None)


def _glo36v1_dataset_path(start_datetime: datetime) -> str:
    # GLO36 data is from 2023, but we artificially use 2024 dates.
    # So we map the 2024 request back to 2023 files.
    actual_datetime = start_datetime - timedelta(weeks=52)
    start_datetime_string = actual_datetime.strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-moi-glo36-oceanbench/public/{start_datetime_string}.zarr"


def glonet() -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(_glonet_dataset_path)


def _glonet_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glonet_full_2024/{start_datetime_string}.zarr"


def xihe() -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(_xihe_dataset_path)


def _xihe_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/XIHE/{start_datetime_string}.zarr"


def wenhai() -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(_wenhai_dataset_path)


def _wenhai_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/WENHAI/{start_datetime_string}.zarr"


def _rename_time_to_lead_day_index(dataset: xarray.Dataset) -> xarray.Dataset:
    return dataset.rename({"time": "lead_day_index"}).assign({"lead_day_index": range(10)})


def _open_multizarr_forecasts_as_challenger_dataset(
    zarr_path_callback,
    preprocess=_rename_time_to_lead_day_index,
) -> xarray.Dataset:
    first_day_datetimes: list[datetime] = generate_dates("2024-01-03", "2024-12-25", 7)

    challenger_dataset: xarray.Dataset = xarray.open_mfdataset(
        list(map(zarr_path_callback, first_day_datetimes)),
        engine="zarr",
        preprocess=preprocess,
        combine="nested",
        concat_dim="first_day_datetime",
        parallel=True,
    ).assign({"first_day_datetime": first_day_datetimes})
    return challenger_dataset
