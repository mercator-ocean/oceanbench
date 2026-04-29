# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

import xarray
from datetime import datetime
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import LEAD_DAYS_COUNT


def _weekly_first_day_datetimes_2024() -> list[datetime]:
    return generate_dates("2024-01-03", "2024-12-25", 7)


def _weekly_first_day_datetimes_2023() -> list[datetime]:
    return generate_dates("2023-01-04", "2023-12-27", 7)


def glo12_nowcasts() -> xarray.Dataset:
    first_day_datetimes = _weekly_first_day_datetimes_2024()
    dataset: xarray.Dataset = xarray.open_mfdataset(
        list(map(_glo12_nowcast_dataset_path, first_day_datetimes)),
        engine="zarr",
        parallel=False,
    )
    return dataset


def _glo12_nowcast_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/{start_datetime_string}.zarr"


def ifs_forcings() -> xarray.Dataset:
    first_day_datetimes = _weekly_first_day_datetimes_2024()
    dataset: xarray.Dataset = xarray.open_mfdataset(
        list(map(_ifs_forcing_dataset_path, first_day_datetimes)),
        engine="netcdf4",
        preprocess=lambda dataset: dataset.rename({"time_counter": "lead_day_index"}).assign(
            {"lead_day_index": range(LEAD_DAYS_COUNT)}
        ),
        combine="nested",
        concat_dim="first_day_datetime",
        decode_timedelta=False,
    ).assign({"first_day_datetime": first_day_datetimes})
    return dataset


def _ifs_forcing_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_{start_datetime_string}.nc#mode=bytes"


def glo12_forcings() -> xarray.Dataset:
    first_day_datetimes = _weekly_first_day_datetimes_2023()
    dataset: xarray.Dataset = xarray.open_mfdataset(
        list(map(_glo12_forcing_dataset_path, first_day_datetimes)),
        engine="zarr",
        parallel=False,
    )
    return dataset


def _glo12_forcing_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return (
        "https://oceanbench-bucket.s3.amazonaws.com/dev/additionnal-data/GLO12/"
        f"glo12_rg_1d-m_nwct_R{start_datetime_string}.zarr"
    )


def ifs_forcings_zarr() -> xarray.Dataset:
    first_day_datetimes = _weekly_first_day_datetimes_2023()
    dataset: xarray.Dataset = xarray.open_mfdataset(
        list(map(_ifs_forcing_zarr_dataset_path, first_day_datetimes)),
        engine="zarr",
        parallel=False,
    )
    return dataset


def _ifs_forcing_zarr_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return (
        "https://oceanbench-bucket.s3.amazonaws.com/dev/additionnal-data/IFS/"
        f"ifs_forcing_rg_forecasts_R{start_datetime_string}.zarr"
    )
