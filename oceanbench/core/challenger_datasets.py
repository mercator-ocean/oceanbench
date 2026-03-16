# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

import xarray
from datetime import datetime
from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import LEAD_DAYS_COUNT
from oceanbench.core.instrumentation import instrumented_operation
from oceanbench.core.remote_http import require_remote_dataset_dimensions, with_remote_http_retries


def glo12() -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(_glo12_dataset_path)


def _glo12_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/GLO12/{start_datetime_string}.zarr"


def glo36v1() -> xarray.Dataset:
    first_day_datetimes = generate_dates("2023-01-04", "2023-12-27", 7)

    challenger_dataset = (
        xarray.open_mfdataset(
            [
                f"https://minio.dive.edito.eu/project-moi-glo36-oceanbench/public/{dt.strftime('%Y%m%d')}.zarr"
                for dt in first_day_datetimes
            ],
            engine="zarr",
            combine="nested",
            concat_dim="first_day_datetime",
            parallel=False,
        )
        .rename({"lat": "latitude", "lon": "longitude"})
        .assign({"first_day_datetime": first_day_datetimes})
    )
    return with_dataset_source(challenger_dataset, kind="challenger", name="glo36v1")


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


def _open_multizarr_forecasts_as_challenger_dataset(
    zarr_path_callback,
) -> xarray.Dataset:
    first_day_datetimes: list[datetime] = generate_dates("2024-01-03", "2024-12-25", 7)
    dataset_name = zarr_path_callback.__name__.removeprefix("_").replace("_dataset_path", "")

    def open_dataset() -> xarray.Dataset:
        with instrumented_operation(
            "challenger_dataset_load",
            dataset=dataset_name,
            weeks_count=len(first_day_datetimes),
        ):
            challenger_dataset: xarray.Dataset = xarray.open_mfdataset(
                list(map(zarr_path_callback, first_day_datetimes)),
                engine="zarr",
                preprocess=lambda dataset: require_remote_dataset_dimensions(
                    dataset,
                    ["time"],
                    "challenger dataset open",
                )
                .rename({"time": "lead_day_index"})
                .assign({"lead_day_index": range(LEAD_DAYS_COUNT)}),
                combine="nested",
                concat_dim="first_day_datetime",
                parallel=False,
            ).assign({"first_day_datetime": first_day_datetimes})
            return with_dataset_source(challenger_dataset, kind="challenger", name=dataset_name)

    return with_remote_http_retries("challenger dataset open", open_dataset)
