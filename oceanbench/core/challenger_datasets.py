# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

import xarray
from datetime import datetime
from collections.abc import Callable

from oceanbench.core.challenger_stage import staged_challenger_dataset, should_stage_challenger_locally
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
    return _open_multizarr_forecasts_as_challenger_dataset(
        _glo36v1_dataset_path,
        first_day_datetimes=generate_dates("2023-01-04", "2023-12-27", 7),
        preprocess_dataset=lambda dataset: dataset.rename({"lat": "latitude", "lon": "longitude"}),
    )


def _glo36v1_dataset_path(start_datetime: datetime) -> str:
    return f"https://minio.dive.edito.eu/project-moi-glo36-oceanbench/public/{start_datetime.strftime('%Y%m%d')}.zarr"


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
    *,
    first_day_datetimes: list[datetime] | None = None,
    preprocess_dataset: Callable[[xarray.Dataset], xarray.Dataset] = lambda dataset: dataset,
) -> xarray.Dataset:
    resolved_first_day_datetimes = (
        first_day_datetimes if first_day_datetimes is not None else generate_dates("2024-01-03", "2024-12-25", 7)
    )
    dataset_name = zarr_path_callback.__name__.removeprefix("_").replace("_dataset_path", "")

    def prepare_week_dataset(dataset: xarray.Dataset, operation_name: str) -> xarray.Dataset:
        challenger_week_dataset = require_remote_dataset_dimensions(dataset, ["time"], operation_name)
        week_lead_days_count = challenger_week_dataset.sizes["time"]
        return challenger_week_dataset.rename({"time": "lead_day_index"}).assign_coords(
            {"lead_day_index": range(week_lead_days_count)}
        )

    def open_dataset() -> xarray.Dataset:
        with instrumented_operation(
            "challenger_dataset_load",
            dataset=dataset_name,
            weeks_count=len(resolved_first_day_datetimes),
        ):
            if should_stage_challenger_locally():
                return staged_challenger_dataset(
                    dataset_name=dataset_name,
                    first_day_datetimes=resolved_first_day_datetimes,
                    lead_days_count=LEAD_DAYS_COUNT,
                    open_week_dataset=lambda first_day_datetime: prepare_week_dataset(
                        preprocess_dataset(
                            xarray.open_dataset(
                                zarr_path_callback(first_day_datetime),
                                engine="zarr",
                            )
                        ),
                        f"{dataset_name} challenger dataset open",
                    ),
                )

            challenger_dataset: xarray.Dataset = xarray.open_mfdataset(
                list(map(zarr_path_callback, resolved_first_day_datetimes)),
                engine="zarr",
                preprocess=lambda dataset: prepare_week_dataset(
                    preprocess_dataset(dataset),
                    f"{dataset_name} challenger dataset open",
                ),
                combine="nested",
                concat_dim="first_day_datetime",
                parallel=False,
            ).assign({"first_day_datetime": resolved_first_day_datetimes})
            return with_dataset_source(challenger_dataset, kind="challenger", name=dataset_name)

    return with_remote_http_retries("challenger dataset open", open_dataset)
