# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

import xarray
from datetime import datetime, timedelta
from collections.abc import Callable

from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import LEAD_DAYS_COUNT
from oceanbench.core.remote_http import require_remote_dataset_dimensions, with_remote_http_retries
from oceanbench.core.runtime_configuration import current_runtime_configuration
from oceanbench.core.weekly_stage import maybe_stage_weekly_dataset
from oceanbench.core.interpolate import interpolate_1_degree

_CLOUDFERRO_ML_FORECASTS_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/ml-forecast-outputs"
_GLO12_FORECASTS_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/dev/additionnal-data/GLO12"
_GLO12_FORECAST_VARIABLE_NAMES = ["so", "thetao", "uo", "vo", "zos"]


def _default_first_day_datetimes() -> list[datetime]:
    return generate_dates("2024-01-03", "2024-12-25", 7)


def glo12() -> xarray.Dataset:
    first_day_datetimes = _default_first_day_datetimes()

    def open_dataset() -> xarray.Dataset:
        return maybe_stage_weekly_dataset(
            stage_key="challenger",
            dataset_kind="challenger",
            dataset_name="glo12",
            first_day_datetimes=first_day_datetimes,
            lead_days_count=LEAD_DAYS_COUNT,
            open_week_dataset=_open_glo12_forecast_week,
            open_remote_dataset=lambda: _remote_glo12_dataset(first_day_datetimes),
            attach_source_metadata_when_not_staged=current_runtime_configuration().has_local_stage(),
        )

    return with_remote_http_retries("glo12 challenger dataset open", open_dataset)


def glo12_1_degree() -> xarray.Dataset:
    return interpolate_1_degree(glo12())


def _glo12_dataset_path(start_datetime: datetime) -> str:
    run_date_string = (start_datetime + timedelta(days=1)).strftime("%Y%m%d")
    return f"{_GLO12_FORECASTS_URL}/glo12_rg_1d-m_fcst_R{run_date_string}.zarr"


def _open_glo12_forecast_week(first_day_datetime: datetime) -> xarray.Dataset:
    forecast_url = _glo12_dataset_path(first_day_datetime)
    forecast_week_dataset = xarray.merge(
        [
            xarray.open_zarr(forecast_url, group=variable_name, consolidated=True)[[variable_name]]
            for variable_name in _GLO12_FORECAST_VARIABLE_NAMES
        ]
    ).isel(time=slice(0, LEAD_DAYS_COUNT))
    return _prepared_challenger_week_dataset(forecast_week_dataset, "glo12 challenger dataset open")


def _remote_glo12_dataset(first_day_datetimes: list[datetime]) -> xarray.Dataset:
    return xarray.concat(
        [_open_glo12_forecast_week(first_day_datetime) for first_day_datetime in first_day_datetimes],
        dim="first_day_datetime",
    ).assign({"first_day_datetime": first_day_datetimes})


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
            parallel=True,
        )
        .rename({"lat": "latitude", "lon": "longitude"})
        .assign({"first_day_datetime": first_day_datetimes})
    )
    if not current_runtime_configuration().has_local_stage():
        return challenger_dataset
    return with_dataset_source(challenger_dataset, kind="challenger", name="glo36v1")


def _glo36v1_dataset_path(start_datetime: datetime) -> str:
    return f"https://minio.dive.edito.eu/project-moi-glo36-oceanbench/public/{start_datetime.strftime('%Y%m%d')}.zarr"


def glonet() -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(_glonet_dataset_path)


def glonet_1_degree() -> xarray.Dataset:
    return interpolate_1_degree(glonet())


def _glonet_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"{_CLOUDFERRO_ML_FORECASTS_URL}/glonet/{start_datetime_string}.zarr"


def xihe() -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(_xihe_dataset_path)


def xihe_1_degree() -> xarray.Dataset:
    return interpolate_1_degree(xihe())


def _xihe_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"{_CLOUDFERRO_ML_FORECASTS_URL}/xihe/{start_datetime_string}.zarr"


def wenhai() -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(_wenhai_dataset_path)


def wenhai_1_degree() -> xarray.Dataset:
    return interpolate_1_degree(wenhai())


def _wenhai_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"{_CLOUDFERRO_ML_FORECASTS_URL}/wenhai/{start_datetime_string}.zarr"


def _challenger_dataset_name(forecast_zarr_path_from_start_datetime: Callable[[datetime], str]) -> str:
    return forecast_zarr_path_from_start_datetime.__name__.removeprefix("_").replace("_dataset_path", "")


def _resolved_first_day_datetimes(first_day_datetimes: list[datetime] | None) -> list[datetime]:
    return first_day_datetimes if first_day_datetimes is not None else _default_first_day_datetimes()


def _prepared_challenger_week_dataset(
    dataset: xarray.Dataset,
    operation_name: str,
) -> xarray.Dataset:
    challenger_week_dataset = require_remote_dataset_dimensions(dataset, ["time"], operation_name)
    week_lead_days_count = challenger_week_dataset.sizes["time"]
    return challenger_week_dataset.rename({"time": "lead_day_index"}).assign_coords(
        {"lead_day_index": range(week_lead_days_count)}
    )


def _opened_challenger_week_dataset(
    forecast_zarr_path_from_start_datetime: Callable[[datetime], str],
    preprocess_dataset: Callable[[xarray.Dataset], xarray.Dataset] | None,
    first_day_datetime: datetime,
) -> xarray.Dataset:
    opened_dataset = xarray.open_dataset(
        forecast_zarr_path_from_start_datetime(first_day_datetime),
        engine="zarr",
    )
    return preprocess_dataset(opened_dataset) if preprocess_dataset is not None else opened_dataset


def _remote_multizarr_forecasts_as_challenger_dataset(
    dataset_name: str,
    forecast_zarr_path_from_start_datetime: Callable[[datetime], str],
    first_day_datetimes: list[datetime],
    preprocess_dataset: Callable[[xarray.Dataset], xarray.Dataset] | None,
) -> xarray.Dataset:
    challenger_dataset: xarray.Dataset = xarray.open_mfdataset(
        list(map(forecast_zarr_path_from_start_datetime, first_day_datetimes)),
        engine="zarr",
        preprocess=lambda dataset: _prepared_challenger_week_dataset(
            preprocess_dataset(dataset) if preprocess_dataset is not None else dataset,
            f"{dataset_name} challenger dataset open",
        ),
        combine="nested",
        concat_dim="first_day_datetime",
        parallel=False,
    ).assign({"first_day_datetime": first_day_datetimes})
    return challenger_dataset


def _open_multizarr_forecasts_as_challenger_dataset(
    forecast_zarr_path_from_start_datetime: Callable[[datetime], str],
    *,
    first_day_datetimes: list[datetime] | None = None,
    preprocess_dataset: Callable[[xarray.Dataset], xarray.Dataset] | None = None,
) -> xarray.Dataset:
    resolved_first_day_datetimes = _resolved_first_day_datetimes(first_day_datetimes)
    dataset_name = _challenger_dataset_name(forecast_zarr_path_from_start_datetime)

    def open_dataset() -> xarray.Dataset:
        return maybe_stage_weekly_dataset(
            stage_key="challenger",
            dataset_kind="challenger",
            dataset_name=dataset_name,
            first_day_datetimes=resolved_first_day_datetimes,
            lead_days_count=LEAD_DAYS_COUNT,
            open_week_dataset=lambda first_day_datetime: _prepared_challenger_week_dataset(
                _opened_challenger_week_dataset(
                    forecast_zarr_path_from_start_datetime,
                    preprocess_dataset,
                    first_day_datetime,
                ),
                f"{dataset_name} challenger dataset open",
            ),
            open_remote_dataset=lambda: _remote_multizarr_forecasts_as_challenger_dataset(
                dataset_name,
                forecast_zarr_path_from_start_datetime,
                resolved_first_day_datetimes,
                preprocess_dataset,
            ),
            attach_source_metadata_when_not_staged=current_runtime_configuration().has_local_stage(),
        )

    return with_remote_http_retries("challenger dataset open", open_dataset)
