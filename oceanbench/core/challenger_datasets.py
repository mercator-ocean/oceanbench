# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

import xarray
from datetime import datetime
from collections.abc import Callable

from oceanbench.core.cloudferro import cloudferro_public_url, zarr_open_kwargs
from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import LEAD_DAYS_COUNT
from oceanbench.core.evaluation_year import (
    evaluation_year_first_day_datetimes,
    validate_evaluation_year,
)
from oceanbench.core.remote_http import (
    require_remote_dataset_dimensions,
    with_remote_http_retries,
)
from oceanbench.core.runtime_configuration import current_runtime_configuration
from oceanbench.core.weekly_stage import maybe_stage_weekly_dataset
from oceanbench.core.interpolate import interpolate_1_degree


def _default_evaluation_year() -> int:
    return current_runtime_configuration().evaluation_year


def glo12(evaluation_year: int | str | None = None) -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(
        _glo12_dataset_path,
        evaluation_year=evaluation_year,
    )


def glo12_1_degree(evaluation_year: int | str | None = None) -> xarray.Dataset:
    return interpolate_1_degree(glo12(evaluation_year=evaluation_year))


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


def glonet(evaluation_year: int | str | None = None) -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(
        _glonet_dataset_path,
        evaluation_year=evaluation_year,
    )


def glonet_1_degree(evaluation_year: int | str | None = None) -> xarray.Dataset:
    return interpolate_1_degree(glonet(evaluation_year=evaluation_year))


def _glonet_dataset_path(start_datetime: datetime) -> str:
    return _ml_forecast_output_dataset_path("glonet", start_datetime)


def langya(evaluation_year: int | str | None = None) -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(
        _langya_dataset_path,
        evaluation_year=evaluation_year,
    )


def langya_1_degree(evaluation_year: int | str | None = None) -> xarray.Dataset:
    return interpolate_1_degree(langya(evaluation_year=evaluation_year))


def _langya_dataset_path(start_datetime: datetime) -> str:
    return _ml_forecast_output_dataset_path("langya", start_datetime)


def xihe(evaluation_year: int | str | None = None) -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(
        _xihe_dataset_path,
        evaluation_year=evaluation_year,
    )


def xihe_1_degree(evaluation_year: int | str | None = None) -> xarray.Dataset:
    return interpolate_1_degree(xihe(evaluation_year=evaluation_year))


def _xihe_dataset_path(start_datetime: datetime) -> str:
    return _ml_forecast_output_dataset_path("xihe", start_datetime)


def wenhai(evaluation_year: int | str | None = None) -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(
        _wenhai_dataset_path,
        evaluation_year=evaluation_year,
    )


def wenhai_1_degree(evaluation_year: int | str | None = None) -> xarray.Dataset:
    return interpolate_1_degree(wenhai(evaluation_year=evaluation_year))


def _wenhai_dataset_path(start_datetime: datetime) -> str:
    return _ml_forecast_output_dataset_path("wenhai", start_datetime)


def _ml_forecast_output_dataset_path(model_name: str, start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return cloudferro_public_url("ml-forecast-outputs", model_name, f"{start_datetime_string}.zarr")


def _challenger_dataset_name(forecast_zarr_path_from_start_datetime: Callable[[datetime], str]) -> str:
    return forecast_zarr_path_from_start_datetime.__name__.removeprefix("_").replace("_dataset_path", "")


def _resolved_evaluation_year(evaluation_year: int | str | None) -> int:
    return validate_evaluation_year(evaluation_year if evaluation_year is not None else _default_evaluation_year())


def _resolved_first_day_datetimes(
    first_day_datetimes: list[datetime] | None,
    evaluation_year: int | str | None,
) -> list[datetime]:
    if first_day_datetimes is not None:
        return first_day_datetimes
    return evaluation_year_first_day_datetimes(_resolved_evaluation_year(evaluation_year))


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
    dataset_path = forecast_zarr_path_from_start_datetime(first_day_datetime)
    opened_dataset = xarray.open_dataset(
        dataset_path,
        engine="zarr",
        **zarr_open_kwargs(dataset_path),
    )
    return preprocess_dataset(opened_dataset) if preprocess_dataset is not None else opened_dataset


def _remote_multizarr_forecasts_as_challenger_dataset(
    dataset_name: str,
    forecast_zarr_path_from_start_datetime: Callable[[datetime], str],
    first_day_datetimes: list[datetime],
    preprocess_dataset: Callable[[xarray.Dataset], xarray.Dataset] | None,
) -> xarray.Dataset:
    dataset_paths = list(map(forecast_zarr_path_from_start_datetime, first_day_datetimes))
    challenger_dataset: xarray.Dataset = xarray.open_mfdataset(
        dataset_paths,
        engine="zarr",
        preprocess=lambda dataset: _prepared_challenger_week_dataset(
            preprocess_dataset(dataset) if preprocess_dataset is not None else dataset,
            f"{dataset_name} challenger dataset open",
        ),
        combine="nested",
        concat_dim="first_day_datetime",
        parallel=False,
        **zarr_open_kwargs(dataset_paths[0]),
    ).assign({"first_day_datetime": first_day_datetimes})
    return challenger_dataset


def _open_multizarr_forecasts_as_challenger_dataset(
    forecast_zarr_path_from_start_datetime: Callable[[datetime], str],
    *,
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
    preprocess_dataset: Callable[[xarray.Dataset], xarray.Dataset] | None = None,
) -> xarray.Dataset:
    resolved_first_day_datetimes = _resolved_first_day_datetimes(first_day_datetimes, evaluation_year)
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
