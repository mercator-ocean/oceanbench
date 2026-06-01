# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
from os import environ
from urllib.parse import unquote, urlparse

import pandas
import xarray

from oceanbench.core.dataset_utils import Dimension, LEAD_DAYS_COUNT
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.remote_http import require_remote_dataset_dimensions, with_remote_http_retries

LIVE_CLASS4_OBSERVATION_ZARR_TEMPLATE = (
    "https://minio.dive.edito.eu/project-oceanbench/public/observations2026/{day}.zarr"
)
LIVE_CLASS4_OBSERVATION_LAST_DAY = "2026-05-23"
LIVE_GLONET_FORECAST_ZARR_TEMPLATE = (
    "https://minio.dive.edito.eu/project-moiai-octo/public/octo/v0/ai-gallery/" "octo-glonet-p1d/{date}/{date}.zarr"
)


def live_class4_observation_zarr_template() -> str:
    return environ.get(
        OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_OBSERVATION_ZARR_TEMPLATE.value,
        LIVE_CLASS4_OBSERVATION_ZARR_TEMPLATE,
    )


def live_class4_observation_last_day() -> str:
    return environ.get(
        OceanbenchEnvironmentVariable.OCEANBENCH_LIVE_OBSERVATION_LAST_DAY.value,
        LIVE_CLASS4_OBSERVATION_LAST_DAY,
    )


def _default_live_first_day_datetime() -> datetime:
    last_observation_day = pandas.Timestamp(live_class4_observation_last_day())
    return (last_observation_day - pandas.Timedelta(days=LEAD_DAYS_COUNT)).to_pydatetime()


def _format_forecast_zarr_template(
    first_day_datetime: datetime,
    zarr_template: str,
) -> str:
    day_string = first_day_datetime.strftime("%Y%m%d")
    date_string = first_day_datetime.strftime("%Y-%m-%d")
    path = zarr_template.format(
        day=day_string,
        date=date_string,
        yyyymmdd=day_string,
        YYYYMMDD=day_string,
    )
    if path.startswith("file://"):
        parsed_path = urlparse(path)
        return unquote(parsed_path.path)
    return path


def _configured_live_glonet_forecast_zarr_template() -> str:
    return environ.get(
        OceanbenchEnvironmentVariable.OCEANBENCH_LIVE_GLONET_FORECAST_ZARR_TEMPLATE.value,
        LIVE_GLONET_FORECAST_ZARR_TEMPLATE,
    )


def _prepared_live_forecast_dataset(
    dataset: xarray.Dataset,
    first_day_datetime: datetime,
) -> xarray.Dataset:
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    if lead_day_key not in dataset.dims:
        dataset = require_remote_dataset_dimensions(dataset, ["time"], "live GLONET forecast dataset open")
        dataset = dataset.rename({"time": lead_day_key})
    lead_days_count = dataset.sizes[lead_day_key]
    dataset = dataset.assign_coords({lead_day_key: range(lead_days_count)})
    return dataset.expand_dims({first_day_key: [first_day_datetime]})


def live_glo12_analysis_zarr_template() -> str | None:
    return environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_LIVE_GLO12_ZARR_TEMPLATE.value)


def live_reference_dataset(
    challenger_dataset: xarray.Dataset,
    zarr_template: str,
) -> xarray.Dataset:
    def open_dataset() -> xarray.Dataset:
        first_day_key = Dimension.FIRST_DAY_DATETIME.key()
        first_day_datetimes = pandas.to_datetime(challenger_dataset[first_day_key].values).to_pydatetime()
        datasets = [
            _prepared_live_forecast_dataset(
                xarray.open_dataset(
                    _format_forecast_zarr_template(first_day_datetime, zarr_template),
                    engine="zarr",
                ),
                first_day_datetime,
            )
            for first_day_datetime in first_day_datetimes
        ]
        if len(datasets) == 1:
            return datasets[0]
        return xarray.concat(datasets, dim=first_day_key)

    return with_remote_http_retries("live reference dataset open", open_dataset)


def glonet_latest(
    first_day_datetime: datetime | None = None,
    zarr_template: str | None = None,
) -> xarray.Dataset:
    resolved_first_day_datetime = first_day_datetime or _default_live_first_day_datetime()
    resolved_zarr_template = zarr_template or _configured_live_glonet_forecast_zarr_template()
    forecast_zarr_path = _format_forecast_zarr_template(
        resolved_first_day_datetime,
        resolved_zarr_template,
    )

    def open_dataset() -> xarray.Dataset:
        return _prepared_live_forecast_dataset(
            xarray.open_dataset(forecast_zarr_path, engine="zarr"),
            resolved_first_day_datetime,
        )

    return with_remote_http_retries("live GLONET forecast dataset open", open_dataset)
