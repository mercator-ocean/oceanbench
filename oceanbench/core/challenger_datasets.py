# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

import xarray
from datetime import datetime, timedelta
from collections.abc import Callable

from oceanbench.core.curvilinear_staging import (
    GLOENS_FORECAST_DAYS,
    GLOENS_MEMBER_DIMENSION,
    GLOENS_SOURCE_DIMENSIONS,
    GLOENS_SOURCE_NAME,
    GLOENS_SUBSURFACE_CONTENTS,
    GLOENS_SURFACE_CONTENT,
    NEMO_TIME_DESCRIPTION_VARIABLES,
    gloens_store_url,
    open_gloens_store,
    with_common_depth_axis,
    without_native_grid_description,
)
from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import LEAD_DAYS_COUNT
from oceanbench.core.ensemble_gridded import ENSEMBLE_DIMENSION
from oceanbench.core.remote_http import require_remote_dataset_dimensions, with_remote_http_retries
from oceanbench.core.runtime_configuration import current_runtime_configuration
from oceanbench.core.weekly_stage import maybe_stage_weekly_dataset
from oceanbench.core.interpolate import interpolate_1_degree

_CLOUDFERRO_ML_FORECASTS_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/ml-forecast-outputs"
_GLO12_FORECASTS_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/dev/additionnal-data/GLO12"
_GLO12_FORECAST_VARIABLE_NAMES = ["so", "thetao", "uo", "vo", "zos"]
_LANGYA_LEAD_DAYS_COUNT = 7
_GLOENS_FIRST_INITIALISATION = "2024-01-04"
_GLOENS_LAST_INITIALISATION = "2024-12-26"
_GLOENS_INITIALISATION_TO_FIRST_DAY = timedelta(days=1)
_GLOENS_SCORED_FORECAST_DAYS = GLOENS_FORECAST_DAYS - 1


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
    return f"{_CLOUDFERRO_ML_FORECASTS_URL}/wenhai/v2/{start_datetime_string}.zarr"


def langya() -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(
        _langya_dataset_path, lead_days_count=_LANGYA_LEAD_DAYS_COUNT
    )


def langya_1_degree() -> xarray.Dataset:
    return interpolate_1_degree(langya())


def _langya_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"{_CLOUDFERRO_ML_FORECASTS_URL}/langya/{start_datetime_string}.zarr"


def gloens() -> xarray.Dataset:
    first_day_datetimes = _gloens_first_day_datetimes()

    def open_dataset() -> xarray.Dataset:
        return maybe_stage_weekly_dataset(
            stage_key="challenger",
            dataset_kind="challenger",
            dataset_name=GLOENS_SOURCE_NAME,
            first_day_datetimes=first_day_datetimes,
            lead_days_count=_GLOENS_SCORED_FORECAST_DAYS,
            open_week_dataset=_open_gloens_forecast_week,
            open_remote_dataset=lambda: _remote_gloens_dataset(first_day_datetimes),
            attach_source_metadata_when_not_staged=current_runtime_configuration().has_local_stage(),
        )

    return with_remote_http_retries("gloens challenger dataset open", open_dataset)


def _gloens_initialisation_datetimes() -> list[datetime]:
    """The initialisations of the GloEns year, which are Thursdays rather than Wednesdays.

    Every other challenger of the benchmark starts its weeks on the Wednesdays of 2024, and
    GloEns is issued a day later, so it has its own list rather than the shared one.
    """
    return generate_dates(_GLOENS_FIRST_INITIALISATION, _GLOENS_LAST_INITIALISATION, 7)


def _gloens_first_day_datetimes() -> list[datetime]:
    """The first scored day of each GloEns week, which is the day after its initialisation.

    Lead day one of the benchmark is the first day a forecast predicts, never the day it starts
    from. GloEns publishes the daily mean of its own initialisation day as field zero of every
    store, a nowcast no other challenger of the benchmark offers, so reading that field as lead
    day one would score this challenger a day ahead of every other one. The week therefore
    begins the day after the initialisation, and field zero is dropped as the store is read.
    """
    return [
        initialisation_datetime + _GLOENS_INITIALISATION_TO_FIRST_DAY
        for initialisation_datetime in _gloens_initialisation_datetimes()
    ]


def _open_gloens_store_content(first_day_datetime: datetime, content: str) -> xarray.Dataset:
    """One store of one GloEns week, with its time axis read as the lead day index.

    The store is named after the initialisation the week starts from, which is the day before
    the first day the week is scored on, and its field for that initialisation day is dropped
    so that the index the library reads as lead day one is the first day the forecast predicts.
    The scalar time descriptions of the store go first: they name the same axis and mean
    nothing once it is an index.
    """
    initialisation_datetime = first_day_datetime - _GLOENS_INITIALISATION_TO_FIRST_DAY
    store_dataset = open_gloens_store(gloens_store_url(initialisation_datetime, content))
    forecast_days = store_dataset.drop_vars(NEMO_TIME_DESCRIPTION_VARIABLES, errors="ignore").isel(time=slice(1, None))
    return _prepared_challenger_week_dataset(forecast_days, "gloens challenger dataset open")


def _with_float32_data_variables(dataset: xarray.Dataset) -> xarray.Dataset:
    """Read every field of a store as float32, which is the precision the benchmark scores in.

    The GloEns stores hold their fields as scaled integers, which xarray decodes to float64
    through the float64 scale factor of the store, so a week arrives at twice the precision
    every other challenger is read at and twice the memory. The cast is taken here, once, on
    the week as it is opened, so that nothing downstream sees the wider type.
    """
    return dataset.assign({name: dataset[name].astype("float32") for name in dataset.data_vars})


def _open_gloens_forecast_week(first_day_datetime: datetime) -> xarray.Dataset:
    """One GloEns initialisation, its five stores read as the one week the metrics score.

    The two-dimensional store comes first and keeps its grid description: it is the only store
    of the initialisation that carries the tracer positions, since the three-dimensional ones
    ship theirs as missing values from end to end, so theirs are dropped rather than merged
    against it. The three vertical axes the stores name the tracer levels under collapse onto
    the one scoring axis, as the producer collapses them, and the ensemble axis takes the name
    the ensemble metrics read it under.

    The sea level of this challenger carries an inverse barometer, which stays in the week as
    its own field: taking it off is the business of the Class IV sea level seam of
    :mod:`oceanbench.core.classIV_support`, which is where the mean sea surface shift of this
    challenger is declared as well.
    """
    surface_week = _open_gloens_store_content(first_day_datetime, GLOENS_SURFACE_CONTENT)
    subsurface_weeks = [
        without_native_grid_description(
            _open_gloens_store_content(first_day_datetime, content),
            GLOENS_SOURCE_DIMENSIONS,
        )
        for content in GLOENS_SUBSURFACE_CONTENTS
    ]
    week_dataset = with_common_depth_axis(xarray.merge([surface_week, *subsurface_weeks]))
    return _with_float32_data_variables(week_dataset).rename({GLOENS_MEMBER_DIMENSION: ENSEMBLE_DIMENSION})


def _remote_gloens_dataset(first_day_datetimes: list[datetime]) -> xarray.Dataset:
    return xarray.concat(
        [_open_gloens_forecast_week(first_day_datetime) for first_day_datetime in first_day_datetimes],
        dim="first_day_datetime",
    ).assign({"first_day_datetime": first_day_datetimes})


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
    lead_days_count: int = LEAD_DAYS_COUNT,
    for_class4: bool = False,
) -> xarray.Dataset:
    resolved_first_day_datetimes = _resolved_first_day_datetimes(first_day_datetimes)
    dataset_name = _challenger_dataset_name(forecast_zarr_path_from_start_datetime)

    def open_dataset() -> xarray.Dataset:
        return maybe_stage_weekly_dataset(
            stage_key="challenger",
            dataset_kind="challenger",
            dataset_name=dataset_name,
            first_day_datetimes=resolved_first_day_datetimes,
            lead_days_count=lead_days_count,
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
            for_class4=for_class4,
        )

    return with_remote_http_retries("challenger dataset open", open_dataset)
