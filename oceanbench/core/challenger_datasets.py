# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

import numpy
import xarray
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timedelta
from collections.abc import Callable

from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import FIELD_CHUNKS, LEAD_DAYS_COUNT
from oceanbench.core.remote_http import (
    open_remote_multizarr,
    open_remote_zarr,
    remote_zarr_store,
    require_remote_dataset_dimensions,
)
from oceanbench.core.runtime_configuration import current_runtime_configuration
from oceanbench.core.weekly_stage import maybe_stage_weekly_dataset
from oceanbench.core.interpolate import interpolate_1_degree
from oceanbench.core.dataset_source import (
    with_challenger_kind,
    with_challenger_own_reference,
    with_glorys_sea_surface_height_datum,
)
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.references import glo12 as glo12_references
from oceanbench.core.references import glorys as glorys_references

# Chunks of the multizarr challenger opens. Unset, they open as before; the viewer maps set FIELD_CHUNKS.
_multizarr_open_chunks: ContextVar[dict | None] = ContextVar("multizarr_open_chunks", default=None)


@contextmanager
def field_chunked_challenger_open():
    """Open the multizarr challengers one horizontal field per chunk, for building the viewer maps only."""
    token = _multizarr_open_chunks.set(FIELD_CHUNKS)
    try:
        yield
    finally:
        _multizarr_open_chunks.reset(token)


_CLOUDFERRO_ML_FORECASTS_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/ml-forecast-outputs"
_CLOUDFERRO_BASELINE_FORECASTS_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/baseline-forecasts"
_GLO12_FORECASTS_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/dev/additionnal-data/GLO12"
_GLO12_FORECAST_VARIABLE_NAMES = ["so", "thetao", "uo", "vo", "zos"]
_GLO36V1_LEAD_DAYS_COUNT = 7
_LANGYA_LEAD_DAYS_COUNT = 7


def _default_first_day_datetimes() -> list[datetime]:
    return generate_dates("2024-01-03", "2024-12-25", 7)


def glo12(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    first_day_datetimes = _resolved_first_day_datetimes(first_day_datetimes)

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


def glo12_1_degree(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return interpolate_1_degree(glo12(first_day_datetimes))


def _glo12_dataset_path(start_datetime: datetime) -> str:
    run_date_string = (start_datetime + timedelta(days=1)).strftime("%Y%m%d")
    return f"{_GLO12_FORECASTS_URL}/glo12_rg_1d-m_fcst_R{run_date_string}.zarr"


def _open_glo12_forecast_week(first_day_datetime: datetime) -> xarray.Dataset:
    forecast_url = _glo12_dataset_path(first_day_datetime)
    forecast_week_dataset = xarray.merge(
        [
            xarray.open_zarr(
                remote_zarr_store(forecast_url), group=variable_name, consolidated=True, chunks=FIELD_CHUNKS
            )[[variable_name]]
            for variable_name in _GLO12_FORECAST_VARIABLE_NAMES
        ]
    ).isel(time=slice(0, LEAD_DAYS_COUNT))
    return _prepared_challenger_week_dataset(forecast_week_dataset, "glo12 challenger dataset open")


def _remote_glo12_dataset(first_day_datetimes: list[datetime]) -> xarray.Dataset:
    return xarray.concat(
        [_open_glo12_forecast_week(first_day_datetime) for first_day_datetime in first_day_datetimes],
        dim="first_day_datetime",
    ).assign({"first_day_datetime": first_day_datetimes})


def glo36v1(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    resolved_first_day_datetimes = (
        first_day_datetimes if first_day_datetimes is not None else generate_dates("2023-01-04", "2023-12-27", 7)
    )

    return maybe_stage_weekly_dataset(
        stage_key="challenger",
        dataset_kind="challenger",
        dataset_name="glo36v1",
        first_day_datetimes=resolved_first_day_datetimes,
        lead_days_count=_GLO36V1_LEAD_DAYS_COUNT,
        open_week_dataset=_open_glo36v1_forecast_week,
        open_remote_dataset=lambda: _remote_glo36v1_dataset(resolved_first_day_datetimes),
        attach_source_metadata_when_not_staged=current_runtime_configuration().has_local_stage(),
    )


def _glo36v1_dataset_path(start_datetime: datetime) -> str:
    return f"https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/glo36v1/{start_datetime.strftime('%Y%m%d')}.zarr"


def _open_glo36v1_forecast_week(first_day_datetime: datetime) -> xarray.Dataset:
    return open_remote_zarr(
        _glo36v1_dataset_path(first_day_datetime),
        chunks="auto",
    ).rename({"lat": "latitude", "lon": "longitude"})


def _remote_glo36v1_dataset(first_day_datetimes: list[datetime]) -> xarray.Dataset:
    return (
        open_remote_multizarr(
            list(map(_glo36v1_dataset_path, first_day_datetimes)),
            combine="nested",
            concat_dim="first_day_datetime",
            parallel=True,
        )
        .rename({"lat": "latitude", "lon": "longitude"})
        .assign({"first_day_datetime": first_day_datetimes})
    )


def glonet(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(
        _glonet_dataset_path, first_day_datetimes=first_day_datetimes
    )


def glonet_1_degree(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return interpolate_1_degree(glonet(first_day_datetimes))


def _glonet_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"{_CLOUDFERRO_ML_FORECASTS_URL}/glonet/{start_datetime_string}.zarr"


def xihe(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(_xihe_dataset_path, first_day_datetimes=first_day_datetimes)


def xihe_1_degree(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return interpolate_1_degree(xihe(first_day_datetimes))


def _xihe_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"{_CLOUDFERRO_ML_FORECASTS_URL}/xihe/{start_datetime_string}.zarr"


def wenhai(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(
        _wenhai_dataset_path, first_day_datetimes=first_day_datetimes
    )


def wenhai_1_degree(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return interpolate_1_degree(wenhai(first_day_datetimes))


def _wenhai_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"{_CLOUDFERRO_ML_FORECASTS_URL}/wenhai/v2/{start_datetime_string}.zarr"


def langya(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(
        _langya_dataset_path,
        first_day_datetimes=first_day_datetimes,
        lead_days_count=_LANGYA_LEAD_DAYS_COUNT,
    )


def langya_1_degree(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return interpolate_1_degree(langya(first_day_datetimes))


def _langya_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"{_CLOUDFERRO_ML_FORECASTS_URL}/langya/v2/{start_datetime_string}.zarr"


def glo12_persistence(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(
        _glo12_persistence_dataset_path, first_day_datetimes=first_day_datetimes
    )


def glo12_persistence_1_degree(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return interpolate_1_degree(glo12_persistence(first_day_datetimes))


def _glo12_persistence_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"{_CLOUDFERRO_BASELINE_FORECASTS_URL}/persistence/{start_datetime_string}.zarr"


def hclimrep(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return _open_multizarr_forecasts_as_challenger_dataset(
        _hclimrep_dataset_path, first_day_datetimes=first_day_datetimes
    )


def hclimrep_1_degree(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return interpolate_1_degree(hclimrep(first_day_datetimes))


def _hclimrep_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"{_CLOUDFERRO_ML_FORECASTS_URL}/hclimrep/{start_datetime_string}.zarr"


# Years from helper_scripts/baseline_generation/compute_glorys_climatology.py (commit 39495ac).
_CLIMATOLOGY_DESCRIPTION = "Daily climatology derived from GLORYS, years 1993 to 2019"
_GLORYS_DESCRIPTION = (
    "Reanalysis; assimilates the SLA, SST and T/S profiles scored here; values are analysis residuals, "
    "not forecast skill; lead is nominal (valid date)."
)
_GLO12_ANALYSIS_DESCRIPTION = (
    "Analysis of the GLO12 system; assimilates SLA, SST and T/S profiles like the ones scored here; values are "
    "analysis residuals, not forecast skill; lead is nominal (valid date)."
)


def _as_climatology(dataset: xarray.Dataset) -> xarray.Dataset:
    return with_glorys_sea_surface_height_datum(with_challenger_kind(dataset, "baseline", _CLIMATOLOGY_DESCRIPTION))


def climatology(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return _as_climatology(
        _open_multizarr_forecasts_as_challenger_dataset(
            _climatology_dataset_path, first_day_datetimes=first_day_datetimes
        )
    )


def climatology_1_degree(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    return _as_climatology(interpolate_1_degree(climatology(first_day_datetimes)))


def _climatology_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"{_CLOUDFERRO_BASELINE_FORECASTS_URL}/climatology/{start_datetime_string}.zarr"


def _forecast_shaped_template(first_day_datetimes: list[datetime]) -> xarray.Dataset:
    # The GLORYS reference openers take their start dates and lead day count from a challenger.
    return xarray.Dataset(
        coords={
            "first_day_datetime": numpy.array(first_day_datetimes, dtype="datetime64[ns]"),
            "lead_day_index": numpy.arange(LEAD_DAYS_COUNT),
        }
    )


def _as_glorys(dataset: xarray.Dataset) -> xarray.Dataset:
    return with_glorys_sea_surface_height_datum(
        with_challenger_own_reference(
            with_challenger_kind(dataset, "reference", _GLORYS_DESCRIPTION), "glorys_reanalysis"
        )
    )


def glorys(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    template = _forecast_shaped_template(_resolved_first_day_datetimes(first_day_datetimes))
    return _as_glorys(glorys_references._glorys_reanalysis_dataset_1_4(template))


def glorys_1_degree(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    template = _forecast_shaped_template(_resolved_first_day_datetimes(first_day_datetimes))
    return _as_glorys(glorys_references._glorys_reanalysis_dataset_1_degree(template))


def _twelfth_degree_template(first_day_datetimes: list[datetime]) -> xarray.Dataset:
    # The twelfth-degree reference openers also take their depths from a challenger: the GLO12 forecast ones.
    depths = _open_glo12_forecast_week(first_day_datetimes[0])[Dimension.DEPTH.key()].values
    return _forecast_shaped_template(first_day_datetimes).assign_coords({Dimension.DEPTH.key(): depths})


def glorys_twelfth_degree(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    template = _twelfth_degree_template(_resolved_first_day_datetimes(first_day_datetimes))
    return _as_glorys(glorys_references._glorys_reanalysis_dataset_1_12(template))


def glo12_analysis(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    template = _twelfth_degree_template(_resolved_first_day_datetimes(first_day_datetimes))
    return with_challenger_own_reference(
        with_challenger_kind(
            glo12_references._glo12_analysis_dataset_1_12(template), "reference", _GLO12_ANALYSIS_DESCRIPTION
        ),
        "glo12_analysis",
    )


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
    first_day_datetime: datetime,
) -> xarray.Dataset:
    return open_remote_zarr(
        forecast_zarr_path_from_start_datetime(first_day_datetime),
        chunks=_multizarr_open_chunks.get() or "auto",
    )


def _remote_multizarr_forecasts_as_challenger_dataset(
    dataset_name: str,
    forecast_zarr_path_from_start_datetime: Callable[[datetime], str],
    first_day_datetimes: list[datetime],
) -> xarray.Dataset:
    challenger_dataset: xarray.Dataset = open_remote_multizarr(
        list(map(forecast_zarr_path_from_start_datetime, first_day_datetimes)),
        preprocess=lambda dataset: _prepared_challenger_week_dataset(
            dataset,
            f"{dataset_name} challenger dataset open",
        ),
        combine="nested",
        concat_dim="first_day_datetime",
        parallel=False,
        **({"chunks": chunks} if (chunks := _multizarr_open_chunks.get()) else {}),
    ).assign({"first_day_datetime": first_day_datetimes})
    return challenger_dataset


def _open_multizarr_forecasts_as_challenger_dataset(
    forecast_zarr_path_from_start_datetime: Callable[[datetime], str],
    *,
    first_day_datetimes: list[datetime] | None = None,
    lead_days_count: int = LEAD_DAYS_COUNT,
) -> xarray.Dataset:
    resolved_first_day_datetimes = _resolved_first_day_datetimes(first_day_datetimes)
    dataset_name = _challenger_dataset_name(forecast_zarr_path_from_start_datetime)

    return maybe_stage_weekly_dataset(
        stage_key="challenger",
        dataset_kind="challenger",
        dataset_name=dataset_name,
        first_day_datetimes=resolved_first_day_datetimes,
        lead_days_count=lead_days_count,
        open_week_dataset=lambda first_day_datetime: _prepared_challenger_week_dataset(
            _opened_challenger_week_dataset(
                forecast_zarr_path_from_start_datetime,
                first_day_datetime,
            ),
            f"{dataset_name} challenger dataset open",
        ),
        open_remote_dataset=lambda: _remote_multizarr_forecasts_as_challenger_dataset(
            dataset_name,
            forecast_zarr_path_from_start_datetime,
            resolved_first_day_datetimes,
        ),
        attach_source_metadata_when_not_staged=current_runtime_configuration().has_local_stage(),
    )
