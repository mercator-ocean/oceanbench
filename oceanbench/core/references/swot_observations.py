# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import hashlib
import logging
from pathlib import Path

import numpy
import pandas
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.local_stage import (
    local_stage_directory,
    open_or_create_local_stage_dataset,
    should_stage_locally,
    write_dataset_to_local_stage,
)
from oceanbench.core.references.observations import (
    LOCAL_STAGE_OBSERVATIONS_KEY,
    OBSERVATIONS_FIRST_AVAILABLE_DATE,
    ObservationBasisVersionError,
    ObservationDataUnavailableError,
    forecast_observation_matches,
)
from oceanbench.core.remote_http import IncompleteRemoteDatasetError, open_remote_multizarr, remote_zarr_store

SWOT_OBSERVATIONS_STAGE_VERSION = "v1"
SWOT_OBSERVATIONS_BASIS_VERSION_ATTRIBUTE = "obs_basis_version"
EXPECTED_SWOT_OBSERVATIONS_BASIS_VERSION = "v2.2"
SWOT_OBSERVATIONS_SUBSAMPLE_FACTOR_ATTRIBUTE = "subsample_factor"
EXPECTED_SWOT_OBSERVATIONS_SUBSAMPLE_FACTOR = 10
OBSERVATION_DIMENSION_KEY = "observations"
SWOT_OBSERVATIONS_LOGGER = logging.getLogger(__name__)


def swot_observation_path(day_datetime: numpy.datetime64) -> str:
    day_timestamp = pandas.Timestamp(day_datetime)
    return (
        "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/class4/swot/l3/zarr-v2/"
        f"{day_timestamp:%Y}/{day_timestamp:%Y%m%d}.zarr"
    )


def _is_converted_day(day_datetime: numpy.datetime64) -> bool:
    try:
        remote_zarr_store(swot_observation_path(day_datetime))[".zmetadata"]
    except KeyError:
        return False
    return True


def _require_expected_conversion(day_swot_observations_dataset: xarray.Dataset) -> xarray.Dataset:
    found_basis_version = day_swot_observations_dataset.attrs.get(SWOT_OBSERVATIONS_BASIS_VERSION_ATTRIBUTE)
    found_subsample_factor = day_swot_observations_dataset.attrs.get(SWOT_OBSERVATIONS_SUBSAMPLE_FACTOR_ATTRIBUTE)
    if (
        found_basis_version != EXPECTED_SWOT_OBSERVATIONS_BASIS_VERSION
        or found_subsample_factor != EXPECTED_SWOT_OBSERVATIONS_SUBSAMPLE_FACTOR
    ):
        raise ObservationBasisVersionError(
            "SWOT observation day store was converted with unexpected settings. "
            f"Expected {SWOT_OBSERVATIONS_BASIS_VERSION_ATTRIBUTE}={EXPECTED_SWOT_OBSERVATIONS_BASIS_VERSION!r} "
            f"and {SWOT_OBSERVATIONS_SUBSAMPLE_FACTOR_ATTRIBUTE}={EXPECTED_SWOT_OBSERVATIONS_SUBSAMPLE_FACTOR!r}, "
            f"found {found_basis_version!r} and {found_subsample_factor!r}."
        )
    return day_swot_observations_dataset


def _consumed_swot_observation_variable_keys() -> list[str]:
    return [
        Dimension.TIME.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    ]


def _select_consumed_swot_observation_variables(day_swot_observations_dataset: xarray.Dataset) -> xarray.Dataset:
    consumed_variable_keys = _consumed_swot_observation_variable_keys()
    missing_variable_keys = [
        variable_key
        for variable_key in consumed_variable_keys
        if variable_key not in day_swot_observations_dataset.variables
    ]
    if missing_variable_keys:
        raise IncompleteRemoteDatasetError(
            f"Remote dataset opened without expected variables {missing_variable_keys} "
            "during SWOT observation dataset open. "
            f"Available variables: {sorted(day_swot_observations_dataset.variables)}"
        )
    return day_swot_observations_dataset[consumed_variable_keys]


def _with_surface_depth(day_swot_observations_dataset: xarray.Dataset) -> xarray.Dataset:
    latitude = day_swot_observations_dataset[Dimension.LATITUDE.key()]
    return day_swot_observations_dataset.assign(
        {
            Dimension.DEPTH.key(): xarray.DataArray(
                numpy.zeros_like(latitude.data),
                dims=latitude.dims,
                attrs={"standard_name": Dimension.DEPTH.key()},
            )
        }
    )


def _prepare_day_swot_observations_dataset(day_swot_observations_dataset: xarray.Dataset) -> xarray.Dataset:
    return _with_surface_depth(
        _select_consumed_swot_observation_variables(
            _require_expected_conversion(day_swot_observations_dataset),
        )
    )


def _converted_observation_days(observation_days: numpy.ndarray) -> list[numpy.datetime64]:
    converted_observation_days = [day for day in observation_days if _is_converted_day(day)]
    if not converted_observation_days:
        raise ObservationDataUnavailableError(
            "SWOT-based Class IV scores were not computed for this challenger. "
            "No converted SWOT observation day covers its forecast period."
        )
    missing_observation_days = [day for day in observation_days if day not in converted_observation_days]
    if missing_observation_days:
        SWOT_OBSERVATIONS_LOGGER.warning(
            "%d day(s) without SWOT observations are skipped: %s",
            len(missing_observation_days),
            ", ".join(str(day) for day in missing_observation_days),
        )
    return converted_observation_days


def _selected_swot_observations_dataset(
    observation_days: numpy.ndarray,
    first_day_timestamps: pandas.DatetimeIndex,
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
) -> xarray.Dataset:
    time_key = Dimension.TIME.key()
    swot_observations_dataset = open_remote_multizarr(
        list(map(swot_observation_path, _converted_observation_days(observation_days))),
        parallel=False,
        concat_dim=OBSERVATION_DIMENSION_KEY,
        combine="nested",
        preprocess=_prepare_day_swot_observations_dataset,
    )
    observation_datetimes = pandas.to_datetime(swot_observations_dataset[time_key].values)
    selected_observation_indices, selected_run_indices = forecast_observation_matches(
        observation_datetimes,
        first_day_timestamps,
        lead_days_count,
    )
    return (
        swot_observations_dataset.assign_coords({time_key: (OBSERVATION_DIMENSION_KEY, observation_datetimes)})
        .isel({OBSERVATION_DIMENSION_KEY: selected_observation_indices})
        .assign_coords(
            {
                Dimension.FIRST_DAY_DATETIME.key(): (
                    (OBSERVATION_DIMENSION_KEY,),
                    first_day_datetimes[selected_run_indices],
                )
            }
        )
    )


def _swot_observations_stage_path(
    first_day_timestamps: pandas.DatetimeIndex,
    first_day_start: str,
    last_day_end: str,
    lead_days_count: int,
) -> Path:
    first_day_payload = ",".join(first_day_timestamps.sort_values().strftime("%Y-%m-%dT%H:%M:%S"))
    first_day_hash = hashlib.sha256(first_day_payload.encode("ascii")).hexdigest()[:12]
    return local_stage_directory() / (
        f"swot-observations-{SWOT_OBSERVATIONS_STAGE_VERSION}-"
        f"{first_day_start.replace('-', '')}-{last_day_end.replace('-', '')}-{lead_days_count}d-"
        f"{len(first_day_timestamps)}starts-{first_day_hash}.zarr"
    )


def _open_staged_swot_observations_dataset(stage_path: Path) -> xarray.Dataset:
    return xarray.open_dataset(stage_path, engine="zarr")


def _build_staged_swot_observations_dataset(
    stage_path: Path,
    observation_days: numpy.ndarray,
    first_day_timestamps: pandas.DatetimeIndex,
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
) -> None:
    swot_observations_dataset = _selected_swot_observations_dataset(
        observation_days=observation_days,
        first_day_timestamps=first_day_timestamps,
        first_day_datetimes=first_day_datetimes,
        lead_days_count=lead_days_count,
    )
    try:
        write_dataset_to_local_stage(
            swot_observations_dataset,
            stage_path,
            load_before_write=True,
            clear_chunk_encoding=True,
        )
    finally:
        swot_observations_dataset.close()


def swot_observations(challenger_dataset: xarray.Dataset) -> xarray.Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    first_challenger_day = first_day_datetimes.astype("datetime64[D]").min()
    if first_challenger_day < OBSERVATIONS_FIRST_AVAILABLE_DATE:
        raise ObservationDataUnavailableError(
            "SWOT-based Class IV scores were not computed for this challenger. "
            f"SWOT observation data is available from {pandas.Timestamp(OBSERVATIONS_FIRST_AVAILABLE_DATE):%Y-%m-%d}, "
            f"while challenger first_day_datetime starts on {pandas.Timestamp(first_challenger_day):%Y-%m-%d}."
        )

    first_day_timestamps = pandas.to_datetime(first_day_datetimes)
    first_day_start = first_day_timestamps.min().strftime("%Y-%m-%d")
    last_day_end = (first_day_timestamps.max() + pandas.Timedelta(days=lead_days_count - 1)).strftime("%Y-%m-%d")
    observation_days = numpy.array(generate_dates(first_day_start, last_day_end, 1), dtype="datetime64[D]")

    if not should_stage_locally(LOCAL_STAGE_OBSERVATIONS_KEY):
        return _selected_swot_observations_dataset(
            observation_days=observation_days,
            first_day_timestamps=first_day_timestamps,
            first_day_datetimes=first_day_datetimes,
            lead_days_count=lead_days_count,
        )
    return open_or_create_local_stage_dataset(
        _swot_observations_stage_path(first_day_timestamps, first_day_start, last_day_end, lead_days_count),
        open_staged_dataset=_open_staged_swot_observations_dataset,
        build_stage=lambda stage_path: _build_staged_swot_observations_dataset(
            stage_path,
            observation_days=observation_days,
            first_day_timestamps=first_day_timestamps,
            first_day_datetimes=first_day_datetimes,
            lead_days_count=lead_days_count,
        ),
    )
