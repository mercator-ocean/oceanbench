# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from functools import lru_cache
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import numpy
import pandas
from xarray import Dataset, open_dataset, open_mfdataset

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.local_stage import (
    local_stage_directory,
    open_or_create_local_stage_dataset,
    should_stage_locally,
    write_dataset_to_local_stage,
)
from oceanbench.core.remote_http import (
    RetriableRemoteDataError,
    require_remote_dataset_dimensions,
    with_remote_http_retries,
)

DEFAULT_OBSERVATIONS_COLLECTION = "observations2024"
OBSERVATIONS_BASE_URI_BY_COLLECTION = {
    "observations2024": "https://minio.dive.edito.eu/project-oceanbench/public/observations2024",
    "observations2026": "https://minio.dive.edito.eu/project-oceanbench/public/observations2026",
}
OBSERVATIONS_BASE_URI_ENVIRONMENT_VARIABLE_BY_COLLECTION = {
    "observations2026": OceanbenchEnvironmentVariable.OCEANBENCH_OBSERVATIONS_2026_BASE_URI,
}
OBSERVATIONS_FIRST_AVAILABLE_DATES = {
    "observations2024": numpy.datetime64("2024-01-01"),
    "observations2026": numpy.datetime64("2026-01-01"),
}
OBSERVATIONS_FIRST_AVAILABLE_DATE = OBSERVATIONS_FIRST_AVAILABLE_DATES[DEFAULT_OBSERVATIONS_COLLECTION]
LOCAL_STAGE_OBSERVATIONS_KEY = "observations"
OBSERVATIONS_STAGE_VERSION = "v4"
OBSERVATION_DATASET_EXISTS_TIMEOUT_SECONDS = 10


class ObservationDataUnavailableError(ValueError):
    pass


def _mean_dynamic_topography_zarr_url(resolution: str) -> str:
    if resolution == "thirty_sixth_degree":
        resolution = "twelfth_degree"
    if resolution == "twelfth_degree":
        return "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_MDT/" "GLO-MFC_001_024_mdt.zarr"
    if resolution == "quarter_degree":
        return "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_MDT/" "GLO-MFC_001_024_mdt_025deg.zarr"
    if resolution == "one_degree":
        return "https://minio.dive.edito.eu/project-oceanbench/public/GLO12_MDT/" "GLO-MFC_001_024_mdt_1_deg.zarr"
    raise ValueError(f"Unsupported resolution : {resolution}.")


def _mean_dynamic_topography_stage_path(resolution: str) -> Path:
    if resolution == "thirty_sixth_degree":
        resolution = "twelfth_degree"
    return local_stage_directory() / f"class4-mean-dynamic-topography-2024-glo12-{resolution}.zarr"


def _open_staged_mean_dynamic_topography_dataset(stage_path: Path) -> Dataset:
    return open_dataset(stage_path, engine="zarr")


def _build_staged_mean_dynamic_topography_dataset(
    mean_dynamic_topography_url: str,
    stage_path: Path,
) -> None:
    mean_dynamic_topography_dataset = open_dataset(
        mean_dynamic_topography_url,
        engine="zarr",
        chunks="auto",
        consolidated=True,
    )
    try:
        write_dataset_to_local_stage(mean_dynamic_topography_dataset, stage_path)
    finally:
        mean_dynamic_topography_dataset.close()


def load_mean_dynamic_topography(resolution: str) -> Dataset:
    def open_mean_dynamic_topography_dataset() -> Dataset:
        mean_dynamic_topography_url = _mean_dynamic_topography_zarr_url(resolution)
        if not should_stage_locally(LOCAL_STAGE_OBSERVATIONS_KEY):
            return open_dataset(
                mean_dynamic_topography_url,
                engine="zarr",
                chunks="auto",
                consolidated=True,
            )
        local_stage_path = _mean_dynamic_topography_stage_path(resolution)
        return open_or_create_local_stage_dataset(
            local_stage_path,
            open_staged_dataset=_open_staged_mean_dynamic_topography_dataset,
            build_stage=lambda stage_path: _build_staged_mean_dynamic_topography_dataset(
                mean_dynamic_topography_url,
                stage_path,
            ),
        )

    dataset = with_remote_http_retries("mean dynamic topography open", open_mean_dynamic_topography_dataset)
    dataset = rename_dataset_with_standard_names(dataset)
    return dataset[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()]


def _observations_collection_for_challenger(first_challenger_day: numpy.datetime64) -> str:
    if first_challenger_day >= OBSERVATIONS_FIRST_AVAILABLE_DATES["observations2026"]:
        return "observations2026"
    return DEFAULT_OBSERVATIONS_COLLECTION


def _is_http_uri(uri: str) -> bool:
    return urlparse(uri).scheme in {"http", "https"}


def _observation_base_uri(observations_collection: str) -> str:
    try:
        default_base_uri = OBSERVATIONS_BASE_URI_BY_COLLECTION[observations_collection]
    except KeyError as error:
        supported_collections = ", ".join(sorted(OBSERVATIONS_BASE_URI_BY_COLLECTION))
        raise ValueError(
            f"Unsupported observation collection: {observations_collection!r}. "
            f"Supported values are: {supported_collections}."
        ) from error
    environment_variable = OBSERVATIONS_BASE_URI_ENVIRONMENT_VARIABLE_BY_COLLECTION.get(observations_collection)
    if environment_variable is None:
        return default_base_uri
    configured_base_uri = os.environ.get(environment_variable.value)
    if configured_base_uri:
        return configured_base_uri.rstrip("/")
    return default_base_uri


def observation_path(
    day_datetime: numpy.datetime64,
    observations_collection: str = DEFAULT_OBSERVATIONS_COLLECTION,
) -> str:
    day_string = pandas.Timestamp(day_datetime).strftime("%Y%m%d")
    return f"{_observation_base_uri(observations_collection)}/{day_string}.zarr"


@lru_cache(maxsize=4096)
def _observation_dataset_exists_uri(observation_uri: str) -> bool:
    if not _is_http_uri(observation_uri):
        return Path(observation_uri).exists()
    request = Request(f"{observation_uri.rstrip('/')}/.zmetadata", method="HEAD")
    try:
        with urlopen(request, timeout=OBSERVATION_DATASET_EXISTS_TIMEOUT_SECONDS):
            return True
    except HTTPError as error:
        if error.code == 404:
            return False
        raise


def _observation_dataset_exists(
    day_datetime: numpy.datetime64,
    observations_collection: str,
) -> bool:
    return _observation_dataset_exists_uri(observation_path(day_datetime, observations_collection))


def _available_observation_days(
    observation_days: numpy.ndarray,
    observations_collection: str,
) -> numpy.ndarray:
    if observations_collection != "observations2026":
        return observation_days
    available_days = [day for day in observation_days if _observation_dataset_exists(day, observations_collection)]
    if available_days:
        return numpy.array(available_days, dtype="datetime64[D]")
    first_day_string = pandas.Timestamp(observation_days[0]).strftime("%Y-%m-%d")
    last_day_string = pandas.Timestamp(observation_days[-1]).strftime("%Y-%m-%d")
    raise ObservationDataUnavailableError(
        f"No {observations_collection} data files were available " f"between {first_day_string} and {last_day_string}."
    )


def _assign_standard_names(observations_dataset: Dataset) -> Dataset:
    standard_name_keys = [
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
        Variable.SEA_WATER_SALINITY.key(),
        Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
        Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
        Dimension.TIME.key(),
        Dimension.DEPTH.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    for standard_name_key in standard_name_keys:
        observations_dataset[standard_name_key].attrs["standard_name"] = standard_name_key
    return observations_dataset


def _should_stage_observations_locally() -> bool:
    return should_stage_locally(LOCAL_STAGE_OBSERVATIONS_KEY)


def _observations_stage_path(
    first_day_start: str,
    last_day_end: str,
    lead_days_count: int,
    observations_collection: str = DEFAULT_OBSERVATIONS_COLLECTION,
) -> Path:
    return local_stage_directory() / (
        f"observations-{OBSERVATIONS_STAGE_VERSION}-{observations_collection}-"
        f"{first_day_start.replace('-', '')}-{last_day_end.replace('-', '')}-{lead_days_count}d.zarr"
    )


def _open_staged_observations_dataset(stage_path: Path) -> Dataset:
    return open_dataset(stage_path, engine="zarr")


def _build_staged_observations_dataset(
    stage_path: Path,
    observation_days: numpy.ndarray,
    first_day_timestamps: pandas.DatetimeIndex,
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
    observations_collection: str = DEFAULT_OBSERVATIONS_COLLECTION,
) -> None:
    observations_dataset = _selected_observations_dataset(
        observation_days=observation_days,
        first_day_timestamps=first_day_timestamps,
        first_day_datetimes=first_day_datetimes,
        lead_days_count=lead_days_count,
        observations_collection=observations_collection,
    )
    try:
        write_dataset_to_local_stage(
            observations_dataset,
            stage_path,
            load_before_write=True,
            clear_chunk_encoding=True,
        )
    finally:
        observations_dataset.close()


def _forecast_observation_matches(
    observation_datetimes: pandas.DatetimeIndex,
    first_day_timestamps: pandas.DatetimeIndex,
    lead_days_count: int,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    observation_values = observation_datetimes.values
    selected_observation_chunks = []
    selected_run_chunks = []
    for run_index, first_day_timestamp in enumerate(first_day_timestamps):
        first_valid_datetime = first_day_timestamp.to_datetime64()
        end_datetime_exclusive = (first_day_timestamp + pandas.Timedelta(days=lead_days_count)).to_datetime64()
        selected_observation_indices = numpy.flatnonzero(
            (observation_values >= first_valid_datetime) & (observation_values < end_datetime_exclusive)
        )
        if selected_observation_indices.size == 0:
            continue
        selected_observation_chunks.append(selected_observation_indices)
        selected_run_chunks.append(numpy.full(selected_observation_indices.size, run_index, dtype=numpy.intp))

    if not selected_observation_chunks:
        return numpy.array([], dtype=numpy.intp), numpy.array([], dtype=numpy.intp)
    return numpy.concatenate(selected_observation_chunks), numpy.concatenate(selected_run_chunks)


def _selected_observations_dataset(
    observation_days: numpy.ndarray,
    first_day_timestamps: pandas.DatetimeIndex,
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
    observations_collection: str = DEFAULT_OBSERVATIONS_COLLECTION,
) -> Dataset:
    time_key = Dimension.TIME.key()
    source_observation_dimension_key = "obs"
    observation_dimension_key = "observations"
    first_day_datetime_key = Dimension.FIRST_DAY_DATETIME.key()

    observations_dataset = open_mfdataset(
        [observation_path(day, observations_collection) for day in observation_days],
        engine="zarr",
        decode_cf=False,
        parallel=False,
        concat_dim=source_observation_dimension_key,
        combine="nested",
    )
    observations_dataset = require_remote_dataset_dimensions(
        observations_dataset,
        [source_observation_dimension_key],
        "observation dataset open",
    )
    if time_key not in observations_dataset.variables:
        raise RetriableRemoteDataError(
            f"Remote dataset opened without expected variable {time_key!r} during observation dataset open. "
            f"Available variables: {sorted(observations_dataset.variables)}"
        )
    observations_dataset = observations_dataset.rename({source_observation_dimension_key: observation_dimension_key})
    observations_dataset = _assign_standard_names(observations_dataset)

    observation_datetimes = pandas.to_datetime(observations_dataset[time_key].values)
    observations_dataset = observations_dataset.assign_coords(
        {time_key: (observation_dimension_key, observation_datetimes)}
    )

    selected_observation_indices, selected_run_indices = _forecast_observation_matches(
        observation_datetimes,
        first_day_timestamps,
        lead_days_count,
    )
    selected_first_day_coord = first_day_datetimes[selected_run_indices]

    return observations_dataset.isel({observation_dimension_key: selected_observation_indices}).assign_coords(
        {
            first_day_datetime_key: (
                (observation_dimension_key,),
                selected_first_day_coord,
            )
        }
    )


def observations(challenger_dataset: Dataset) -> Dataset:
    lead_day_index_key = Dimension.LEAD_DAY_INDEX.key()
    first_day_datetime_key = Dimension.FIRST_DAY_DATETIME.key()
    first_day_datetimes = challenger_dataset[first_day_datetime_key].values
    lead_days_count = challenger_dataset.sizes[lead_day_index_key]
    first_day_dates = first_day_datetimes.astype("datetime64[D]")
    first_challenger_day = first_day_dates.min()
    last_challenger_day = first_day_dates.max() + numpy.timedelta64(lead_days_count - 1, "D")
    observations_collection = _observations_collection_for_challenger(first_challenger_day)
    observations_first_available_date = OBSERVATIONS_FIRST_AVAILABLE_DATES[observations_collection]
    if last_challenger_day < observations_first_available_date:
        last_challenger_day_string = pandas.Timestamp(last_challenger_day).strftime("%Y-%m-%d")
        first_available_day_string = pandas.Timestamp(observations_first_available_date).strftime("%Y-%m-%d")
        raise ObservationDataUnavailableError(
            "Observation-based Class IV scores were not computed for this challenger. "
            f"{observations_collection} data is available from {first_available_day_string}, "
            f"while challenger forecast windows end on {last_challenger_day_string}."
        )

    first_day_timestamps = pandas.to_datetime(first_day_datetimes)
    observation_start_day = max(first_challenger_day, observations_first_available_date)
    first_day_start = pandas.Timestamp(observation_start_day).strftime("%Y-%m-%d")
    last_day_end = (first_day_timestamps.max() + pandas.Timedelta(days=lead_days_count - 1)).strftime("%Y-%m-%d")
    observation_days = numpy.array(generate_dates(first_day_start, last_day_end, 1), dtype="datetime64[D]")
    observation_days = _available_observation_days(observation_days, observations_collection)
    local_stage_path = _observations_stage_path(first_day_start, last_day_end, lead_days_count, observations_collection)

    def open_selected_observations() -> Dataset:
        if not _should_stage_observations_locally():
            return _selected_observations_dataset(
                observation_days=observation_days,
                first_day_timestamps=first_day_timestamps,
                first_day_datetimes=first_day_datetimes,
                lead_days_count=lead_days_count,
                observations_collection=observations_collection,
            )
        return open_or_create_local_stage_dataset(
            local_stage_path,
            open_staged_dataset=_open_staged_observations_dataset,
            build_stage=lambda stage_path: _build_staged_observations_dataset(
                stage_path,
                observation_days=observation_days,
                first_day_timestamps=first_day_timestamps,
                first_day_datetimes=first_day_datetimes,
                lead_days_count=lead_days_count,
                observations_collection=observations_collection,
            ),
        )

    return with_remote_http_retries("observation dataset open", open_selected_observations)
