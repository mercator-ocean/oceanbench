# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
from os import environ
from urllib.parse import unquote, urlparse
import hashlib

import numpy
import pandas
from xarray import Dataset, open_dataset, open_mfdataset

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
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

OBSERVATIONS_FIRST_AVAILABLE_DATE = numpy.datetime64("2024-01-01")
LOCAL_STAGE_OBSERVATIONS_KEY = "observations"
OBSERVATIONS_STAGE_VERSION = "v2"
DEFAULT_OBSERVATION_ZARR_TEMPLATE = "https://minio.dive.edito.eu/project-oceanbench/public/observations2024/{day}.zarr"


class ObservationDataUnavailableError(ValueError):
    pass


def _mean_dynamic_topography_zarr_url(resolution: str) -> str:
    if resolution == "twelfth_degree":
        return "https://minio.dive.edito.eu/project-oceanbench/public/glorys12_mdt_2024/" "GLO-MFC_001_030_mdt.zarr"
    if resolution == "quarter_degree":
        return (
            "https://minio.dive.edito.eu/project-oceanbench/public/glorys12_mdt_2024/" "GLO-MFC_001_030_mdt_025deg.zarr"
        )
    if resolution == "one_degree":
        return (
            "https://minio.dive.edito.eu/project-oceanbench/public/glorys12_mdt_2024/" "GLO-MFC_001_030_mdt_1_deg.zarr"
        )
    if resolution == "thirty_sixth_degree":
        # IBI 1/36 regional MDT from the IBI reanalysis (IBI_MULTIYEAR_PHY_005_002,
        # cmems_mod_ibi_phy_my_0.027deg-3D_static, part 'mdt') the IBI model was trained on.
        return (
            "https://minio.dive.edito.eu/project-oceanbench/public/ibi36_mdt_2024/"
            "cmems_mod_ibi_phy_my_0.027deg_mdt.zarr"
        )
    raise ValueError(f"Unsupported resolution : {resolution}.")


def _mean_dynamic_topography_stage_path(resolution: str) -> Path:
    return local_stage_directory() / f"class4-mean-dynamic-topography-2024-{resolution}.zarr"


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


def _configured_observation_zarr_template() -> str:
    return environ.get(
        OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_OBSERVATION_ZARR_TEMPLATE.value,
        DEFAULT_OBSERVATION_ZARR_TEMPLATE,
    )


def _format_observation_zarr_template(
    day_datetime: numpy.datetime64,
    zarr_template: str | None,
) -> str:
    day_string = pandas.Timestamp(day_datetime).strftime("%Y%m%d")
    date_string = pandas.Timestamp(day_datetime).strftime("%Y-%m-%d")
    template = zarr_template or _configured_observation_zarr_template()
    path = template.format(
        compact_date=day_string,
        day=day_string,
        date=date_string,
        yyyymmdd=day_string,
        YYYYMMDD=day_string,
    )
    if path.startswith("file://"):
        parsed_path = urlparse(path)
        return unquote(parsed_path.path)
    return path


def observation_path(
    day_datetime: numpy.datetime64,
    zarr_template: str | None = None,
) -> str:
    return _format_observation_zarr_template(day_datetime, zarr_template)


def _is_empty_configuration_value(value) -> bool:
    return value is None or (isinstance(value, str) and value == "")


def _parse_day(day: str | numpy.datetime64 | pandas.Timestamp | None) -> numpy.datetime64 | None:
    if _is_empty_configuration_value(day):
        return None
    return numpy.datetime64(pandas.Timestamp(day).strftime("%Y-%m-%d"))


def _configured_last_available_day() -> numpy.datetime64 | None:
    return _parse_day(
        environ.get(
            OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_OBSERVATION_LAST_DAY.value,
        )
    )


def _observation_source_stage_identifier(
    zarr_template: str | None,
    last_available_day: str | numpy.datetime64 | pandas.Timestamp | None,
) -> str:
    configured_template = (
        zarr_template
        if not _is_empty_configuration_value(zarr_template)
        else environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_OBSERVATION_ZARR_TEMPLATE.value)
    )
    configured_last_day = (
        last_available_day
        if not _is_empty_configuration_value(last_available_day)
        else environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_OBSERVATION_LAST_DAY.value)
    )
    if _is_empty_configuration_value(configured_template) and _is_empty_configuration_value(configured_last_day):
        return ""
    source_key = f"{configured_template or DEFAULT_OBSERVATION_ZARR_TEMPLATE}|{configured_last_day or ''}"
    return hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12]


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
    source_identifier: str = "",
) -> Path:
    source_suffix = f"-{source_identifier}" if source_identifier else ""
    return local_stage_directory() / (
        f"observations-{OBSERVATIONS_STAGE_VERSION}-"
        f"{first_day_start.replace('-', '')}-{last_day_end.replace('-', '')}-{lead_days_count}d"
        f"{source_suffix}.zarr"
    )


def _open_staged_observations_dataset(stage_path: Path) -> Dataset:
    return open_dataset(stage_path, engine="zarr")


def _build_staged_observations_dataset(
    stage_path: Path,
    observation_days: numpy.ndarray,
    first_day_timestamps: pandas.DatetimeIndex,
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
    zarr_template: str | None,
) -> None:
    observations_dataset = _selected_observations_dataset(
        observation_days=observation_days,
        first_day_timestamps=first_day_timestamps,
        first_day_datetimes=first_day_datetimes,
        lead_days_count=lead_days_count,
        zarr_template=zarr_template,
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
        first_valid_datetime = (first_day_timestamp + pandas.Timedelta(days=1)).to_datetime64()
        end_datetime_exclusive = (first_day_timestamp + pandas.Timedelta(days=lead_days_count + 1)).to_datetime64()
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
    zarr_template: str | None = None,
) -> Dataset:
    time_key = Dimension.TIME.key()
    source_observation_dimension_key = "obs"
    observation_dimension_key = "observations"
    first_day_datetime_key = Dimension.FIRST_DAY_DATETIME.key()

    observations_dataset = open_mfdataset(
        [observation_path(observation_day, zarr_template) for observation_day in observation_days],
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


def _capped_last_day_end(
    forecast_last_day_end: numpy.datetime64,
    last_available_day: str | numpy.datetime64 | pandas.Timestamp | None,
) -> numpy.datetime64:
    parsed_last_available_day = _parse_day(last_available_day)
    if parsed_last_available_day is None:
        parsed_last_available_day = _configured_last_available_day()
    if parsed_last_available_day is None:
        return forecast_last_day_end
    return min(forecast_last_day_end, parsed_last_available_day)


def observations(
    challenger_dataset: Dataset,
    zarr_template: str | None = None,
    last_available_day: str | numpy.datetime64 | pandas.Timestamp | None = None,
) -> Dataset:
    lead_day_index_key = Dimension.LEAD_DAY_INDEX.key()
    first_day_datetime_key = Dimension.FIRST_DAY_DATETIME.key()
    first_day_datetimes = challenger_dataset[first_day_datetime_key].values
    lead_days_count = challenger_dataset.sizes[lead_day_index_key]
    first_day_dates = first_day_datetimes.astype("datetime64[D]")
    first_challenger_day = first_day_dates.min()
    if first_challenger_day < OBSERVATIONS_FIRST_AVAILABLE_DATE:
        first_challenger_day_string = pandas.Timestamp(first_challenger_day).strftime("%Y-%m-%d")
        first_available_day_string = pandas.Timestamp(OBSERVATIONS_FIRST_AVAILABLE_DATE).strftime("%Y-%m-%d")
        raise ObservationDataUnavailableError(
            "Observation-based Class IV scores were not computed for this challenger. "
            f"Observation data is available from {first_available_day_string}, "
            f"while challenger first_day_datetime starts on {first_challenger_day_string}."
        )

    first_day_timestamps = pandas.to_datetime(first_day_datetimes)
    first_day_start = first_day_timestamps.min().strftime("%Y-%m-%d")
    first_observation_day_start = (first_day_timestamps.min() + pandas.Timedelta(days=1)).strftime("%Y-%m-%d")
    forecast_last_day_end = (
        (first_day_timestamps.max() + pandas.Timedelta(days=lead_days_count)).to_datetime64().astype("datetime64[D]")
    )
    capped_last_day_end = _capped_last_day_end(forecast_last_day_end, last_available_day)
    if capped_last_day_end < first_challenger_day:
        last_available_day_string = pandas.Timestamp(capped_last_day_end).strftime("%Y-%m-%d")
        raise ObservationDataUnavailableError(
            "Observation-based Class IV scores were not computed for this challenger. "
            f"Observation data is capped at {last_available_day_string}, "
            f"while challenger first_day_datetime starts on {first_day_start}."
        )
    last_day_end = pandas.Timestamp(capped_last_day_end).strftime("%Y-%m-%d")
    observation_days = numpy.array(generate_dates(first_observation_day_start, last_day_end, 1), dtype="datetime64[D]")
    source_identifier = _observation_source_stage_identifier(zarr_template, last_available_day)
    local_stage_path = _observations_stage_path(first_day_start, last_day_end, lead_days_count, source_identifier)

    def open_selected_observations() -> Dataset:
        if not _should_stage_observations_locally():
            return _selected_observations_dataset(
                observation_days=observation_days,
                first_day_timestamps=first_day_timestamps,
                first_day_datetimes=first_day_datetimes,
                lead_days_count=lead_days_count,
                zarr_template=zarr_template,
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
                zarr_template=zarr_template,
            ),
        )

    return with_remote_http_retries("observation dataset open", open_selected_observations)
