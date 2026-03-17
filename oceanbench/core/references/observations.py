# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import shutil

import numpy
import pandas
from xarray import Dataset, open_dataset, open_mfdataset

from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.local_stage import local_stage_build_guard, local_stage_directory, should_stage_locally
from oceanbench.core.remote_http import RetriableRemoteDataError, require_remote_dataset_dimensions

OBSERVATIONS_FIRST_AVAILABLE_DATE = numpy.datetime64("2024-01-01")
LOCAL_STAGE_OBSERVATIONS_KEY = "observations"


class ObservationDataUnavailableError(ValueError):
    pass


def observation_path(day_datetime: numpy.datetime64) -> str:
    day_string = pandas.Timestamp(day_datetime).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/observations2024/{day_string}.zarr"


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


def _observations_stage_path(first_day_start: str, last_day_end: str, lead_days_count: int) -> Path:
    return local_stage_directory() / (
        f"observations-{first_day_start.replace('-', '')}-{last_day_end.replace('-', '')}-{lead_days_count}d.zarr"
    )


def _write_staged_observations_dataset(observations_dataset: Dataset, stage_path: Path) -> None:
    staged_dataset = observations_dataset.load()
    for variable_name in staged_dataset.variables:
        staged_dataset[variable_name].encoding.pop("chunks", None)
    temporary_stage_path = stage_path.with_name(f"{stage_path.name}.tmp")
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(temporary_stage_path, ignore_errors=True)
    staged_dataset.to_zarr(temporary_stage_path, mode="w")
    shutil.rmtree(stage_path, ignore_errors=True)
    temporary_stage_path.rename(stage_path)


def _open_staged_observations_dataset(stage_path: Path) -> Dataset:
    return open_dataset(stage_path, engine="zarr")


def _selected_observations_dataset(
    observation_days: numpy.ndarray,
    first_day_timestamps: pandas.DatetimeIndex,
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
) -> Dataset:
    time_key = Dimension.TIME.key()
    source_observation_dimension_key = "obs"
    observation_dimension_key = "observations"
    first_day_datetime_key = Dimension.FIRST_DAY_DATETIME.key()

    observations_dataset = open_mfdataset(
        list(map(observation_path, observation_days)),
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

    first_valid_datetimes = (first_day_timestamps + pandas.Timedelta(days=1)).values[:, numpy.newaxis]
    end_datetimes_exclusive = (first_day_timestamps + pandas.Timedelta(days=lead_days_count + 1)).values[
        :, numpy.newaxis
    ]

    observation_window_masks = (observation_datetimes.values >= first_valid_datetimes) & (
        observation_datetimes.values < end_datetimes_exclusive
    )

    selected_observations_mask = numpy.any(observation_window_masks, axis=0)
    selected_run_indices = numpy.argmax(observation_window_masks[:, selected_observations_mask], axis=0)
    selected_first_day_coord = first_day_datetimes[selected_run_indices]

    return observations_dataset.isel({observation_dimension_key: selected_observations_mask}).assign_coords(
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
    last_day_end = (first_day_timestamps.max() + pandas.Timedelta(days=lead_days_count)).strftime("%Y-%m-%d")
    observation_days = numpy.array(generate_dates(first_day_start, last_day_end, 1), dtype="datetime64[D]")
    local_stage_path = _observations_stage_path(first_day_start, last_day_end, lead_days_count)
    if _should_stage_observations_locally() and local_stage_path.exists():
        return _open_staged_observations_dataset(local_stage_path)

    if not _should_stage_observations_locally():
        return _selected_observations_dataset(
            observation_days=observation_days,
            first_day_timestamps=first_day_timestamps,
            first_day_datetimes=first_day_datetimes,
            lead_days_count=lead_days_count,
        )
    with local_stage_build_guard(local_stage_path) as should_build_stage:
        if not should_build_stage:
            return _open_staged_observations_dataset(local_stage_path)
        observations_dataset = _selected_observations_dataset(
            observation_days=observation_days,
            first_day_timestamps=first_day_timestamps,
            first_day_datetimes=first_day_datetimes,
            lead_days_count=lead_days_count,
        )
        _write_staged_observations_dataset(observations_dataset, local_stage_path)
        return _open_staged_observations_dataset(local_stage_path)
