# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
from xarray import Dataset, open_mfdataset
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.memory_diagnostics import default_memory_tracker, describe_dataset

OBSERVATIONS_FIRST_AVAILABLE_DATE = numpy.datetime64("2024-01-01")
OBSERVATION_FIRST_DAY_INDEX_KEY = "first_day_index"
OBSERVATION_FIRST_DAY_LOOKUP_KEY = "first_day_datetime_lookup"
OBSERVATION_FIRST_DAY_LOOKUP_DIMENSION_KEY = "first_day_lookup_index"
OBSERVATION_LEAD_DAY_KEY = "lead_day"
OBSERVATION_SELECTION_BLOCK_SIZE = 1_000_000
_memory_tracker = default_memory_tracker("reference_observations")


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


def observations(challenger_dataset: Dataset) -> Dataset:
    with _memory_tracker.step("observations_pipeline"):
        return _observations(challenger_dataset)


def _observations(challenger_dataset: Dataset) -> Dataset:
    describe_dataset(challenger_dataset, "challenger_dataset_for_observations", _memory_tracker)
    time_key = Dimension.TIME.key()
    lead_day_index_key = Dimension.LEAD_DAY_INDEX.key()
    first_day_datetime_key = Dimension.FIRST_DAY_DATETIME.key()
    source_observation_dimension_key = "obs"
    observation_dimension_key = "observations"
    first_day_datetimes = challenger_dataset[first_day_datetime_key].values
    lead_days_count = challenger_dataset.sizes[lead_day_index_key]
    first_day_dates = first_day_datetimes.astype("datetime64[D]")
    first_challenger_day = first_day_dates.min()
    if first_challenger_day < OBSERVATIONS_FIRST_AVAILABLE_DATE:
        first_challenger_day_string = pandas.Timestamp(first_challenger_day).strftime("%Y-%m-%d")
        first_available_day_string = pandas.Timestamp(OBSERVATIONS_FIRST_AVAILABLE_DATE).strftime("%Y-%m-%d")
        raise ValueError(
            "OBSERVATIONS_NOT_AVAILABLE: Observation-based Class IV scores were not computed for this challenger. "
            f"Observation data is available from {first_available_day_string}, "
            f"while challenger first_day_datetime starts on {first_challenger_day_string}."
        )

    first_day_timestamps = pandas.to_datetime(first_day_datetimes)
    first_day_start = first_day_timestamps.min().strftime("%Y-%m-%d")
    last_day_end = (first_day_timestamps.max() + pandas.Timedelta(days=lead_days_count)).strftime("%Y-%m-%d")
    observation_days = numpy.array(generate_dates(first_day_start, last_day_end, 1), dtype="datetime64[D]")

    _memory_tracker.checkpoint(f"observation_days_count={len(observation_days)}")
    with _memory_tracker.step("open_observations_mfdataset"):
        observations_dataset = open_mfdataset(
            list(map(observation_path, observation_days)),
            engine="zarr",
            decode_cf=False,
            parallel=True,
            concat_dim=source_observation_dimension_key,
            combine="nested",
        )
    with _memory_tracker.step("normalize_observations_dataset"):
        observations_dataset = observations_dataset.rename(
            {source_observation_dimension_key: observation_dimension_key}
        )
        observations_dataset = _assign_standard_names(observations_dataset)
    describe_dataset(observations_dataset, "observations_loaded", _memory_tracker)

    with _memory_tracker.step("build_observation_selection_mask"):
        observation_count = observations_dataset.sizes[observation_dimension_key]
        selected_observations_mask = numpy.zeros(observation_count, dtype=bool)
        selected_run_indices = numpy.full(observation_count, -1, dtype=numpy.int16)
        selected_lead_days = numpy.full(observation_count, -1, dtype=numpy.int16)

        first_valid_datetimes = (first_day_timestamps + pandas.Timedelta(days=1)).to_numpy(dtype="datetime64[ns]")
        end_datetimes_exclusive = (first_day_timestamps + pandas.Timedelta(days=lead_days_count + 1)).to_numpy(
            dtype="datetime64[ns]"
        )

        time_variable = observations_dataset[time_key]
        total_blocks = (observation_count + OBSERVATION_SELECTION_BLOCK_SIZE - 1) // OBSERVATION_SELECTION_BLOCK_SIZE

        for block_index, start_index in enumerate(
            range(0, observation_count, OBSERVATION_SELECTION_BLOCK_SIZE), start=1
        ):
            stop_index = min(start_index + OBSERVATION_SELECTION_BLOCK_SIZE, observation_count)
            with _memory_tracker.step(f"selection_block {block_index}/{total_blocks} [{start_index}:{stop_index}]"):
                time_chunk = pandas.to_datetime(
                    time_variable.isel({observation_dimension_key: slice(start_index, stop_index)}).values
                ).to_numpy(dtype="datetime64[ns]")

                # Vectorized mapping to the earliest forecast run that contains each observation datetime.
                candidate_indices = numpy.searchsorted(end_datetimes_exclusive, time_chunk, side="right")
                valid_indices_mask = candidate_indices < len(first_valid_datetimes)
                bounded_indices = numpy.clip(candidate_indices, 0, len(first_valid_datetimes) - 1)
                starts_before_time = time_chunk >= first_valid_datetimes[bounded_indices]
                chunk_selected_mask = valid_indices_mask & starts_before_time
                chunk_run_indices = numpy.where(chunk_selected_mask, bounded_indices, -1).astype(numpy.int16)
                chunk_lead_days = numpy.full(time_chunk.shape, -1, dtype=numpy.int16)
                if numpy.any(chunk_selected_mask):
                    selected_times = time_chunk[chunk_selected_mask]
                    selected_starts = first_valid_datetimes[bounded_indices[chunk_selected_mask]]
                    lead_days = ((selected_times - selected_starts) // numpy.timedelta64(1, "D")).astype(numpy.int16)
                    chunk_lead_days[chunk_selected_mask] = lead_days

                selected_observations_mask[start_index:stop_index] = chunk_selected_mask
                selected_run_indices[start_index:stop_index] = chunk_run_indices
                selected_lead_days[start_index:stop_index] = chunk_lead_days

        selected_run_indices = selected_run_indices[selected_observations_mask]
        selected_lead_days = selected_lead_days[selected_observations_mask]
        _memory_tracker.emit(f"selected_observations_count={int(selected_observations_mask.sum())}")

    with _memory_tracker.step("select_observations_for_challenger_windows"):
        selected_observations = observations_dataset.isel(
            {observation_dimension_key: selected_observations_mask}
        ).assign_coords(
            {
                OBSERVATION_FIRST_DAY_INDEX_KEY: (
                    (observation_dimension_key,),
                    selected_run_indices,
                ),
                OBSERVATION_LEAD_DAY_KEY: (
                    (observation_dimension_key,),
                    selected_lead_days,
                ),
                OBSERVATION_FIRST_DAY_LOOKUP_KEY: (
                    (OBSERVATION_FIRST_DAY_LOOKUP_DIMENSION_KEY,),
                    first_day_datetimes,
                ),
            }
        )
    describe_dataset(selected_observations, "observations_selected", _memory_tracker)
    return selected_observations
