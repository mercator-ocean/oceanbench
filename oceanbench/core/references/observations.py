# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
from xarray import Dataset, open_mfdataset
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import Dimension, Variable

OBSERVATIONS_FIRST_AVAILABLE_DATE = numpy.datetime64("2024-01-01")


def observation_path(day_datetime: numpy.datetime64) -> str:
    day_string = pandas.Timestamp(day_datetime).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/obs_finales_v2/{day_string}.zarr"


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

    observations_dataset = open_mfdataset(
        list(map(observation_path, observation_days)),
        engine="zarr",
        decode_cf=False,
        parallel=True,
        concat_dim=source_observation_dimension_key,
        combine="nested",
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
