# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
from xarray import Dataset, open_mfdataset
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import Dimension, LEAD_DAYS_COUNT


def observation_path(day_datetime: numpy.datetime64) -> str:
    day_string = pandas.Timestamp(day_datetime).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/obs_finales_v2/{day_string}.zarr"


def observations(challenger_dataset: Dataset) -> Dataset:
    time_key = Dimension.TIME.key()
    source_observation_dimension_key = "obs"
    observation_dimension_key = "observations"
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    first_day_timestamps = pandas.to_datetime(first_day_datetimes)
    first_day_start = first_day_timestamps.min().strftime("%Y-%m-%d")
    last_day_end = (first_day_timestamps.max() + pandas.Timedelta(days=LEAD_DAYS_COUNT)).strftime("%Y-%m-%d")
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

    observation_datetimes = pandas.to_datetime(observations_dataset[time_key].values)
    observations_dataset = observations_dataset.assign_coords(
        {time_key: (observation_dimension_key, observation_datetimes)}
    )

    first_valid_datetimes = (first_day_timestamps + pandas.Timedelta(days=1)).values[:, numpy.newaxis]
    end_datetimes_exclusive = (first_day_timestamps + pandas.Timedelta(days=LEAD_DAYS_COUNT + 1)).values[
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
            Dimension.FIRST_DAY_DATETIME.key(): (
                (observation_dimension_key,),
                selected_first_day_coord,
            )
        }
    )
