# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import numpy
from datetime import datetime
from xarray import Dataset, open_mfdataset
import logging
from oceanbench.core.dataset_utils import Dimension

logger = logging.getLogger("obs_insitu")
logger.setLevel(level=logging.WARNING)


def _observation_insitu_path(day_datetime: numpy.datetime64) -> str:
    day_string = datetime.fromisoformat(str(day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-ml-compression/public/observations_by_day/{day_string}.zarr"


def observation_insitu_dataset(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    all_days = set()
    for first_day_datetime in first_day_datetimes:
        for day_offset in range(11):
            day_timestamp = pandas.Timestamp(first_day_datetime) + pandas.Timedelta(days=day_offset)
            all_days.add(numpy.datetime64(day_timestamp.date()))

    observation_paths = [_observation_insitu_path(day) for day in sorted(all_days)]

    observations_full = open_mfdataset(
        observation_paths,
        engine="zarr",
        decode_cf=False,
        parallel=True,
        concat_dim="obs",
        combine="nested",
    )

    time_datetime = pandas.to_datetime(observations_full.time.values)
    observations_full = observations_full.assign_coords(time=("obs", time_datetime))

    filtered_datasets = []
    for first_day_datetime in first_day_datetimes:
        first_day = numpy.datetime64(pandas.Timestamp(first_day_datetime))
        end_day = numpy.datetime64(
            pandas.Timestamp(first_day_datetime) + pandas.Timedelta(days=10, hours=23, minutes=59)
        )

        time_mask = (observations_full.time.values >= first_day) & (observations_full.time.values <= end_day)
        dataset_filtered = observations_full.isel(obs=time_mask)

        if len(dataset_filtered.obs) > 0:
            dataset_filtered = dataset_filtered.assign_coords(
                {
                    Dimension.FIRST_DAY_DATETIME.key(): (
                        ("obs",),
                        numpy.full(len(dataset_filtered.obs), first_day_datetime, dtype="datetime64[ns]"),
                    )
                }
            )
            filtered_datasets.append(dataset_filtered)

    variables = ["thetao", "so", "uo", "vo", "zos"]
    coordinates = ["time", "latitude", "longitude", "depth", "first_day_datetime"]

    stacked_data = {
        variable: numpy.concatenate([dataset[variable].values for dataset in filtered_datasets])
        for variable in variables + coordinates
    }

    return Dataset(
        {variable: (["obs"], stacked_data[variable]) for variable in variables},
        coords={coordinate: (["obs"], stacked_data[coordinate]) for coordinate in coordinates},
    )
