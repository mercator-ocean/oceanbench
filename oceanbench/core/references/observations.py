# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import numpy
import pandas
from xarray import Dataset, open_mfdataset
import logging
from oceanbench.core.classIV import LEAD_DAYS_COUNT
from oceanbench.core.dataset_utils import Dimension

logger = logging.getLogger("obs_insitu")
logger.setLevel(level=logging.WARNING)


def _observation_insitu_path(day_datetime: numpy.datetime64) -> str:
    day_string = datetime.fromisoformat(str(day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-ml-compression/public/observations_by_day/{day_string}.zarr"


def observation_insitu_dataset(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    all_days = numpy.unique(
        [
            numpy.datetime64((pandas.Timestamp(first_day) + pandas.Timedelta(days=day_offset)).date())
            for first_day in first_day_datetimes
            for day_offset in range(1, LEAD_DAYS_COUNT + 1)
        ]
    )

    observations = open_mfdataset(
        list(map(_observation_insitu_path, all_days)),
        engine="zarr",
        decode_cf=False,
        parallel=True,
        concat_dim="obs",
        combine="nested",
    )

    time_datetime = pandas.to_datetime(observations.time.values)
    observations = observations.assign_coords(time=("obs", time_datetime))

    time_values = observations.time.values
    first_days = first_day_datetimes[:, numpy.newaxis]  # Shape: (n_runs, 1)
    end_days = (
        pandas.to_datetime(first_day_datetimes) + pandas.Timedelta(days=LEAD_DAYS_COUNT, hours=23, minutes=59)
    ).values[:, numpy.newaxis]

    masks = (time_values >= first_days) & (time_values <= end_days)  # Shape: (n_runs, n_obs)

    run_indices = numpy.argmax(masks, axis=0)
    combined_mask = numpy.any(masks, axis=0)

    first_day_coord = first_day_datetimes[run_indices]

    return observations.isel(obs=combined_mask).assign_coords(
        {Dimension.FIRST_DAY_DATETIME.key(): (("obs",), first_day_coord[combined_mask])}
    )
