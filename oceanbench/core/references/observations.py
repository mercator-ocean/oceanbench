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


def _obs_insitu_path(day_datetime: numpy.datetime64) -> str:
    """Generate path to daily observation Zarr file."""
    day = datetime.fromisoformat(str(day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-ml-compression/public/observations_by_day/{day}.zarr"


def obs_insitu_dataset(challenger_dataset: Dataset) -> Dataset:
    """
    Load in-situ observations for 10-day windows matching challenger dataset.

    For each first_day_datetime in the challenger, loads observations from
    first_day to first_day+10 days (inclusive) using individual daily Zarr files.

    Args:
        challenger_dataset: Dataset with first_day_datetime dimension

    Returns:
        Dataset with observations for all periods, with first_day_datetime coordinate
    """
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    # Generate list of all days needed (11 days per period: lead 0-10)
    all_days = set()
    for first_day in first_day_datetimes:
        for day_offset in range(11):
            day = pandas.Timestamp(first_day) + pandas.Timedelta(days=day_offset)
            all_days.add(numpy.datetime64(day.date()))

    # Generate paths for all unique days
    paths = [_obs_insitu_path(day) for day in sorted(all_days)]

    logger.info(f"Loading {len(paths)} days of observations...")

    # Load all daily Zarr files with open_mfdataset (parallel)
    obs_full = open_mfdataset(
        paths,
        engine="zarr",
        decode_cf=False,
        parallel=True,
        concat_dim="obs",
        combine="nested",
    )

    # Convert time from string to datetime
    time_dt = pandas.to_datetime(obs_full.time.values)
    obs_full = obs_full.assign_coords(time=("obs", time_dt))

    # Filter and assign first_day_datetime for each period
    all_datasets = []
    for first_day_datetime in first_day_datetimes:
        first_day = numpy.datetime64(pandas.Timestamp(first_day_datetime))
        end_day = numpy.datetime64(
            pandas.Timestamp(first_day_datetime) + pandas.Timedelta(days=10, hours=23, minutes=59)
        )

        time_mask = (obs_full.time.values >= first_day) & (obs_full.time.values <= end_day)
        ds_filtered = obs_full.isel(obs=time_mask)

        if len(ds_filtered.obs) > 0:
            ds_filtered = ds_filtered.assign_coords(
                {
                    Dimension.FIRST_DAY_DATETIME.key(): (
                        ("obs",),
                        numpy.full(len(ds_filtered.obs), first_day_datetime, dtype="datetime64[ns]"),
                    )
                }
            )
            all_datasets.append(ds_filtered)

    if not all_datasets:
        raise ValueError("No observations found for any of the requested periods")

    # Stack manually to preserve data
    variables = ["thetao", "so", "uo", "vo", "zos"]
    coords = ["time", "latitude", "longitude", "depth", "first_day_datetime"]

    stacked_data = {var: numpy.concatenate([ds[var].values for ds in all_datasets]) for var in variables + coords}

    return Dataset(
        {var: (["obs"], stacked_data[var]) for var in variables},
        coords={coord: (["obs"], stacked_data[coord]) for coord in coords},
    )
