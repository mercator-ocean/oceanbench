# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import numpy as np
from xarray import Dataset, open_dataset
import logging
from oceanbench.core.dataset_utils import Dimension

logger = logging.getLogger("obs_insitu")
logger.setLevel(level=logging.WARNING)

OBSERVATIONS_ZARR_URL = "https://minio.dive.edito.eu/project-ml-compression/public/observations_valid_only.zarr"


def obs_insitu_dataset(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    # Load observations and convert time
    obs_full = open_dataset(OBSERVATIONS_ZARR_URL, engine="zarr", decode_cf=False)
    time_dt = pandas.to_datetime(obs_full.time.values)
    obs_full = obs_full.assign_coords(time=("obs", time_dt))

    # Filter and stack for each period
    all_datasets = []
    for first_day_datetime in first_day_datetimes:
        first_day = np.datetime64(pandas.Timestamp(first_day_datetime))
        end_day = np.datetime64(pandas.Timestamp(first_day_datetime) + pandas.Timedelta(days=9, hours=23, minutes=59))

        time_mask = (obs_full.time.values >= first_day) & (obs_full.time.values <= end_day)
        ds_filtered = obs_full.isel(obs=time_mask)

        if len(ds_filtered.obs) > 0:
            ds_filtered = ds_filtered.assign_coords(
                {
                    Dimension.FIRST_DAY_DATETIME.key(): (
                        ("obs",),
                        np.full(len(ds_filtered.obs), first_day_datetime, dtype="datetime64[ns]"),
                    )
                }
            )
            all_datasets.append(ds_filtered)

    if not all_datasets:
        raise ValueError("No observations found for any of the requested periods")

    # Stack manually
    variables = ["TEMP", "PSAL", "EWCT", "NSCT", "SLEV"]
    coords = ["time", "latitude", "longitude", "depth", "first_day_datetime"]

    stacked_data = {var: np.concatenate([ds[var].values for ds in all_datasets]) for var in variables + coords}

    return Dataset(
        {var: (["obs"], stacked_data[var]) for var in variables},
        coords={coord: (["obs"], stacked_data[coord]) for coord in coords},
    )
