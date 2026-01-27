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


# URL du Zarr sur EDITO
OBSERVATIONS_ZARR_URL = "https://minio.dive.edito.eu/project-ml-compression/public/observations_valid_only.zarr"


def obs_insitu_dataset(challenger_dataset: Dataset) -> Dataset:
    """
    Load in-situ observations matching the challenger dataset's time range.

    Reads from pre-computed Zarr file on EDITO.

    Args:
        challenger_dataset: Dataset with first_day_datetime dimension

    Returns:
        Combined dataset with observations for all time periods
    """
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    # Load observations WITHOUT automatic decoding
    logger.info(f"Loading observations from EDITO: {OBSERVATIONS_ZARR_URL}")
    obs_full = open_dataset(OBSERVATIONS_ZARR_URL, engine="zarr", decode_cf=False)

    # Convert time from string to datetime64
    logger.info("Converting time from string to datetime...")
    time_dt = pandas.to_datetime(obs_full.time.values)
    obs_full = obs_full.assign_coords(time=("obs", time_dt))
    logger.info(f"Time converted, dtype: {obs_full.time.dtype}")

    # Filter and stack for each first_day_datetime
    all_datasets = []
    for first_day_datetime in first_day_datetimes:
        # Convert to numpy.datetime64 for compatibility
        first_day = np.datetime64(pandas.Timestamp(first_day_datetime))
        end_day = np.datetime64(pandas.Timestamp(first_day_datetime) + pandas.Timedelta(days=9, hours=23, minutes=59))

        # Filter by time using .values for direct numpy comparison
        time_mask = (obs_full.time.values >= first_day) & (obs_full.time.values <= end_day)
        ds_filtered = obs_full.isel(obs=time_mask)

        if len(ds_filtered.obs) > 0:
            # Add first_day_datetime as a COORDINATE (not dimension)
            ds_filtered = ds_filtered.assign_coords(
                {
                    Dimension.FIRST_DAY_DATETIME.key(): (
                        ("obs",),
                        np.full(len(ds_filtered.obs), first_day_datetime, dtype="datetime64[ns]"),
                    )
                }
            )
            all_datasets.append(ds_filtered)
            logger.info(f"Found {len(ds_filtered.obs)} observations for {pandas.Timestamp(first_day).date()}")
        else:
            logger.warning(f"No observations found for {pandas.Timestamp(first_day).date()}")

    if not all_datasets:
        raise ValueError("No observations found for any of the requested periods")

    # Stack MANUELLEMENT to preserve data (concat doesn't work properly)
    logger.info("Stacking datasets manually...")

    # Extract arrays from each dataset
    stacked_data = {}
    variables = ["TEMP", "PSAL", "EWCT", "NSCT", "SLEV"]
    coords = ["time", "latitude", "longitude", "depth", "first_day_datetime"]

    for var in variables + coords:
        arrays = [ds[var].values for ds in all_datasets]
        stacked_data[var] = np.concatenate(arrays)
        logger.info(f"  {var}: {len(stacked_data[var])} total points")

    # Recreate dataset with stacked data
    combined_dataset = Dataset(
        {
            "TEMP": (["obs"], stacked_data["TEMP"]),
            "PSAL": (["obs"], stacked_data["PSAL"]),
            "EWCT": (["obs"], stacked_data["EWCT"]),
            "NSCT": (["obs"], stacked_data["NSCT"]),
            "SLEV": (["obs"], stacked_data["SLEV"]),
        },
        coords={
            "time": (["obs"], stacked_data["time"]),
            "latitude": (["obs"], stacked_data["latitude"]),
            "longitude": (["obs"], stacked_data["longitude"]),
            "depth": (["obs"], stacked_data["depth"]),
            "first_day_datetime": (["obs"], stacked_data["first_day_datetime"]),
        },
    )

    # Copy attributes from original
    combined_dataset.attrs = obs_full.attrs

    logger.info(f"Combined dataset created: {len(combined_dataset.obs)} total observations")

    return combined_dataset
