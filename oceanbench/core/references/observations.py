# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
from xarray import Dataset, open_dataset, concat
import logging
from oceanbench.core.dataset_utils import Dimension

logger = logging.getLogger("obs_insitu")
logger.setLevel(level=logging.WARNING)


# URL du Zarr sur EDITO
OBSERVATIONS_ZARR_URL = "https://minio.dive.edito.eu/project-ml-compression/public/observations_ALL_2daysV2.zarr"


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

    # Load the full observations dataset
    logger.info(f"Loading observations from EDITO: {OBSERVATIONS_ZARR_URL}")
    obs_full = open_dataset(OBSERVATIONS_ZARR_URL, engine="zarr")

    # Filter and concatenate for each first_day_datetime
    datasets = []
    for first_day_datetime in first_day_datetimes:
        # Convert to datetime
        first_day = pandas.Timestamp(first_day_datetime).to_pydatetime()
        end_day = first_day + pandas.Timedelta(days=9, hours=23, minutes=59)

        # Filter by time
        time_mask = (obs_full.time >= first_day) & (obs_full.time <= end_day)
        ds_filtered = obs_full.isel(obs=time_mask.values)

        if len(ds_filtered.obs) > 0:
            datasets.append(ds_filtered)
            logger.info(f"Found {len(ds_filtered.obs)} observations for {first_day.date()}")
        else:
            logger.warning(f"No observations found for {first_day.date()}")

    if not datasets:
        raise ValueError("No observations found for any of the requested periods")

    # Concatenate along first_day_datetime dimension
    combined_dataset = concat(datasets, dim=Dimension.FIRST_DAY_DATETIME.key()).assign_coords(
        {Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes}
    )

    return combined_dataset
