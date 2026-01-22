# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
from xarray import Dataset, open_zarr, concat
import logging
from oceanbench.core.dataset_utils import Dimension

logger = logging.getLogger("obs_insitu")
logger.setLevel(level=logging.WARNING)


import os
from pathlib import Path

# URL du fichier Zarr - essaie distant puis local
OBSERVATIONS_ZARR_URL_REMOTE = (
    "https://minio.dive.edito.eu/project-oceanbench/public/observations/observations_ALL_2days.zarr"
)

# Chemin local - peut être configuré via variable d'environnement
OBSERVATIONS_ZARR_URL_LOCAL = os.environ.get(
    "OBSERVATIONS_ZARR_PATH", str(Path.home() / "Downloads" / "observations_ALL_2days.zarr")
)

# Cache global
_OBSERVATIONS_CACHE = None


def _load_observations():
    """Load and cache the observations Zarr."""
    global _OBSERVATIONS_CACHE

    if _OBSERVATIONS_CACHE is None:
        # Essayer d'abord l'URL distante
        try:
            logger.info(f"Loading observations from remote: {OBSERVATIONS_ZARR_URL_REMOTE}")
            _OBSERVATIONS_CACHE = open_zarr(OBSERVATIONS_ZARR_URL_REMOTE).compute()
            logger.info(f"Observations loaded from remote: {len(_OBSERVATIONS_CACHE.obs)} points")
        except Exception as e:
            logger.warning(f"Failed to load from remote: {e}")
            logger.info(f"Trying local path: {OBSERVATIONS_ZARR_URL_LOCAL}")
            try:
                _OBSERVATIONS_CACHE = open_zarr(OBSERVATIONS_ZARR_URL_LOCAL).compute()
                logger.info(f"Observations loaded from local: {len(_OBSERVATIONS_CACHE.obs)} points")
            except Exception as e2:
                raise RuntimeError(
                    f"Cannot load observations from remote or local. " f"Remote error: {e}, Local error: {e2}"
                )

    return _OBSERVATIONS_CACHE


def obs_insitu_dataset(challenger_dataset: Dataset) -> Dataset:
    """
    Load in-situ observations matching the challenger dataset's time range.

    Reads from a single pre-computed Zarr file on EDITO.

    Args:
        challenger_dataset: Dataset with first_day_datetime dimension

    Returns:
        Combined dataset with observations for all time periods
    """
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    # Load the full observations dataset (cached)
    obs_full = _load_observations()

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
