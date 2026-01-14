# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2


import pandas
from xarray import Dataset, concat
import logging
import copernicusmarine

logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)


def _obs_insitu_path(first_day_datetime, target_depths=None) -> Dataset:
    """
    Load in-situ observations for a given first day and 10 days ahead.

    Args:
        first_day_datetime: Starting date for observations
        target_depths: Optional depths to select (will use nearest)

    Returns:
        Dataset with observations for the 10-day period
    """
    # Convert numpy.datetime64 to Python datetime
    first_day = pandas.Timestamp(first_day_datetime).to_pydatetime()

    # Dates for the request - 10 days of observations
    start_datetime = first_day.strftime("%Y-%m-%dT00:00:00")
    end_datetime = (first_day + pandas.Timedelta(days=9, hours=23, minutes=59)).strftime("%Y-%m-%dT23:59:59")

    # Load the in-situ observation dataset
    dataset = copernicusmarine.open_dataset(
        dataset_id="cmems_obs-ins_glo_phybgcwav_mynrt_na_irr",
        dataset_part="monthly",
        variables=["TEMP", "PSAL", "EWCT", "NSCT", "SLEV"],
        minimum_longitude=-180,
        maximum_longitude=179.99989318847656,
        minimum_latitude=-78.25827026367188,
        maximum_latitude=90,
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        minimum_depth=-20,
        maximum_depth=600,
    )

    # Select closest depths if specified
    if target_depths is not None and "depth" in dataset.dims:
        dataset = dataset.sel(depth=target_depths, method="nearest")

    return dataset


def obs_insitu_dataset(challenger_dataset: Dataset) -> Dataset:
    """
    Load in-situ observations matching the challenger dataset's time range.

    Args:
        challenger_dataset: Dataset with first_day_datetime dimension

    Returns:
        Combined dataset with observations for all time periods
    """
    from oceanbench.core.dataset_utils import Dimension

    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    # Extract depths from the challenger_dataset if available
    target_depths = challenger_dataset["depth"].values if "depth" in challenger_dataset.dims else None

    # Load each dataset one by one
    datasets = []
    for first_day_datetime in first_day_datetimes:
        print(f"Loading observations for {first_day_datetime}...")
        dataset = _obs_insitu_path(first_day_datetime, target_depths=target_depths)
        datasets.append(dataset)

    # Concatenate all datasets on the time dimension
    combined_dataset = concat(datasets, dim="time")

    return combined_dataset
