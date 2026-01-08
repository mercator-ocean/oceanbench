# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import numpy
import pandas
from xarray import Dataset, open_mfdataset, concat
import logging
import copernicusmarine
from oceanbench.core.resolution import is_quarter_degree_dataset
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.climate_forecast_standard_names import StandardVariable

logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)

COPERNICUSMARINE_USERNAME = "zaissa@mercator-ocean.fr"
COPERNICUSMARINE_PASSWORD = "Banana-!31100"


def _glorys_1_4_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-glonet/public/glorys14_refull_2024/{first_day}.zarr"


def _glorys_reanalysis_dataset_1_4(challenger_dataset: Dataset) -> Dataset:

    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    return open_mfdataset(
        list(map(_glorys_1_4_path, first_day_datetimes)),
        engine="zarr",
        preprocess=lambda dataset: dataset.rename({"time": Dimension.LEAD_DAY_INDEX.key()}).assign(
            {Dimension.LEAD_DAY_INDEX.key(): range(10)}
        ),
        combine="nested",
        concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
        parallel=True,
    ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})


def _glorys_1_12_path(first_day_datetime, target_depths=None) -> Dataset:
    """
    Args:
       first_day_datetime: Start date
       target_depths: Optional list of target depths to select"""
    first_day = pandas.Timestamp(first_day_datetime).to_pydatetime()

    dataset = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        username=COPERNICUSMARINE_USERNAME,
        password=COPERNICUSMARINE_PASSWORD,
        variables=[
            StandardVariable.SEA_WATER_POTENTIAL_TEMPERATURE.value,
            StandardVariable.SEA_WATER_SALINITY.value,
            StandardVariable.EASTWARD_SEA_WATER_VELOCITY.value,
            StandardVariable.NORTHWARD_SEA_WATER_VELOCITY.value,
            StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.value,
        ],
        start_datetime=first_day.strftime("%Y-%m-%dT00:00:00"),
        end_datetime=(first_day + pandas.Timedelta(days=9)).strftime("%Y-%m-%dT00:00:00"),
    )

    # Select closest depths if specified
    if target_depths is not None:
        dataset = dataset.sel(depth=target_depths, method="nearest")

    return dataset


def _glorys_reanalysis_dataset_1_12(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    # Extract depths from challenger_dataset
    target_depths = challenger_dataset["depth"].values

    # Load each dataset one by one
    datasets = []
    for first_day_datetime in first_day_datetimes:
        dataset = _glorys_1_12_path(first_day_datetime, target_depths=target_depths)
        # Rename 'time' to 'lead_day_index' and assign indices 0-9
        dataset = dataset.rename({"time": Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
            {Dimension.LEAD_DAY_INDEX.key(): range(10)}
        )
        datasets.append(dataset)

    # Concatenate all datasets along the first_day_datetime dimension
    combined_dataset = concat(datasets, dim=Dimension.FIRST_DAY_DATETIME.key()).assign_coords(
        {Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes}
    )

    return combined_dataset


def glorys_reanalysis_dataset(challenger_dataset: Dataset) -> Dataset:
    return (
        _glorys_reanalysis_dataset_1_4(challenger_dataset)
        if is_quarter_degree_dataset(challenger_dataset)
        else _glorys_reanalysis_dataset_1_12(challenger_dataset)
    )
