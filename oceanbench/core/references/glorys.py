# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import numpy
from xarray import Dataset, open_mfdataset
import xarray
import logging
import copernicusmarine


from oceanbench.core.resolution import is_quarter_degree_dataset

logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)

from oceanbench.core.dataset_utils import Dimension


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


def _glorys_1_12_path(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    datasets = []
    for first_day_datetime in first_day_datetimes:
        first_day = datetime.fromisoformat(str(first_day_datetime))

        # Download dataset from Copernicus Marine
        ds = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            variables=["thetao", "so", "uo", "vo", "zos"],
            start_datetime=first_day.strftime("%Y-%m-%dT00:00:00"),
            end_datetime=(first_day + numpy.timedelta64(9, "D")).strftime("%Y-%m-%dT00:00:00"),
            minimum_depth=0.49402499198913574,
            maximum_depth=0.49402499198913574,
        )
        datasets.append(ds)
    # Concatenate the datasets along lead_day_index
    result = xarray.concat(datasets, dim=Dimension.FIRST_DAY_DATETIME.key())
    return result


def _glorys_reanalysis_dataset_1_12(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    return open_mfdataset(
        list(map(_glorys_1_12_path, first_day_datetimes)),
        engine="zarr",
        preprocess=lambda dataset: dataset.rename({"time": Dimension.LEAD_DAY_INDEX.key()}).assign(
            {Dimension.LEAD_DAY_INDEX.key(): range(10)}
        ),
        combine="nested",
        concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
        parallel=True,
    ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})


def glorys_reanalysis_dataset(challenger_dataset: Dataset) -> Dataset:
    return (
        _glorys_reanalysis_dataset_1_4(challenger_dataset)
        if is_quarter_degree_dataset(challenger_dataset)
        else _glorys_reanalysis_dataset_1_12(challenger_dataset)
    )
