# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import numpy
from xarray import Dataset, open_mfdataset, concat, merge
import logging
from oceanbench.core.resolution import is_quarter_degree_dataset
from oceanbench.core.dataset_utils import _select_closest_depths
import copernicusmarine
import os
import pandas as pd


logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)

from oceanbench.core.dataset_utils import Dimension


def _glo12_1_4_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glo14/{first_day}.zarr"


def _glo12_analysis_dataset_1_4(challenger_dataset: Dataset) -> Dataset:

    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    return open_mfdataset(
        list(map(_glo12_1_4_path, first_day_datetimes)),
        engine="zarr",
        preprocess=lambda dataset: dataset.rename({"time": Dimension.LEAD_DAY_INDEX.key()}).assign(
            {Dimension.LEAD_DAY_INDEX.key(): range(10)}
        ),
        combine="nested",
        concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
        parallel=True,
    ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})


def _glo12_1_12_path(first_day_datetime, target_depths=None) -> Dataset:
    # Convert numpy.datetime64 to Python datetime
    first_day = pd.Timestamp(first_day_datetime).to_pydatetime()

    # Dates for the request
    start_date = first_day.strftime("%Y-%m-%dT00:00:00")
    end_date = (first_day + pd.Timedelta(days=9)).strftime("%Y-%m-%dT00:00:00")

    # Load each variable separately as the dataset_ids are different

    # 1. Temperature (thetao)
    ds_thetao = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=["thetao"],
        start_datetime=start_date,
        end_datetime=end_date,
        username=os.environ.get("COPERNICUSMARINE_USERNAME"),
        password=os.environ.get("COPERNICUSMARINE_PASSWORD"),
    )

    # 2. Salinity (so)
    ds_so = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        variables=["so"],
        start_datetime=start_date,
        end_datetime=end_date,
        username=os.environ.get("COPERNICUSMARINE_USERNAME"),
        password=os.environ.get("COPERNICUSMARINE_PASSWORD"),
    )

    # 3. Currents (uo, vo)
    ds_currents = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=["uo", "vo"],
        start_datetime=start_date,
        end_datetime=end_date,
        username=os.environ.get("COPERNICUSMARINE_USERNAME"),
        password=os.environ.get("COPERNICUSMARINE_PASSWORD"),
    )

    # 4. Sea surface height (zos)
    ds_zos = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["zos"],
        start_datetime=start_date,
        end_datetime=end_date,
        username=os.environ.get("COPERNICUSMARINE_USERNAME"),
        password=os.environ.get("COPERNICUSMARINE_PASSWORD"),
    )

    # Select closest depths if specified
    if target_depths is not None:
        ds_thetao = _select_closest_depths(ds_thetao, target_depths)
        ds_so = _select_closest_depths(ds_so, target_depths)
        ds_currents = _select_closest_depths(ds_currents, target_depths)
        # Note: zos has no depth dimension, so we don't apply it

    # Merge all datasets - order determines variable order
    ds = merge([ds_thetao, ds_so, ds_currents, ds_zos])

    # Ensure variables are in the correct order
    ds = ds[["thetao", "so", "uo", "vo", "zos"]]

    return ds


def _glo12_analysis_dataset_1_12(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    # Extract depths from the challenger_dataset
    target_depths = challenger_dataset["depth"].values

    # Load each dataset one by one
    datasets = []
    for first_day_datetime in first_day_datetimes:
        ds = _glo12_1_12_path(first_day_datetime, target_depths=target_depths)
        # Rename 'time' to 'lead_day_index' and assign indices 0-9
        ds = ds.rename({"time": Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
            {Dimension.LEAD_DAY_INDEX.key(): range(10)}
        )
        datasets.append(ds)

    # Concatenate all datasets on the first_day_datetime dimension
    combined_ds = concat(datasets, dim=Dimension.FIRST_DAY_DATETIME.key()).assign_coords(
        {Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes}
    )

    return combined_ds


def glo12_analysis_dataset(challenger_dataset: Dataset) -> Dataset:
    return (
        _glo12_analysis_dataset_1_4(challenger_dataset)
        if is_quarter_degree_dataset(challenger_dataset)
        else _glo12_analysis_dataset_1_12(challenger_dataset)
    )
