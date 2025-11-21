# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import numpy
from xarray import Dataset, open_mfdataset
import xarray
import logging
import copernicusmarine
import os


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


os.environ["COPERNICUSMARINE_USERNAME"] = "zaissa@mercator-ocean.fr"
os.environ["COPERNICUSMARINE_PASSWORD"] = "Banana-!31100"

from oceanbench.core.dataset_utils import DepthLevel
import pandas as pd
import numpy as np


def _glorys_1_12_path(first_day_datetime) -> Dataset:
    # Convertir numpy.datetime64 en datetime Python
    first_day = pd.Timestamp(first_day_datetime).to_pydatetime()

    # Récupérer les profondeurs attendues
    target_depths = sorted([depth_level.value for depth_level in DepthLevel])

    ds = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        variables=["thetao", "so", "uo", "vo", "zos"],
        start_datetime=first_day.strftime("%Y-%m-%dT00:00:00"),
        end_datetime=(first_day + pd.Timedelta(days=9)).strftime("%Y-%m-%dT00:00:00"),
        minimum_depth=min(target_depths),
        maximum_depth=max(target_depths),
        username=os.environ.get("COPERNICUSMARINE_USERNAME"),
        password=os.environ.get("COPERNICUSMARINE_PASSWORD"),
    )

    # Sélectionner les profondeurs les plus proches
    ds_selected = ds.sel(depth=target_depths, method="nearest")

    # FORCER les coordonnées depth aux valeurs exactes attendues
    # en recréant explicitement la coordonnée
    ds_selected = ds_selected.drop_vars("depth").assign_coords(
        depth=("depth", np.array(target_depths, dtype=np.float64))
    )

    return ds_selected


from xarray import concat


def _glorys_reanalysis_dataset_1_12(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    # Charger chaque dataset un par un
    datasets = []
    for first_day_datetime in first_day_datetimes:
        ds = _glorys_1_12_path(first_day_datetime)
        # Renommer 'time' en 'lead_day_index' et assigner des indices 0-9
        ds = ds.rename({"time": Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
            {Dimension.LEAD_DAY_INDEX.key(): range(10)}
        )
        datasets.append(ds)

    # Concaténer tous les datasets sur la dimension first_day_datetime
    combined_ds = concat(datasets, dim=Dimension.FIRST_DAY_DATETIME.key()).assign_coords(
        {Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes}
    )

    return combined_ds


def glorys_reanalysis_dataset(challenger_dataset: Dataset) -> Dataset:
    return (
        _glorys_reanalysis_dataset_1_4(challenger_dataset)
        if is_quarter_degree_dataset(challenger_dataset)
        else _glorys_reanalysis_dataset_1_12(challenger_dataset)
    )
