# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import numpy
from xarray import Dataset, open_mfdataset
import logging
from oceanbench.core.resolution import is_quarter_degree_dataset
import copernicusmarine
import os

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


from oceanbench.core.dataset_utils import DepthLevel
import pandas as pd
import numpy as np
from xarray import concat, merge


def _glo12_1_12_path(first_day_datetime) -> Dataset:
    # Convertir numpy.datetime64 en datetime Python
    first_day = pd.Timestamp(first_day_datetime).to_pydatetime()

    # Récupérer les profondeurs attendues
    target_depths = sorted([depth_level.value for depth_level in DepthLevel])

    # Dates pour la requête
    start_date = first_day.strftime("%Y-%m-%dT00:00:00")
    end_date = (first_day + pd.Timedelta(days=9)).strftime("%Y-%m-%dT00:00:00")

    # Charger chaque variable séparément car les dataset_id sont différents

    # 1. Température (thetao)
    ds_thetao = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=["thetao"],
        start_datetime=start_date,
        end_datetime=end_date,
        minimum_depth=min(target_depths),
        maximum_depth=max(target_depths),
        username=os.environ.get("COPERNICUSMARINE_USERNAME"),
        password=os.environ.get("COPERNICUSMARINE_PASSWORD"),
    )

    # 2. Salinité (so)
    ds_so = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        variables=["so"],
        start_datetime=start_date,
        end_datetime=end_date,
        minimum_depth=min(target_depths),
        maximum_depth=max(target_depths),
        username=os.environ.get("COPERNICUSMARINE_USERNAME"),
        password=os.environ.get("COPERNICUSMARINE_PASSWORD"),
    )

    # 3. Courants (uo, vo)
    ds_currents = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=["uo", "vo"],
        start_datetime=start_date,
        end_datetime=end_date,
        minimum_depth=min(target_depths),
        maximum_depth=max(target_depths),
        username=os.environ.get("COPERNICUSMARINE_USERNAME"),
        password=os.environ.get("COPERNICUSMARINE_PASSWORD"),
    )

    # 4. Hauteur de la surface de la mer (zos)
    ds_zos = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=["zos"],
        start_datetime=start_date,
        end_datetime=end_date,
        username=os.environ.get("COPERNICUSMARINE_USERNAME"),
        password=os.environ.get("COPERNICUSMARINE_PASSWORD"),
    )

    # Sélectionner les profondeurs les plus proches pour les variables 3D
    ds_thetao_selected = ds_thetao.sel(depth=target_depths, method="nearest")
    ds_so_selected = ds_so.sel(depth=target_depths, method="nearest")
    ds_currents_selected = ds_currents.sel(depth=target_depths, method="nearest")

    # FORCER les coordonnées depth aux valeurs exactes attendues
    ds_thetao_selected = ds_thetao_selected.drop_vars("depth").assign_coords(
        depth=("depth", np.array(target_depths, dtype=np.float64))
    )
    ds_so_selected = ds_so_selected.drop_vars("depth").assign_coords(
        depth=("depth", np.array(target_depths, dtype=np.float64))
    )
    ds_currents_selected = ds_currents_selected.drop_vars("depth").assign_coords(
        depth=("depth", np.array(target_depths, dtype=np.float64))
    )

    # Fusionner tous les datasets - l'ordre détermine l'ordre des variables
    ds = merge([ds_thetao_selected, ds_so_selected, ds_currents_selected, ds_zos])

    # S'assurer que les variables sont dans le bon ordre
    ds = ds[["thetao", "so", "uo", "vo", "zos"]]

    return ds


def _glo12_analysis_dataset_1_12(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values

    # Charger chaque dataset un par un
    datasets = []
    for first_day_datetime in first_day_datetimes:
        ds = _glo12_1_12_path(first_day_datetime)
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


def glo12_analysis_dataset(challenger_dataset: Dataset) -> Dataset:
    return (
        _glo12_analysis_dataset_1_4(challenger_dataset)
        if is_quarter_degree_dataset(challenger_dataset)
        else _glo12_analysis_dataset_1_12(challenger_dataset)
    )
