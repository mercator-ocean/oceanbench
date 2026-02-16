# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import numpy
import pandas
from xarray import Dataset, open_mfdataset, merge, concat
import logging
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.resolution import get_dataset_resolution
import copernicusmarine
from oceanbench.core.climate_forecast_standard_names import StandardVariable

logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)


def _glo12_1_4_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glo14/{first_day}.zarr"


def _glo12_1_degree_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glo12_1degree_2024_V2/{first_day}.zarr"


def _glo12_analysis_dataset_1_4(challenger_dataset: Dataset) -> Dataset:

    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    return open_mfdataset(
        list(map(_glo12_1_4_path, first_day_datetimes)),
        engine="zarr",
        preprocess=lambda dataset: dataset.isel(time=slice(0, lead_days_count))
        .rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()})
        .assign({Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count)}),
        combine="nested",
        concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
        parallel=True,
    ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})


def _glo12_1_12_path(first_day_datetime, days_count, target_depths=None) -> Dataset:
    # Convert numpy.datetime64 to Python datetime
    first_day = pandas.Timestamp(first_day_datetime).to_pydatetime()

    # Dates for the request
    start_datetime = first_day.strftime("%Y-%m-%dT00:00:00")
    end_datetime = (first_day + pandas.Timedelta(days=days_count - 1)).strftime("%Y-%m-%dT00:00:00")

    # Load each variable separately as the dataset_ids are different

    # 1. Temperature (thetao)
    dataset_thetao = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        variables=[StandardVariable.SEA_WATER_POTENTIAL_TEMPERATURE.value],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    # 2. Salinity (so)
    dataset_so = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
        variables=[StandardVariable.SEA_WATER_SALINITY.value],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    # 3. Currents (uo, vo)
    dataset_current = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        variables=[
            StandardVariable.EASTWARD_SEA_WATER_VELOCITY.value,
            StandardVariable.NORTHWARD_SEA_WATER_VELOCITY.value,
        ],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    # 4. Sea surface height (zos)
    dataset_zos = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
        variables=[StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.value],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    # Select closest depths if specified
    if target_depths is not None:
        dataset_thetao = dataset_thetao.sel(depth=target_depths, method="nearest")
        dataset_so = dataset_so.sel(depth=target_depths, method="nearest")
        dataset_current = dataset_current.sel(depth=target_depths, method="nearest")
        # Note: zos has no depth dimension, so we don't apply it

    # Merge all datasets - order determines variable order
    dataset = merge([dataset_thetao, dataset_so, dataset_current, dataset_zos])

    return dataset


def _glo12_analysis_dataset_1_12(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]

    # Extract depths from the challenger_dataset
    target_depths = challenger_dataset[Dimension.DEPTH.key()].values

    # Load each dataset one by one
    datasets = []
    for first_day_datetime in first_day_datetimes:
        dataset = _glo12_1_12_path(first_day_datetime, days_count=lead_days_count, target_depths=target_depths)
        # Rename 'time' to 'lead_day_index' and assign indices
        dataset = dataset.rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
            {Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count)}
        )
        datasets.append(dataset)

    # Concatenate all datasets on the first_day_datetime dimension
    combined_dataset = concat(datasets, dim=Dimension.FIRST_DAY_DATETIME.key()).assign_coords(
        {Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes}
    )

    return combined_dataset


def _glo12_analysis_dataset_1_degree(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    return open_mfdataset(
        list(map(_glo12_1_degree_path, first_day_datetimes)),
        engine="zarr",
        preprocess=lambda dataset: dataset.rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()}).assign(
            {Dimension.LEAD_DAY_INDEX.key(): range(10)}
        ),
        combine="nested",
        concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
        parallel=True,
    ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})


def glo12_analysis_dataset(challenger_dataset: Dataset) -> Dataset:

    resolution = get_dataset_resolution(challenger_dataset)

    if resolution == "quarter_degree":
        return _glo12_analysis_dataset_1_4(challenger_dataset)
    elif resolution == "twelfth_degree":
        return _glo12_analysis_dataset_1_12(challenger_dataset)
    elif resolution == "one_degree":
        return _glo12_analysis_dataset_1_degree(challenger_dataset)
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")
