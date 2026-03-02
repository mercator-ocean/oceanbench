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
from oceanbench.core.memory_diagnostics import default_memory_tracker, describe_dataset

logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)
_memory_tracker = default_memory_tracker("reference_glo12")


def _glo12_1_4_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glo14/{first_day}.zarr"


def _glo12_1_degree_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glo12_1degree_2024_V2/{first_day}.zarr"


def _glo12_analysis_dataset_1_4(challenger_dataset: Dataset) -> Dataset:
    _memory_tracker.checkpoint("prepare_glo12_1_4")
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    with _memory_tracker.step("open_glo12_1_4_mfdataset"):
        dataset = open_mfdataset(
            list(map(_glo12_1_4_path, first_day_datetimes)),
            engine="zarr",
            preprocess=lambda dataset: dataset.isel(time=slice(0, lead_days_count))
            .rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()})
            .assign({Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count)}),
            combine="nested",
            concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
            parallel=True,
        ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})
    describe_dataset(dataset, "glo12_1_4", _memory_tracker)
    return dataset


def _glo12_1_12_path(first_day_datetime, days_count: int, target_depths: numpy.ndarray) -> Dataset:
    _memory_tracker.checkpoint(f"download_glo12_1_12_start {first_day_datetime}")
    first_day = pandas.Timestamp(first_day_datetime).to_pydatetime()

    start_datetime = first_day.strftime("%Y-%m-%dT00:00:00")
    end_datetime = (first_day + pandas.Timedelta(days=days_count - 1)).strftime("%Y-%m-%dT00:00:00")

    # Load each variable separately as the dataset_ids are different

    with _memory_tracker.step(f"open_temperature_glo12_1_12 {first_day_datetime}"):
        dataset_temperature = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
            variables=[StandardVariable.SEA_WATER_POTENTIAL_TEMPERATURE.value],
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )

    with _memory_tracker.step(f"open_salinity_glo12_1_12 {first_day_datetime}"):
        dataset_salinity = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m",
            variables=[StandardVariable.SEA_WATER_SALINITY.value],
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )

    with _memory_tracker.step(f"open_currents_glo12_1_12 {first_day_datetime}"):
        dataset_current = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
            variables=[
                StandardVariable.EASTWARD_SEA_WATER_VELOCITY.value,
                StandardVariable.NORTHWARD_SEA_WATER_VELOCITY.value,
            ],
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )

    with _memory_tracker.step(f"open_ssh_glo12_1_12 {first_day_datetime}"):
        dataset_sea_surface_height = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy_anfc_0.083deg_P1D-m",
            variables=[StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.value],
            start_datetime=start_datetime,
            end_datetime=end_datetime,
        )

    with _memory_tracker.step(f"select_depth_glo12_1_12 {first_day_datetime}"):
        dataset_temperature = dataset_temperature.sel(depth=target_depths, method="nearest")
        dataset_salinity = dataset_salinity.sel(depth=target_depths, method="nearest")
        dataset_current = dataset_current.sel(depth=target_depths, method="nearest")

    with _memory_tracker.step(f"merge_glo12_1_12 {first_day_datetime}"):
        dataset = merge([dataset_temperature, dataset_salinity, dataset_current, dataset_sea_surface_height])

    describe_dataset(dataset, f"glo12_1_12_single_{first_day_datetime}", _memory_tracker)
    return dataset


def _glo12_analysis_dataset_1_12(challenger_dataset: Dataset) -> Dataset:
    _memory_tracker.checkpoint("prepare_glo12_1_12")
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]

    target_depths = challenger_dataset[Dimension.DEPTH.key()].values

    datasets = []
    for index, first_day_datetime in enumerate(first_day_datetimes):
        _memory_tracker.checkpoint(f"glo12_1_12_iteration {index + 1}/{len(first_day_datetimes)} {first_day_datetime}")
        dataset = _glo12_1_12_path(first_day_datetime, days_count=lead_days_count, target_depths=target_depths)
        with _memory_tracker.step(f"rename_coords_glo12_1_12 {first_day_datetime}"):
            dataset = dataset.rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
                {Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count)}
            )
        datasets.append(dataset)

    with _memory_tracker.step("concat_glo12_1_12_datasets"):
        combined_dataset = concat(datasets, dim=Dimension.FIRST_DAY_DATETIME.key()).assign_coords(
            {Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes}
        )

    describe_dataset(combined_dataset, "glo12_1_12_combined", _memory_tracker)
    return combined_dataset


def _glo12_analysis_dataset_1_degree(challenger_dataset: Dataset) -> Dataset:
    _memory_tracker.checkpoint("prepare_glo12_1_degree")
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    with _memory_tracker.step("open_glo12_1_degree_mfdataset"):
        dataset = open_mfdataset(
            list(map(_glo12_1_degree_path, first_day_datetimes)),
            engine="zarr",
            preprocess=lambda dataset: dataset.rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()}).assign(
                {Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count)}
            ),
            combine="nested",
            concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
            parallel=True,
        ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})
    describe_dataset(dataset, "glo12_1_degree", _memory_tracker)
    return dataset


def glo12_analysis_dataset(challenger_dataset: Dataset) -> Dataset:
    describe_dataset(challenger_dataset, "challenger_dataset_for_glo12", _memory_tracker)

    resolution = get_dataset_resolution(challenger_dataset)
    _memory_tracker.emit(f"resolution_detected={resolution}")

    if resolution == "quarter_degree":
        return _glo12_analysis_dataset_1_4(challenger_dataset)
    elif resolution == "twelfth_degree":
        return _glo12_analysis_dataset_1_12(challenger_dataset)
    elif resolution == "one_degree":
        return _glo12_analysis_dataset_1_degree(challenger_dataset)
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")
