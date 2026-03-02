# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
import numpy
import pandas
from xarray import Dataset, open_mfdataset, concat
import logging
import copernicusmarine
from oceanbench.core.resolution import get_dataset_resolution
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.climate_forecast_standard_names import StandardVariable
from oceanbench.core.memory_diagnostics import default_memory_tracker, describe_dataset

logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)
_memory_tracker = default_memory_tracker("reference_glorys")


def _glorys_1_4_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-glonet/public/glorys14_refull_2024/{first_day}.zarr"


def _glorys_1_degree_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glorys_1degree_2024_V2/{first_day}.zarr"


def _glorys_reanalysis_dataset_1_4(challenger_dataset: Dataset) -> Dataset:
    _memory_tracker.checkpoint("prepare_glorys_1_4")
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    with _memory_tracker.step("open_glorys_1_4_mfdataset"):
        dataset = open_mfdataset(
            list(map(_glorys_1_4_path, first_day_datetimes)),
            engine="zarr",
            preprocess=lambda dataset: dataset.isel(time=slice(0, lead_days_count))
            .rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()})
            .assign({Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count)}),
            combine="nested",
            concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
            parallel=True,
        ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})
    describe_dataset(dataset, "glorys_1_4", _memory_tracker)
    return dataset


def _glorys_1_12_path(first_day_datetime, days_count: int, target_depths: numpy.ndarray) -> Dataset:
    _memory_tracker.checkpoint(f"download_glorys_1_12_start {first_day_datetime}")
    first_day = pandas.Timestamp(first_day_datetime).to_pydatetime()
    with _memory_tracker.step(f"copernicus_open_dataset_glorys_1_12 {first_day_datetime}"):
        dataset = copernicusmarine.open_dataset(
            dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            variables=[
                StandardVariable.SEA_WATER_POTENTIAL_TEMPERATURE.value,
                StandardVariable.SEA_WATER_SALINITY.value,
                StandardVariable.EASTWARD_SEA_WATER_VELOCITY.value,
                StandardVariable.NORTHWARD_SEA_WATER_VELOCITY.value,
                StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.value,
            ],
            start_datetime=first_day.strftime("%Y-%m-%dT00:00:00"),
            end_datetime=(first_day + pandas.Timedelta(days=days_count - 1)).strftime("%Y-%m-%dT00:00:00"),
        )

    with _memory_tracker.step(f"select_depth_glorys_1_12 {first_day_datetime}"):
        dataset = dataset.sel(depth=target_depths, method="nearest")
    describe_dataset(dataset, f"glorys_1_12_single_{first_day_datetime}", _memory_tracker)
    return dataset


def _glorys_reanalysis_dataset_1_12(challenger_dataset: Dataset) -> Dataset:
    _memory_tracker.checkpoint("prepare_glorys_1_12")
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]

    target_depths = challenger_dataset[Dimension.DEPTH.key()].values

    datasets = []
    for index, first_day_datetime in enumerate(first_day_datetimes):
        _memory_tracker.checkpoint(f"glorys_1_12_iteration {index + 1}/{len(first_day_datetimes)} {first_day_datetime}")
        dataset = _glorys_1_12_path(first_day_datetime, days_count=lead_days_count, target_depths=target_depths)
        with _memory_tracker.step(f"rename_coords_glorys_1_12 {first_day_datetime}"):
            dataset = dataset.rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
                {Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count)}
            )
        datasets.append(dataset)

    with _memory_tracker.step("concat_glorys_1_12_datasets"):
        combined_dataset = concat(datasets, dim=Dimension.FIRST_DAY_DATETIME.key()).assign_coords(
            {Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes}
        )

    describe_dataset(combined_dataset, "glorys_1_12_combined", _memory_tracker)
    return combined_dataset


def _glorys_reanalysis_dataset_1_degree(challenger_dataset: Dataset) -> Dataset:
    _memory_tracker.checkpoint("prepare_glorys_1_degree")
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    with _memory_tracker.step("open_glorys_1_degree_mfdataset"):
        dataset = open_mfdataset(
            list(map(_glorys_1_degree_path, first_day_datetimes)),
            engine="zarr",
            preprocess=lambda dataset: dataset.rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()}).assign(
                {Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count)}
            ),
            combine="nested",
            concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
            parallel=True,
        ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})
    describe_dataset(dataset, "glorys_1_degree", _memory_tracker)
    return dataset


def glorys_reanalysis_dataset(challenger_dataset: Dataset) -> Dataset:
    describe_dataset(challenger_dataset, "challenger_dataset_for_glorys", _memory_tracker)

    resolution = get_dataset_resolution(challenger_dataset)
    _memory_tracker.emit(f"resolution_detected={resolution}")

    if resolution == "quarter_degree":
        return _glorys_reanalysis_dataset_1_4(challenger_dataset)
    elif resolution == "twelfth_degree":
        return _glorys_reanalysis_dataset_1_12(challenger_dataset)
    elif resolution == "one_degree":
        return _glorys_reanalysis_dataset_1_degree(challenger_dataset)
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")
