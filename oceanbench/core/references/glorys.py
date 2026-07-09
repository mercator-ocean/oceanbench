# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime
from functools import partial
import numpy
import pandas
from xarray import Dataset, concat
import logging
import copernicusmarine
from oceanbench.core.resolution import get_dataset_resolution
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.climate_forecast_standard_names import StandardVariable
from oceanbench.core.computed_dataset_cache import cached_computed_dataset
from oceanbench.core.reference_depths import (
    reference_depth_grid_variant,
    with_reference_depth_grid_metadata,
)
from oceanbench.core.reference_week import open_remote_reference_weeks, prepare_reference_week_dataset
from oceanbench.core.remote_http import with_remote_http_retries

logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)

_GLORYS_REANALYSIS_DATASET_CACHE: dict[int, Dataset] = {}


def _glorys_1_4_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-glonet/public/glorys14_refull_2024/{first_day}.zarr"


def _glorys_1_degree_path(first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/glorys_1degree_2024_V2/{first_day}.zarr"


def _glorys_reanalysis_dataset_1_4(challenger_dataset: Dataset) -> Dataset:
    return open_remote_reference_weeks(
        _glorys_1_4_path,
        challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values,
        challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()],
        "GLORYS quarter-degree dataset open",
    )


def _glorys_1_12_path(first_day_datetime, days_count: int, target_depths: numpy.ndarray) -> Dataset:
    first_day = pandas.Timestamp(first_day_datetime).to_pydatetime()
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

    dataset = dataset.sel(depth=target_depths, method="nearest")

    return dataset


def _prepare_glorys_1_12_week_dataset(
    first_day_datetime: numpy.datetime64,
    lead_days_count: int,
    target_depths: numpy.ndarray,
) -> Dataset:
    return with_reference_depth_grid_metadata(
        prepare_reference_week_dataset(
            _glorys_1_12_path(first_day_datetime, days_count=lead_days_count, target_depths=target_depths),
            lead_days_count=lead_days_count,
            operation_name="GLORYS twelfth-degree dataset open",
        ),
        target_depths,
    )


def _glorys_1_12_week_cache_key(
    first_day_datetime: numpy.datetime64,
    lead_days_count: int,
    depth_grid_variant: str,
) -> str:
    first_day = pandas.Timestamp(first_day_datetime).strftime("%Y%m%d")
    return f"reference-glorys-twelfth_degree-{depth_grid_variant}-{first_day}-{lead_days_count}d"


def _glorys_reanalysis_dataset_1_12(challenger_dataset: Dataset) -> Dataset:
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    target_depths = challenger_dataset[Dimension.DEPTH.key()].values
    depth_grid_variant = reference_depth_grid_variant(target_depths)

    week_datasets = [
        cached_computed_dataset(
            _glorys_1_12_week_cache_key(first_day_datetime, lead_days_count, depth_grid_variant),
            partial(_prepare_glorys_1_12_week_dataset, first_day_datetime, lead_days_count, target_depths),
        )
        for first_day_datetime in first_day_datetimes
    ]
    return concat(week_datasets, dim=Dimension.FIRST_DAY_DATETIME.key()).assign_coords(
        {Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes}
    )


def _glorys_reanalysis_dataset_1_degree(challenger_dataset: Dataset) -> Dataset:
    return open_remote_reference_weeks(
        _glorys_1_degree_path,
        challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values,
        challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()],
        "GLORYS one-degree dataset open",
    )


def glorys_reanalysis_dataset(challenger_dataset: Dataset) -> Dataset:
    cache_key = id(challenger_dataset)
    cached_dataset = _GLORYS_REANALYSIS_DATASET_CACHE.get(cache_key)
    if cached_dataset is not None:
        return cached_dataset

    def open_dataset() -> Dataset:
        resolution = get_dataset_resolution(challenger_dataset)

        if resolution == "quarter_degree":
            return _glorys_reanalysis_dataset_1_4(challenger_dataset)
        elif resolution == "twelfth_degree":
            return _glorys_reanalysis_dataset_1_12(challenger_dataset)
        elif resolution == "one_degree":
            return _glorys_reanalysis_dataset_1_degree(challenger_dataset)
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")

    reference_dataset = with_remote_http_retries("GLORYS reference dataset open", open_dataset)
    _GLORYS_REANALYSIS_DATASET_CACHE[cache_key] = reference_dataset
    return reference_dataset
