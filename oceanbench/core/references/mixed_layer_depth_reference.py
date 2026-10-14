# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable

import numpy
from xarray import Dataset, concat

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.mixed_layer_depth import keep_mixed_layer_depth_levels
from oceanbench.core.weekly_stage import maybe_stage_weekly_dataset, prepare_reference_week_dataset

MIXED_LAYER_DEPTH_STAGE_VARIANT = "mixed-layer-depth"


def on_horizontal_grid(dataset: Dataset, grid_dataset: Dataset) -> Dataset:
    grid = rename_dataset_with_standard_names(grid_dataset)
    coordinates = {
        Dimension.LATITUDE.key(): grid[Dimension.LATITUDE.key()].values,
        Dimension.LONGITUDE.key(): grid[Dimension.LONGITUDE.key()].values,
    }
    return dataset.sel(coordinates, method="nearest").assign_coords(coordinates)


def mixed_layer_depth_reference_dataset(
    challenger_dataset: Dataset,
    dataset_name: str,
    resolution: str,
    open_twelfth_degree_week: Callable[[numpy.datetime64, int], Dataset],
    open_quarter_degree_grid: Callable[[numpy.datetime64], Dataset],
) -> Dataset:
    """
    Temperature and salinity of the reference on all its native levels down to the first level below the
    mixed layer depth cap, on the twelfth-degree grid or subsampled to the quarter-degree reference grid.
    The result does not depend on the challenger depth levels.
    """
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]

    def open_week_dataset(first_day_datetime: numpy.datetime64) -> Dataset:
        dataset = keep_mixed_layer_depth_levels(open_twelfth_degree_week(first_day_datetime, lead_days_count))
        if resolution == "quarter_degree":
            dataset = on_horizontal_grid(dataset, open_quarter_degree_grid(first_day_datetime))
        return prepare_reference_week_dataset(
            dataset,
            lead_days_count=lead_days_count,
            operation_name=f"{dataset_name} mixed layer depth reference open",
        )

    def open_remote_dataset() -> Dataset:
        return concat(
            [open_week_dataset(first_day_datetime) for first_day_datetime in first_day_datetimes],
            dim=Dimension.FIRST_DAY_DATETIME.key(),
        ).assign_coords({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})

    return maybe_stage_weekly_dataset(
        stage_key="references",
        dataset_kind="reference",
        dataset_name=dataset_name,
        first_day_datetimes=first_day_datetimes,
        lead_days_count=lead_days_count,
        open_week_dataset=open_week_dataset,
        open_remote_dataset=open_remote_dataset,
        resolution=resolution,
        stage_variant=MIXED_LAYER_DEPTH_STAGE_VARIANT,
    )
