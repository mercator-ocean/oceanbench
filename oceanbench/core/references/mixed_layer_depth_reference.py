# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Reference mixed layer depth, precomputed once per reference and resolution.

The stores are built by helper_scripts/reference_mixed_layer_depth from the reference's full vertical levels
with build_reference_mixed_layer_depth_week, so the reference mixed layer depth does not depend on the
challenger depth levels.
"""

from datetime import datetime

import numpy
from xarray import Dataset

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.mixed_layer_depth import compute_mixed_layer_depth, keep_mixed_layer_depth_levels
from oceanbench.core.remote_http import open_remote_multizarr, open_remote_zarr
from oceanbench.core.resolution import get_dataset_resolution
from oceanbench.core.weekly_stage import maybe_stage_weekly_dataset, prepare_reference_week_dataset

REFERENCE_MIXED_LAYER_DEPTH_STAGE_VARIANT = "mixed-layer-depth"
REFERENCE_MIXED_LAYER_DEPTH_STORE_NAME = "{dataset_name}_mixed_layer_depth_{resolution}_2024"
REFERENCE_MIXED_LAYER_DEPTH_ROOT_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/references"


def reference_mixed_layer_depth_path(dataset_name: str, resolution: str, first_day_datetime: numpy.datetime64) -> str:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    store_name = REFERENCE_MIXED_LAYER_DEPTH_STORE_NAME.format(dataset_name=dataset_name, resolution=resolution)
    return f"{REFERENCE_MIXED_LAYER_DEPTH_ROOT_URL}/{store_name}/{first_day}.zarr"


def on_horizontal_grid(dataset: Dataset, grid_dataset: Dataset) -> Dataset:
    grid = rename_dataset_with_standard_names(grid_dataset)
    coordinates = {
        Dimension.LATITUDE.key(): grid[Dimension.LATITUDE.key()].values,
        Dimension.LONGITUDE.key(): grid[Dimension.LONGITUDE.key()].values,
    }
    return dataset.sel(coordinates, method="nearest").assign_coords(coordinates)


def build_reference_mixed_layer_depth_week(
    full_level_dataset: Dataset,
    horizontal_grid_dataset: Dataset | None = None,
) -> Dataset:
    """
    Mixed layer depth of a reference week from its full vertical levels, optionally sampled at the points
    of a coarser horizontal grid that coincide with the reference grid.
    """
    dataset = keep_mixed_layer_depth_levels(rename_dataset_with_standard_names(full_level_dataset))
    if horizontal_grid_dataset is not None:
        dataset = on_horizontal_grid(dataset, horizontal_grid_dataset)
    mixed_layer_depth = compute_mixed_layer_depth(dataset)
    variable_name = Variable.MIXED_LAYER_DEPTH.key()
    return mixed_layer_depth.assign({variable_name: mixed_layer_depth[variable_name].astype(numpy.float32)})


def _reference_mixed_layer_depth_dataset(challenger_dataset: Dataset, dataset_name: str) -> Dataset:
    resolution = get_dataset_resolution(challenger_dataset)
    first_day_datetimes = challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    operation_name = f"{dataset_name} reference mixed layer depth open"

    def path(first_day_datetime: numpy.datetime64) -> str:
        return reference_mixed_layer_depth_path(dataset_name, resolution, first_day_datetime)

    return maybe_stage_weekly_dataset(
        stage_key="references",
        dataset_kind="reference",
        dataset_name=dataset_name,
        first_day_datetimes=first_day_datetimes,
        lead_days_count=lead_days_count,
        open_week_dataset=lambda first_day_datetime: prepare_reference_week_dataset(
            open_remote_zarr(path(first_day_datetime)),
            lead_days_count=lead_days_count,
            operation_name=operation_name,
        ),
        open_remote_dataset=lambda: open_remote_multizarr(
            list(map(path, first_day_datetimes)),
            preprocess=lambda dataset: prepare_reference_week_dataset(
                dataset,
                lead_days_count=lead_days_count,
                operation_name=operation_name,
            ),
            combine="nested",
            concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
            parallel=False,
        ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes}),
        resolution=resolution,
        stage_variant=REFERENCE_MIXED_LAYER_DEPTH_STAGE_VARIANT,
    )


def glorys_reanalysis_mixed_layer_depth_dataset(challenger_dataset: Dataset) -> Dataset:
    return _reference_mixed_layer_depth_dataset(challenger_dataset, "glorys")


def glo12_analysis_mixed_layer_depth_dataset(challenger_dataset: Dataset) -> Dataset:
    return _reference_mixed_layer_depth_dataset(challenger_dataset, "glo12")
