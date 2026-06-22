# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable

import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.remote_http import require_remote_dataset_dimensions, resilient_zarr_store


def prepare_reference_week_dataset(
    dataset: xarray.Dataset,
    lead_days_count: int,
    operation_name: str,
) -> xarray.Dataset:
    week_dataset = require_remote_dataset_dimensions(dataset, [Dimension.TIME.key()], operation_name)
    week_dataset = week_dataset.isel({Dimension.TIME.key(): slice(0, lead_days_count)})
    week_lead_days_count = week_dataset.sizes[Dimension.TIME.key()]
    return week_dataset.rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
        {Dimension.LEAD_DAY_INDEX.key(): range(week_lead_days_count)}
    )


def open_remote_reference_weeks(
    reference_zarr_path_from_first_day_datetime: Callable[[numpy.datetime64], str],
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
    operation_name: str,
) -> xarray.Dataset:
    return xarray.open_mfdataset(
        [
            resilient_zarr_store(reference_zarr_path_from_first_day_datetime(first_day_datetime))
            for first_day_datetime in first_day_datetimes
        ],
        engine="zarr",
        preprocess=lambda dataset: prepare_reference_week_dataset(
            dataset,
            lead_days_count=lead_days_count,
            operation_name=operation_name,
        ),
        combine="nested",
        concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
        parallel=False,
    ).assign({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})
