# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy
import xarray

from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.local_stage import (
    ensure_local_stage,
    local_stage_directory,
    run_in_local_stage_workers,
    should_stage_locally,
    write_dataset_to_local_stage,
)
from oceanbench.core.remote_http import require_remote_dataset_dimensions


def _weekly_stage_directory(
    dataset_kind: str,
    dataset_name: str,
    lead_days_count: int,
    resolution: str | None = None,
) -> Path:
    resolution_suffix = "" if resolution is None else f"-{resolution}"
    return local_stage_directory() / f"{dataset_kind}-{dataset_name}{resolution_suffix}-{lead_days_count}d"


def _weekly_stage_path(
    dataset_kind: str,
    dataset_name: str,
    lead_days_count: int,
    first_day_datetime: numpy.datetime64 | datetime,
    resolution: str | None = None,
) -> Path:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return _weekly_stage_directory(dataset_kind, dataset_name, lead_days_count, resolution) / f"{first_day}.zarr"


def staged_weekly_dataset(
    *,
    dataset_kind: str,
    dataset_name: str,
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
    open_week_dataset: Callable[[numpy.datetime64], xarray.Dataset],
    resolution: str | None = None,
) -> xarray.Dataset:
    def stage_week(first_day_datetime: numpy.datetime64) -> Path:
        stage_path = _weekly_stage_path(
            dataset_kind=dataset_kind,
            dataset_name=dataset_name,
            lead_days_count=lead_days_count,
            first_day_datetime=first_day_datetime,
            resolution=resolution,
        )

        def build_stage(path: Path) -> None:
            week_dataset = open_week_dataset(first_day_datetime)
            try:
                write_dataset_to_local_stage(week_dataset, path)
            finally:
                week_dataset.close()

        return ensure_local_stage(
            stage_path,
            build_stage=build_stage,
        )

    stage_paths = run_in_local_stage_workers(list(first_day_datetimes), stage_week)
    staged_dataset = xarray.open_mfdataset(
        [str(stage_path) for stage_path in stage_paths],
        engine="zarr",
        combine="nested",
        concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
        parallel=False,
    ).assign_coords({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})
    return with_dataset_source(
        staged_dataset,
        kind=dataset_kind,
        name=dataset_name,
        resolution=resolution,
    )


def maybe_stage_weekly_dataset(
    *,
    stage_key: str,
    dataset_kind: str,
    dataset_name: str,
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
    open_week_dataset: Callable[[numpy.datetime64], xarray.Dataset],
    open_remote_dataset: Callable[[], xarray.Dataset],
    resolution: str | None = None,
    attach_source_metadata_when_not_staged: bool = True,
) -> xarray.Dataset:
    if should_stage_locally(stage_key):
        return staged_weekly_dataset(
            dataset_kind=dataset_kind,
            dataset_name=dataset_name,
            first_day_datetimes=first_day_datetimes,
            lead_days_count=lead_days_count,
            open_week_dataset=open_week_dataset,
            resolution=resolution,
        )
    remote_dataset = open_remote_dataset()
    if not attach_source_metadata_when_not_staged:
        return remote_dataset
    return with_dataset_source(
        remote_dataset,
        kind=dataset_kind,
        name=dataset_name,
        resolution=resolution,
    )


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
