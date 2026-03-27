# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
import shutil

import numpy
import xarray

from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.local_stage import (
    local_stage_build_guard,
    local_stage_directory,
    run_in_local_stage_workers,
    should_stage_locally,
)

LOCAL_STAGE_CHALLENGER_KEY = "challenger"


def should_stage_challenger_locally() -> bool:
    return should_stage_locally(LOCAL_STAGE_CHALLENGER_KEY)


def _challenger_stage_directory(dataset_name: str, lead_days_count: int) -> Path:
    return local_stage_directory() / f"challenger-{dataset_name}-{lead_days_count}d"


def _challenger_stage_path(
    dataset_name: str,
    lead_days_count: int,
    first_day_datetime: numpy.datetime64,
) -> Path:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return _challenger_stage_directory(dataset_name, lead_days_count) / f"{first_day}.zarr"


def _write_staged_challenger_dataset(dataset: xarray.Dataset, stage_path: Path) -> None:
    temporary_stage_path = stage_path.with_name(f"{stage_path.name}.tmp")
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(temporary_stage_path, ignore_errors=True)
    dataset.to_zarr(temporary_stage_path, mode="w")
    shutil.rmtree(stage_path, ignore_errors=True)
    temporary_stage_path.rename(stage_path)


def _open_staged_challenger_dataset(
    dataset_name: str,
    first_day_datetimes: numpy.ndarray,
    stage_paths: list[Path],
) -> xarray.Dataset:
    staged_dataset = xarray.open_mfdataset(
        [str(stage_path) for stage_path in stage_paths],
        engine="zarr",
        combine="nested",
        concat_dim=Dimension.FIRST_DAY_DATETIME.key(),
        parallel=False,
    ).assign_coords({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetimes})
    return with_dataset_source(
        staged_dataset,
        kind="challenger",
        name=dataset_name,
    )


def staged_challenger_dataset(
    dataset_name: str,
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
    open_week_dataset: Callable[[numpy.datetime64], xarray.Dataset],
) -> xarray.Dataset:
    def stage_week(first_day_datetime: numpy.datetime64) -> Path:
        stage_path = _challenger_stage_path(dataset_name, lead_days_count, first_day_datetime)
        with local_stage_build_guard(stage_path) as should_build_stage:
            if should_build_stage:
                week_dataset = open_week_dataset(first_day_datetime)
                try:
                    _write_staged_challenger_dataset(week_dataset, stage_path)
                finally:
                    week_dataset.close()
        return stage_path

    stage_paths = run_in_local_stage_workers(
        list(first_day_datetimes),
        stage_week,
    )
    return _open_staged_challenger_dataset(dataset_name, first_day_datetimes, stage_paths)
