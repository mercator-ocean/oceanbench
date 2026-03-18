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
from oceanbench.core.instrumentation import instrumented_operation, log_event
from oceanbench.core.local_stage import local_stage_directory, should_stage_locally

LOCAL_STAGE_REFERENCES_KEY = "references"


def should_stage_reference_locally() -> bool:
    return should_stage_locally(LOCAL_STAGE_REFERENCES_KEY)


def _reference_stage_directory(dataset_name: str, resolution: str, lead_days_count: int) -> Path:
    return local_stage_directory() / f"reference-{dataset_name}-{resolution}-{lead_days_count}d"


def _reference_stage_path(
    dataset_name: str,
    resolution: str,
    lead_days_count: int,
    first_day_datetime: numpy.datetime64,
) -> Path:
    first_day = datetime.fromisoformat(str(first_day_datetime)).strftime("%Y%m%d")
    return _reference_stage_directory(dataset_name, resolution, lead_days_count) / f"{first_day}.zarr"


def _write_staged_reference_dataset(dataset: xarray.Dataset, stage_path: Path) -> None:
    temporary_stage_path = stage_path.with_name(f"{stage_path.name}.tmp")
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(temporary_stage_path, ignore_errors=True)
    dataset.to_zarr(temporary_stage_path, mode="w")
    shutil.rmtree(stage_path, ignore_errors=True)
    temporary_stage_path.rename(stage_path)


def _open_staged_reference_dataset(
    dataset_name: str,
    resolution: str,
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
        kind="reference",
        name=dataset_name,
        resolution=resolution,
    )


def staged_reference_dataset(
    dataset_name: str,
    resolution: str,
    first_day_datetimes: numpy.ndarray,
    lead_days_count: int,
    open_week_dataset: Callable[[numpy.datetime64], xarray.Dataset],
) -> xarray.Dataset:
    stage_paths = []
    weeks_count = len(first_day_datetimes)
    for week_index, first_day_datetime in enumerate(first_day_datetimes, start=1):
        stage_path = _reference_stage_path(dataset_name, resolution, lead_days_count, first_day_datetime)
        if stage_path.exists():
            log_event(
                "reference_stage_reused",
                dataset=dataset_name,
                resolution=resolution,
                stage_path=str(stage_path),
                week_index=week_index,
                weeks_count=weeks_count,
                first_day_datetime=str(first_day_datetime),
            )
        else:
            with instrumented_operation(
                "reference_stage_build",
                dataset=dataset_name,
                resolution=resolution,
                stage_path=str(stage_path),
                week_index=week_index,
                weeks_count=weeks_count,
                first_day_datetime=str(first_day_datetime),
            ):
                week_dataset = open_week_dataset(first_day_datetime)
                try:
                    _write_staged_reference_dataset(week_dataset, stage_path)
                finally:
                    week_dataset.close()
        stage_paths.append(stage_path)
    return _open_staged_reference_dataset(dataset_name, resolution, first_day_datetimes, stage_paths)
