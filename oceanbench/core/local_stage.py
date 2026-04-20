# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import shutil
import socket
from pathlib import Path
from time import sleep
from typing import Iterator, TypeVar

import xarray

from oceanbench.core.runtime_configuration import current_runtime_configuration

DEFAULT_LOCAL_STAGE_LOCK_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_LOCAL_STAGE_LOCK_POLL_SECONDS = 5.0
ITEM = TypeVar("ITEM")
RESULT = TypeVar("RESULT")
DATASET = TypeVar("DATASET")


def should_stage_locally(stage_key: str) -> bool:
    return current_runtime_configuration().should_stage(stage_key)


def local_stage_directory() -> Path:
    return current_runtime_configuration().resolved_stage_directory()


def local_stage_max_workers() -> int:
    return current_runtime_configuration().stage_max_workers


def _local_stage_lock_timeout_seconds() -> float:
    return DEFAULT_LOCAL_STAGE_LOCK_TIMEOUT_SECONDS


def _local_stage_lock_poll_seconds() -> float:
    return DEFAULT_LOCAL_STAGE_LOCK_POLL_SECONDS


def cleanup_local_stage_directory(stage_directory: Path | None = None) -> None:
    resolved_stage_directory = stage_directory if stage_directory is not None else local_stage_directory()
    shutil.rmtree(resolved_stage_directory, ignore_errors=True)


def run_in_local_stage_workers(items: list[ITEM], callback: Callable[[ITEM], RESULT]) -> list[RESULT]:
    max_workers = min(local_stage_max_workers(), len(items))
    if max_workers <= 1:
        return [callback(item) for item in items]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(callback, items))


def write_dataset_to_local_stage(
    dataset: xarray.Dataset,
    stage_path: Path,
    *,
    prepare_dataset: Callable[[xarray.Dataset], xarray.Dataset] | None = None,
    load_before_write: bool = False,
    clear_chunk_encoding: bool = False,
) -> None:
    staged_dataset = prepare_dataset(dataset) if prepare_dataset is not None else dataset
    if load_before_write:
        staged_dataset = staged_dataset.load()
    if clear_chunk_encoding:
        for variable_name in staged_dataset.variables:
            staged_dataset[variable_name].encoding.pop("chunks", None)
    temporary_stage_path = stage_path.with_name(f"{stage_path.name}.tmp")
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(temporary_stage_path, ignore_errors=True)
    staged_dataset.to_zarr(temporary_stage_path, mode="w")
    shutil.rmtree(stage_path, ignore_errors=True)
    temporary_stage_path.rename(stage_path)


def open_or_create_local_stage_dataset(
    stage_path: Path,
    open_staged_dataset: Callable[[Path], DATASET],
    build_stage: Callable[[Path], None],
) -> DATASET:
    ensure_local_stage(stage_path, build_stage)
    return open_staged_dataset(stage_path)


def ensure_local_stage(
    stage_path: Path,
    build_stage: Callable[[Path], None],
) -> Path:
    if stage_path.exists():
        return stage_path
    with local_stage_build_guard(stage_path) as should_build_stage:
        if should_build_stage:
            build_stage(stage_path)
    return stage_path


def _local_stage_lock_path(stage_path: Path) -> Path:
    return stage_path.with_name(f"{stage_path.name}.lock")


def _write_local_stage_lock_metadata(lock_path: Path) -> None:
    metadata_path = lock_path / "owner.json"
    metadata_path.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _is_stale_local_stage_lock(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    lock_age_seconds = datetime.now(timezone.utc).timestamp() - lock_path.stat().st_mtime
    return lock_age_seconds > _local_stage_lock_timeout_seconds()


@contextmanager
def local_stage_build_guard(stage_path: Path) -> Iterator[bool]:
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _local_stage_lock_path(stage_path)
    while True:
        if stage_path.exists():
            yield False
            return
        try:
            lock_path.mkdir()
        except FileExistsError:
            if stage_path.exists():
                yield False
                return
            if _is_stale_local_stage_lock(lock_path):
                shutil.rmtree(lock_path, ignore_errors=True)
                continue
            sleep(_local_stage_lock_poll_seconds())
            continue
        try:
            _write_local_stage_lock_metadata(lock_path)
            if stage_path.exists():
                yield False
                return
            yield True
            return
        finally:
            shutil.rmtree(lock_path, ignore_errors=True)
