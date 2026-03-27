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

from oceanbench.core.runtime_configuration import current_runtime_configuration

DEFAULT_LOCAL_STAGE_LOCK_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_LOCAL_STAGE_LOCK_POLL_SECONDS = 5.0
ITEM = TypeVar("ITEM")
RESULT = TypeVar("RESULT")


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
