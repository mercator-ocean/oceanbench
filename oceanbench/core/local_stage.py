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
from os import environ
from pathlib import Path
from time import sleep
from typing import Iterator, TypeVar

from oceanbench.core.instrumentation import log_event

LOCAL_STAGE_ENVIRONMENT_VARIABLE = "OCEANBENCH_LOCAL_STAGE"
LOCAL_STAGE_DIRECTORY_ENVIRONMENT_VARIABLE = "OCEANBENCH_LOCAL_STAGE_DIRECTORY"
LOCAL_STAGE_CLEANUP_ENVIRONMENT_VARIABLE = "OCEANBENCH_LOCAL_STAGE_CLEANUP"
LOCAL_STAGE_MAX_WORKERS_ENVIRONMENT_VARIABLE = "OCEANBENCH_LOCAL_STAGE_MAX_WORKERS"
LOCAL_STAGE_LOCK_TIMEOUT_SECONDS_ENVIRONMENT_VARIABLE = "OCEANBENCH_LOCAL_STAGE_LOCK_TIMEOUT_SECONDS"
LOCAL_STAGE_LOCK_POLL_SECONDS_ENVIRONMENT_VARIABLE = "OCEANBENCH_LOCAL_STAGE_LOCK_POLL_SECONDS"
LOCAL_STAGE_ALL_KEY = "all"
LOCAL_STAGE_CLEANUP_NEVER = "never"
LOCAL_STAGE_CLEANUP_SUCCESS = "success"
LOCAL_STAGE_CLEANUP_ALWAYS = "always"
DEFAULT_LOCAL_STAGE_MAX_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_LOCAL_STAGE_LOCK_TIMEOUT_SECONDS = 24 * 60 * 60
DEFAULT_LOCAL_STAGE_LOCK_POLL_SECONDS = 5.0
ITEM = TypeVar("ITEM")
RESULT = TypeVar("RESULT")


def local_stage_keys() -> set[str]:
    raw_value = environ.get(LOCAL_STAGE_ENVIRONMENT_VARIABLE, "")
    return {key.strip().lower() for key in raw_value.split(",") if key.strip()}


def has_local_stage_configuration() -> bool:
    return bool(local_stage_keys())


def should_stage_locally(stage_key: str) -> bool:
    configured_keys = local_stage_keys()
    return stage_key.lower() in configured_keys or LOCAL_STAGE_ALL_KEY in configured_keys


def local_stage_directory() -> Path:
    default_stage_directory = Path(environ.get("TMPDIR", "/tmp")) / "oceanbench-stage"
    return Path(environ.get(LOCAL_STAGE_DIRECTORY_ENVIRONMENT_VARIABLE, str(default_stage_directory)))


def local_stage_max_workers() -> int:
    max_workers = int(environ.get(LOCAL_STAGE_MAX_WORKERS_ENVIRONMENT_VARIABLE, DEFAULT_LOCAL_STAGE_MAX_WORKERS))
    if max_workers < 1:
        raise ValueError(
            f"Unsupported {LOCAL_STAGE_MAX_WORKERS_ENVIRONMENT_VARIABLE} value: {max_workers!r}. "
            "Expected an integer greater than or equal to 1."
        )
    return max_workers


def _local_stage_lock_timeout_seconds() -> float:
    return float(
        environ.get(
            LOCAL_STAGE_LOCK_TIMEOUT_SECONDS_ENVIRONMENT_VARIABLE,
            DEFAULT_LOCAL_STAGE_LOCK_TIMEOUT_SECONDS,
        )
    )


def _local_stage_lock_poll_seconds() -> float:
    return float(
        environ.get(
            LOCAL_STAGE_LOCK_POLL_SECONDS_ENVIRONMENT_VARIABLE,
            DEFAULT_LOCAL_STAGE_LOCK_POLL_SECONDS,
        )
    )


def local_stage_cleanup_policy() -> str:
    raw_value = environ.get(LOCAL_STAGE_CLEANUP_ENVIRONMENT_VARIABLE, LOCAL_STAGE_CLEANUP_NEVER).strip().lower()
    cleanup_policy = raw_value or LOCAL_STAGE_CLEANUP_NEVER
    if cleanup_policy not in {
        LOCAL_STAGE_CLEANUP_NEVER,
        LOCAL_STAGE_CLEANUP_SUCCESS,
        LOCAL_STAGE_CLEANUP_ALWAYS,
    }:
        raise ValueError(
            f"Unsupported {LOCAL_STAGE_CLEANUP_ENVIRONMENT_VARIABLE} value: {cleanup_policy!r}. "
            f"Expected one of: {LOCAL_STAGE_CLEANUP_NEVER!r}, {LOCAL_STAGE_CLEANUP_SUCCESS!r}, "
            f"{LOCAL_STAGE_CLEANUP_ALWAYS!r}."
        )
    return cleanup_policy


def should_cleanup_local_stage_directory(operation_succeeded: bool) -> bool:
    cleanup_policy = local_stage_cleanup_policy()
    return cleanup_policy == LOCAL_STAGE_CLEANUP_ALWAYS or (
        cleanup_policy == LOCAL_STAGE_CLEANUP_SUCCESS and operation_succeeded
    )


def cleanup_local_stage_directory() -> None:
    shutil.rmtree(local_stage_directory(), ignore_errors=True)


def run_in_local_stage_workers(items: list[ITEM], callback: Callable[[ITEM], RESULT]) -> list[RESULT]:
    max_workers = min(local_stage_max_workers(), len(items))
    if max_workers <= 1:
        return [callback(item) for item in items]
    log_event(
        "local_stage_parallel_execution",
        items_count=len(items),
        max_workers=max_workers,
    )
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
    did_wait_for_lock = False
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
                log_event(
                    "local_stage_lock_stale_removed",
                    stage_path=str(stage_path),
                    lock_path=str(lock_path),
                )
                shutil.rmtree(lock_path, ignore_errors=True)
                continue
            if not did_wait_for_lock:
                did_wait_for_lock = True
                log_event(
                    "local_stage_lock_waiting",
                    stage_path=str(stage_path),
                    lock_path=str(lock_path),
                    poll_seconds=_local_stage_lock_poll_seconds(),
                )
            sleep(_local_stage_lock_poll_seconds())
            continue
        try:
            _write_local_stage_lock_metadata(lock_path)
            if stage_path.exists():
                yield False
                return
            log_event(
                "local_stage_lock_acquired",
                stage_path=str(stage_path),
                lock_path=str(lock_path),
            )
            yield True
            return
        finally:
            shutil.rmtree(lock_path, ignore_errors=True)
