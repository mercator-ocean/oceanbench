# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Local cache for computed datasets that are not plain remote zarr and so cannot be served
by the resilient chunk cache -- for example depth-regridded reanalysis references opened
through copernicusmarine, or the observation subset selected for a challenger.

The cache lives under the ``OCEANBENCH_LOCAL_CACHE`` directory and is keyed by a caller
supplied content key. Without that directory configured the dataset is simply recomputed,
so the pure-online mode never touches local storage."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import shutil
import socket
from pathlib import Path
from time import sleep

import xarray

from oceanbench.core.runtime_configuration import current_runtime_configuration

_BUILD_LOCK_TIMEOUT_SECONDS = 60 * 60
_BUILD_LOCK_POLL_SECONDS = 5.0


def cached_computed_dataset(content_key: str, build_dataset: Callable[[], xarray.Dataset]) -> xarray.Dataset:
    """Return ``build_dataset()``, persisting the computed result under the local cache
    directory keyed by ``content_key`` and reusing it on later runs. The dataset is
    recomputed every call when no local cache directory is configured.

    No invalidation contract: once a ``content_key`` is cached the stored dataset is
    returned verbatim forever. Nothing is revalidated against the source, so if the
    upstream data is republished under the same identity the stale copy keeps being
    served. Point at a fresh cache directory (or delete the entry) when data is
    republished at the same URL."""
    cache_directory = current_runtime_configuration().local_cache_directory()
    if cache_directory is None:
        return build_dataset()
    cache_path = cache_directory / "computed" / f"{content_key}.zarr"
    if not cache_path.exists():
        with _cache_build_guard(cache_path) as should_build:
            if should_build:
                _write_computed_dataset(build_dataset(), cache_path)
    return xarray.open_dataset(cache_path, engine="zarr")


def _write_computed_dataset(dataset: xarray.Dataset, cache_path: Path) -> None:
    loaded_dataset = dataset.load()
    for variable_name in loaded_dataset.variables:
        loaded_dataset[variable_name].encoding.pop("chunks", None)
    temporary_path = cache_path.with_name(f"{cache_path.name}.tmp")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(temporary_path, ignore_errors=True)
    loaded_dataset.to_zarr(temporary_path, mode="w")
    shutil.rmtree(cache_path, ignore_errors=True)
    temporary_path.rename(cache_path)
    dataset.close()


def _cache_lock_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.lock")


def _write_cache_lock_metadata(lock_path: Path) -> None:
    (lock_path / "owner.json").write_text(
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


def _process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # The process exists but is owned by another user.
        return True
    return True


def _lock_owner_is_dead_local_process(lock_path: Path) -> bool:
    """Return ``True`` when the lock records a pid on this host that is no longer running.

    A dead owner on the local host means the build that took the lock has crashed or been
    killed, so the lock can be reclaimed immediately without waiting for the timeout. The
    hostname guard keeps us from misreading a pid that belongs to a different machine."""
    try:
        owner = json.loads((lock_path / "owner.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if owner.get("hostname") != socket.gethostname():
        return False
    pid = owner.get("pid")
    if not isinstance(pid, int):
        return False
    return not _process_is_alive(pid)


def _is_stale_cache_lock(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    if _lock_owner_is_dead_local_process(lock_path):
        return True
    lock_age_seconds = datetime.now(timezone.utc).timestamp() - lock_path.stat().st_mtime
    return lock_age_seconds > _BUILD_LOCK_TIMEOUT_SECONDS


@contextmanager
def _cache_build_guard(cache_path: Path) -> Iterator[bool]:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _cache_lock_path(cache_path)
    while True:
        if cache_path.exists():
            yield False
            return
        try:
            lock_path.mkdir()
        except FileExistsError:
            if cache_path.exists():
                yield False
                return
            if _is_stale_cache_lock(lock_path):
                shutil.rmtree(lock_path, ignore_errors=True)
                continue
            sleep(_BUILD_LOCK_POLL_SECONDS)
            continue
        try:
            _write_cache_lock_metadata(lock_path)
            if cache_path.exists():
                yield False
                return
            yield True
            return
        finally:
            shutil.rmtree(lock_path, ignore_errors=True)
