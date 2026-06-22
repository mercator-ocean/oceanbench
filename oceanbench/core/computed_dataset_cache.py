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

_BUILD_LOCK_TIMEOUT_SECONDS = 24 * 60 * 60
_BUILD_LOCK_POLL_SECONDS = 5.0


def cached_computed_dataset(content_key: str, build_dataset: Callable[[], xarray.Dataset]) -> xarray.Dataset:
    """Return ``build_dataset()``, persisting the computed result under the local cache
    directory keyed by ``content_key`` and reusing it on later runs. The dataset is
    recomputed every call when no local cache directory is configured."""
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


def _is_stale_cache_lock(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
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
