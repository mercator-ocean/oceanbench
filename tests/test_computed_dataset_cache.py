# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import os
import socket

import numpy
import xarray

from oceanbench.core.computed_dataset_cache import (
    _is_stale_cache_lock,
    cached_computed_dataset,
)
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable


def _build_three_value_dataset(build_calls: list) -> xarray.Dataset:
    build_calls.append(1)
    return xarray.Dataset({"x": ("a", numpy.arange(3.0))})


def test_cached_computed_dataset_recomputes_without_cache_directory(monkeypatch) -> None:
    monkeypatch.delenv(OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE.value, raising=False)
    build_calls: list = []

    first = cached_computed_dataset("recompute-key", lambda: _build_three_value_dataset(build_calls))
    second = cached_computed_dataset("recompute-key", lambda: _build_three_value_dataset(build_calls))

    assert first["x"].values.tolist() == [0.0, 1.0, 2.0]
    assert second["x"].values.tolist() == [0.0, 1.0, 2.0]
    assert len(build_calls) == 2


def test_cached_computed_dataset_persists_and_reuses(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE.value, str(tmp_path))
    build_calls: list = []

    first = cached_computed_dataset("persisted-key", lambda: _build_three_value_dataset(build_calls))
    second = cached_computed_dataset("persisted-key", lambda: _build_three_value_dataset(build_calls))

    assert first["x"].values.tolist() == [0.0, 1.0, 2.0]
    assert second["x"].values.tolist() == [0.0, 1.0, 2.0]
    assert len(build_calls) == 1
    assert (tmp_path / "computed" / "persisted-key.zarr").exists()


def _write_lock_owner(lock_path, pid: int, hostname: str) -> None:
    lock_path.mkdir()
    (lock_path / "owner.json").write_text(
        json.dumps({"pid": pid, "hostname": hostname, "created_at": "2024-01-01T00:00:00+00:00"}),
        encoding="utf-8",
    )


def _dead_pid() -> int:
    import subprocess

    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


def test_lock_with_dead_local_owner_is_stale_immediately(tmp_path) -> None:
    lock_path = tmp_path / "fresh.zarr.lock"
    _write_lock_owner(lock_path, _dead_pid(), socket.gethostname())

    # Freshly-mtimed lock: only the dead-owner probe can make it stale.
    assert _is_stale_cache_lock(lock_path) is True


def test_lock_with_live_local_owner_is_not_stale(tmp_path) -> None:
    lock_path = tmp_path / "fresh.zarr.lock"
    _write_lock_owner(lock_path, os.getpid(), socket.gethostname())

    assert _is_stale_cache_lock(lock_path) is False


def test_lock_owned_by_other_host_is_not_stale_before_timeout(tmp_path) -> None:
    lock_path = tmp_path / "fresh.zarr.lock"
    # A dead-looking pid but a foreign hostname must not be reclaimed on the pid probe.
    _write_lock_owner(lock_path, _dead_pid(), socket.gethostname() + "-other")

    assert _is_stale_cache_lock(lock_path) is False
