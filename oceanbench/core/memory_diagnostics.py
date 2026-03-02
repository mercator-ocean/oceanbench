# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
import gc
import os
import time
from typing import Any


_ENV_ENABLE = "OCEANBENCH_DIAGNOSTICS"
_ENV_LOG_PATH = "OCEANBENCH_DIAGNOSTICS_LOG_PATH"
_ENV_FORCE_GC = "OCEANBENCH_DIAGNOSTICS_FORCE_GC"


def _is_enabled_from_env() -> bool:
    return os.environ.get(_ENV_ENABLE, "").strip().lower() in {"1", "true", "yes", "on"}


def _force_gc_from_env() -> bool:
    return os.environ.get(_ENV_FORCE_GC, "").strip().lower() in {"1", "true", "yes", "on"}


def _format_bytes(raw_bytes: int | None) -> str:
    if raw_bytes is None:
        return "n/a"
    unit = 1024.0
    gib = raw_bytes / unit / unit / unit
    mib = raw_bytes / unit / unit
    if gib >= 1.0:
        return f"{gib:.2f} GiB"
    return f"{mib:.1f} MiB"


def _current_rss_bytes() -> int | None:
    # Prefer /proc to avoid extra dependencies in constrained runtime environments.
    try:
        with open("/proc/self/status", "r", encoding="utf8") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        pass

    try:
        import resource

        rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(rss_kib) * 1024
    except Exception:
        return None


def _available_memory_bytes() -> int | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf8") as meminfo_file:
            for line in meminfo_file:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return int(parts[1]) * 1024
    except Exception:
        return None
    return None


def _read_int_file(path: str) -> int | None:
    try:
        with open(path, "r", encoding="utf8") as file:
            raw_value = file.read().strip()
            if raw_value == "max":
                return None
            return int(raw_value)
    except Exception:
        return None


def _cgroup_memory_current_bytes() -> int | None:
    return _read_int_file("/sys/fs/cgroup/memory.current") or _read_int_file(
        "/sys/fs/cgroup/memory/memory.usage_in_bytes"
    )


def _cgroup_memory_limit_bytes() -> int | None:
    return _read_int_file("/sys/fs/cgroup/memory.max") or _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")


@dataclass
class MemoryDiagnostics:
    component: str
    enabled: bool = field(default_factory=_is_enabled_from_env)
    log_path: str | None = field(default_factory=lambda: os.environ.get(_ENV_LOG_PATH))
    force_gc: bool = field(default_factory=_force_gc_from_env)
    _start_monotonic: float = field(default_factory=time.monotonic)
    _start_rss_bytes: int | None = field(default_factory=_current_rss_bytes)
    _last_rss_bytes: int | None = field(default_factory=_current_rss_bytes)

    def emit(self, message: str) -> None:
        if not self.enabled:
            return
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        line = f"[OCEANBENCH_DIAG][{timestamp}][{self.component}] {message}"
        print(line, flush=True)
        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf8") as log_file:
                    log_file.write(line + "\n")
            except Exception:
                # Logging to file is best effort and should not break evaluation.
                pass

    def checkpoint(self, label: str) -> None:
        if not self.enabled:
            return
        if self.force_gc:
            gc.collect()

        elapsed_seconds = time.monotonic() - self._start_monotonic
        current_rss = _current_rss_bytes()
        previous_rss = self._last_rss_bytes
        start_rss = self._start_rss_bytes
        delta_last = None if current_rss is None or previous_rss is None else current_rss - previous_rss
        delta_start = None if current_rss is None or start_rss is None else current_rss - start_rss
        available_memory = _available_memory_bytes()
        cgroup_current_memory = _cgroup_memory_current_bytes()
        cgroup_limit_memory = _cgroup_memory_limit_bytes()

        self.emit(
            f"{label} | rss={_format_bytes(current_rss)}"
            f" | delta={_format_bytes(delta_last)}"
            f" | total_delta={_format_bytes(delta_start)}"
            f" | mem_available={_format_bytes(available_memory)}"
            f" | cgroup_mem={_format_bytes(cgroup_current_memory)}"
            f" | cgroup_limit={_format_bytes(cgroup_limit_memory)}"
            f" | elapsed={elapsed_seconds:.1f}s"
        )
        self._last_rss_bytes = current_rss

    @contextmanager
    def step(self, label: str):
        self.checkpoint(f"START {label}")
        try:
            yield
        except Exception as exception:
            self.checkpoint(f"ERROR {label}: {type(exception).__name__}: {exception}")
            raise
        self.checkpoint(f"END {label}")


_TRACKERS: dict[str, MemoryDiagnostics] = {}


def default_memory_tracker(component: str) -> MemoryDiagnostics:
    tracker = _TRACKERS.get(component)
    if tracker is None:
        tracker = MemoryDiagnostics(component=component)
        _TRACKERS[component] = tracker
    return tracker


def enable_memory_diagnostics(
    log_path: str | None = None,
    force_gc: bool | None = None,
) -> None:
    os.environ[_ENV_ENABLE] = "1"
    if log_path is not None:
        os.environ[_ENV_LOG_PATH] = log_path
    if force_gc is not None:
        os.environ[_ENV_FORCE_GC] = "1" if force_gc else "0"

    for tracker in _TRACKERS.values():
        tracker.enabled = True
        if log_path is not None:
            tracker.log_path = log_path
        if force_gc is not None:
            tracker.force_gc = force_gc


def describe_dataset(dataset: Any, name: str, tracker: MemoryDiagnostics) -> None:
    if not tracker.enabled:
        return
    try:
        dims = {dim: int(size) for dim, size in dataset.sizes.items()}
        variables = list(dataset.data_vars)
        estimated_size = getattr(dataset, "nbytes", None)
        tracker.emit(
            f"DATASET {name} | dims={dims} | vars={variables}" f" | estimated_nbytes={_format_bytes(estimated_size)}"
        )
    except Exception as exception:
        tracker.emit(f"DATASET {name} summary failed: {type(exception).__name__}: {exception}")
