# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
import atexit
import json
import os
import resource
import sys
import time
from collections.abc import Iterator

INSTRUMENTATION_LOG_PATH_ENVIRONMENT_VARIABLE = "OCEANBENCH_INSTRUMENTATION_LOG_PATH"
INSTRUMENTATION_SAMPLE_SECONDS_ENVIRONMENT_VARIABLE = "OCEANBENCH_INSTRUMENTATION_SAMPLE_SECONDS"
DEFAULT_INSTRUMENTATION_SAMPLE_SECONDS = 60.0

_resource_sampler_lock = Lock()
_resource_sampler_stop_event = Event()
_resource_sampler_thread: Thread | None = None
_write_lock = Lock()


def _reset_after_fork() -> None:
    global _resource_sampler_lock
    global _resource_sampler_stop_event
    global _resource_sampler_thread
    global _write_lock
    _resource_sampler_lock = Lock()
    _resource_sampler_stop_event = Event()
    _resource_sampler_thread = None
    _write_lock = Lock()


def is_instrumentation_enabled() -> bool:
    return _log_path() is not None


def log_event(event_name: str, **payload) -> None:
    if not is_instrumentation_enabled():
        return
    _ensure_resource_sampler_started()
    _write_event(event_name, payload)


@contextmanager
def instrumented_operation(event_scope: str, **payload) -> Iterator[None]:
    start_time = time.monotonic()
    log_event(f"{event_scope}_started", **payload)
    try:
        yield
    except Exception as error:
        log_event(
            f"{event_scope}_failed",
            duration_seconds=time.monotonic() - start_time,
            error_type=error.__class__.__name__,
            error_module=error.__class__.__module__,
            error_message=str(error),
            **payload,
        )
        raise
    else:
        log_event(f"{event_scope}_completed", duration_seconds=time.monotonic() - start_time, **payload)


def _log_path() -> Path | None:
    raw_path = os.environ.get(INSTRUMENTATION_LOG_PATH_ENVIRONMENT_VARIABLE)
    if raw_path in (None, ""):
        return None
    return Path(raw_path)


def _sample_seconds() -> float:
    return float(
        os.environ.get(
            INSTRUMENTATION_SAMPLE_SECONDS_ENVIRONMENT_VARIABLE,
            DEFAULT_INSTRUMENTATION_SAMPLE_SECONDS,
        )
    )


def _write_event(event_name: str, payload: dict) -> None:
    log_path = _log_path()
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_name,
        "pid": os.getpid(),
        **payload,
    }
    with _write_lock, log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _ensure_resource_sampler_started() -> None:
    global _resource_sampler_thread
    if _resource_sampler_thread is not None and _resource_sampler_thread.is_alive():
        return
    with _resource_sampler_lock:
        if _resource_sampler_thread is not None and _resource_sampler_thread.is_alive():
            return
        _resource_sampler_stop_event.clear()
        _resource_sampler_thread = Thread(
            target=_resource_sampler_loop,
            name="oceanbench-resource-sampler",
            daemon=True,
        )
        _resource_sampler_thread.start()


def _resource_sampler_loop() -> None:
    previous_wall_time = time.monotonic()
    previous_process_time = time.process_time()
    while not _resource_sampler_stop_event.wait(_sample_seconds()):
        current_wall_time = time.monotonic()
        current_process_time = time.process_time()
        elapsed_wall_time = max(current_wall_time - previous_wall_time, 1e-9)
        elapsed_process_time = current_process_time - previous_process_time
        _write_event(
            "process_sample",
            {
                "cpu_percent": (elapsed_process_time / elapsed_wall_time) * 100.0,
                "cpu_time_seconds": current_process_time,
                "max_rss_kb": _max_rss_kb(),
                "rss_kb": _current_rss_kb(),
            },
        )
        previous_wall_time = current_wall_time
        previous_process_time = current_process_time


def _current_rss_kb() -> int | None:
    status_file = Path("/proc/self/status")
    if not status_file.exists():
        return None
    for line in status_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1])
    return None


def _max_rss_kb() -> int:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return int(max_rss / 1024)
    return int(max_rss)


def _stop_resource_sampler() -> None:
    _resource_sampler_stop_event.set()
    if _resource_sampler_thread is not None:
        _resource_sampler_thread.join(timeout=1.0)


atexit.register(_stop_resource_sampler)

if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
