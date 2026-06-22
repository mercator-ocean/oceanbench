# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from time import sleep
from typing import TypeVar
import hashlib
import logging
import os
import threading

import zarr

from oceanbench.core.runtime_configuration import current_runtime_configuration

DEFAULT_RETRY_BACKOFF_SECONDS = 2
RETRIABLE_HTTP_ERROR_TOKENS = (
    "Server disconnected",
    "Connection reset by peer",
    "Not enough data to satisfy content length header",
    "Response payload is not completed",
)
RETRIABLE_REMOTE_BACKEND_MODULE_PREFIXES = (
    "aiohttp",
    "botocore",
)
RETRIABLE_HTTP_STATUS_CODES = (408, 429, 500, 502, 503, 504)
NON_RETRIABLE_HTTP_STATUS_CODES = (400, 401, 403, 404, 405, 409, 410, 416, 422)

REMOTE_ZARR_LOGGER = logging.getLogger(__name__)
# Keep transparent retries out of generated report notebooks: a Jupyter kernel captures
# every stderr write into the executing cell, so a retry warning would otherwise be baked
# into the published report. A successful retry needs no report output; a failed one still
# raises and surfaces a real traceback.
REMOTE_ZARR_LOGGER.addHandler(logging.NullHandler())
REMOTE_ZARR_LOGGER.propagate = False

CALLBACK_RESULT = TypeVar("CALLBACK_RESULT")
DATASET = TypeVar("DATASET")


class RetriableRemoteDataError(RuntimeError):
    pass


def _exception_chain(error: Exception):
    seen_exceptions = set()
    current_exception: Exception | None = error
    while current_exception is not None and id(current_exception) not in seen_exceptions:
        yield current_exception
        seen_exceptions.add(id(current_exception))
        current_exception = current_exception.__cause__ or current_exception.__context__


def _originates_from_retriable_remote_backend(exception: Exception) -> bool:
    return exception.__class__.__module__.startswith(RETRIABLE_REMOTE_BACKEND_MODULE_PREFIXES)


def _remote_http_status_code(exception: Exception) -> int | None:
    for attribute_name in ("status", "status_code"):
        status = getattr(exception, attribute_name, None)
        if isinstance(status, int):
            return status
    response = getattr(exception, "response", None)
    if isinstance(response, Mapping):
        http_status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
        if isinstance(http_status, int):
            return http_status
    return None


def _is_retriable_remote_data_exception(exception: Exception) -> bool:
    if isinstance(exception, RetriableRemoteDataError):
        return True
    status_code = _remote_http_status_code(exception)
    if status_code in RETRIABLE_HTTP_STATUS_CODES:
        return True
    if status_code in NON_RETRIABLE_HTTP_STATUS_CODES:
        return False
    if _originates_from_retriable_remote_backend(exception):
        return True
    return any(token in str(exception) for token in RETRIABLE_HTTP_ERROR_TOKENS)


def _is_retriable_remote_data_error(error: Exception) -> bool:
    return any(_is_retriable_remote_data_exception(exception) for exception in _exception_chain(error))


def require_remote_dataset_dimensions(
    dataset: DATASET,
    expected_dimensions: Iterable[str],
    operation_name: str,
) -> DATASET:
    missing_dimensions = sorted(set(expected_dimensions) - set(dataset.dims))
    if missing_dimensions:
        raise RetriableRemoteDataError(
            f"Remote dataset opened without expected dimensions {missing_dimensions} during {operation_name}. "
            f"Available dimensions: {sorted(dataset.dims)}"
        )
    return dataset


def with_remote_http_retries(
    operation_name: str,
    callback: Callable[[], CALLBACK_RESULT],
) -> CALLBACK_RESULT:
    retry_count = current_runtime_configuration().remote_retries
    for attempt in range(1, retry_count + 1):
        try:
            return callback()
        except Exception as error:
            is_retriable_error = _is_retriable_remote_data_error(error)
            if not is_retriable_error or attempt == retry_count:
                raise
            backoff_seconds = DEFAULT_RETRY_BACKOFF_SECONDS * attempt
            REMOTE_ZARR_LOGGER.warning(
                "Remote data read failed during %s (%s/%s): %s. Retrying in %ss.",
                operation_name,
                attempt,
                retry_count,
                error,
                backoff_seconds,
            )
            sleep(backoff_seconds)

    raise RuntimeError(f"Remote data retries exhausted for {operation_name}")


class _RetryingRemoteMapper:
    def __init__(self, inner_mapper, cache_directory: Path | None = None):
        self._inner_mapper = inner_mapper
        self._cache_directory = cache_directory

    def __getattr__(self, name):
        if name in ("_inner_mapper", "_cache_directory") or (name.startswith("__") and name.endswith("__")):
            raise AttributeError(name)
        return getattr(self._inner_mapper, name)

    def _cache_path(self, key) -> Path:
        identity = hashlib.sha1(self._inner_mapper._key_to_str(key).encode()).hexdigest()
        return self._cache_directory / identity[:2] / identity

    def _cached_value(self, key) -> bytes | None:
        if self._cache_directory is None:
            return None
        cache_path = self._cache_path(key)
        return cache_path.read_bytes() if cache_path.exists() else None

    def _store_cached_value(self, key, value) -> None:
        if self._cache_directory is None or not isinstance(value, (bytes, bytearray)):
            return
        cache_path = self._cache_path(key)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = cache_path.with_name(f"{cache_path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        temporary_path.write_bytes(value)
        os.replace(temporary_path, cache_path)

    def __getitem__(self, key):
        cached_value = self._cached_value(key)
        if cached_value is not None:
            return cached_value
        value = with_remote_http_retries(f"remote zarr key read '{key}'", lambda: self._inner_mapper[key])
        self._store_cached_value(key, value)
        return value

    def getitems(self, keys, on_error="return", **keyword_arguments):
        results = {}
        keys_to_fetch = []
        for key in keys:
            cached_value = self._cached_value(key)
            if cached_value is not None:
                results[key] = cached_value
            else:
                keys_to_fetch.append(key)
        for key, value in self._fetch_with_retries(keys_to_fetch, **keyword_arguments).items():
            self._store_cached_value(key, value)
            results[key] = value
        return results

    def _fetch_with_retries(self, keys, **keyword_arguments) -> dict:
        if not keys:
            return {}
        fetched_values = dict(self._inner_mapper.getitems(list(keys), on_error="return", **keyword_arguments))
        retry_count = current_runtime_configuration().remote_retries
        for attempt in range(1, retry_count):
            retriable_keys = [
                key
                for key, value in fetched_values.items()
                if isinstance(value, Exception) and _is_retriable_remote_data_error(value)
            ]
            if not retriable_keys:
                break
            backoff_seconds = DEFAULT_RETRY_BACKOFF_SECONDS * attempt
            REMOTE_ZARR_LOGGER.warning(
                "Remote zarr chunk read failed for %s key(s) (%s/%s). Retrying in %ss.",
                len(retriable_keys),
                attempt,
                retry_count,
                backoff_seconds,
            )
            sleep(backoff_seconds)
            fetched_values.update(self._inner_mapper.getitems(retriable_keys, on_error="return", **keyword_arguments))
        return fetched_values

    def __setitem__(self, key, value):
        self._inner_mapper[key] = value

    def __delitem__(self, key):
        del self._inner_mapper[key]

    def __contains__(self, key):
        return key in self._inner_mapper

    def __iter__(self):
        return iter(self._inner_mapper)

    def __len__(self):
        return len(self._inner_mapper)


def resilient_zarr_store(url: str, **storage_options) -> zarr.storage.FSStore:
    """Open a remote zarr store whose every metadata and chunk read retries on transient
    network failures, so lazy xarray/dask computation survives a flaky connection without
    staging. Returns a drop-in replacement for the url argument of
    ``xarray.open_zarr``/``open_dataset``/``open_mfdataset``.

    When the ``OCEANBENCH_LOCAL_CACHE`` directory is configured, each fetched chunk is
    additionally persisted there -- only after a complete, retried read, written atomically
    -- and reused across runs, so the same opener serves both the pure-online and the
    local-cache modes without the caller branching. The cache holds one file per chunk with
    no shared index, so concurrent dask reads are safe; it is keyed by source URL, so point
    it at a fresh directory when a dataset is republished at the same URL."""
    cache_directory = current_runtime_configuration().local_cache_directory()
    store = zarr.storage.FSStore(url, mode="r", **storage_options)
    store.map = _RetryingRemoteMapper(store.map, cache_directory)
    return store
