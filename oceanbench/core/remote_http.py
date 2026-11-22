# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Awaitable, Callable, Iterable, Sequence
from functools import partial
from http import HTTPStatus
from typing import Any, TypeVar
import asyncio
import logging
import os

import aiohttp
import xarray
import zarr
from fsspec.implementations.http import HTTPFileSystem

from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.runtime_configuration import current_runtime_configuration

FIRST_RETRY_BACKOFF_SECONDS = 4
MAXIMUM_RETRY_BACKOFF_SECONDS = 32
DEFAULT_HTTP_SOCK_CONNECT_TIMEOUT_SECONDS = 30.0
DEFAULT_HTTP_SOCK_READ_TIMEOUT_SECONDS = 120.0
RETRIABLE_REQUEST_ERRORS = (
    aiohttp.ClientConnectionError,
    aiohttp.ClientPayloadError,
    asyncio.TimeoutError,
    TimeoutError,
    ConnectionError,
)

# FSStore reads these as an absent chunk (fill value); fsspec raises KeyError for absent keys,
# so download failures (OSError subclasses) propagate instead of being staged as fill values.
REMOTE_ZARR_STORE_MISSING_KEY_EXCEPTIONS: tuple[type[BaseException], ...] = (KeyError,)

REMOTE_ZARR_LOGGER = logging.getLogger(__name__)

CALLBACK_RESULT = TypeVar("CALLBACK_RESULT")
DATASET = TypeVar("DATASET")


class IncompleteRemoteDatasetError(RuntimeError):
    pass


def _timeout_seconds_from_environment(environment_variable: OceanbenchEnvironmentVariable, default: float) -> float:
    seconds = float(os.environ.get(environment_variable.value, default))
    if seconds <= 0:
        raise ValueError(f"{environment_variable.value} must be greater than 0.")
    return seconds


def remote_http_client_timeout() -> aiohttp.ClientTimeout:
    # No total limit, so large chunks may stream slowly, but a socket that stays silent
    # (request never answered, peer gone) raises and goes through the retries below.
    return aiohttp.ClientTimeout(
        total=None,
        sock_connect=_timeout_seconds_from_environment(
            OceanbenchEnvironmentVariable.OCEANBENCH_HTTP_SOCK_CONNECT_TIMEOUT,
            DEFAULT_HTTP_SOCK_CONNECT_TIMEOUT_SECONDS,
        ),
        sock_read=_timeout_seconds_from_environment(
            OceanbenchEnvironmentVariable.OCEANBENCH_HTTP_SOCK_READ_TIMEOUT,
            DEFAULT_HTTP_SOCK_READ_TIMEOUT_SECONDS,
        ),
    )


def _client_kwargs_with_timeout(client_kwargs: dict[str, Any] | None) -> dict[str, Any]:
    return {"timeout": remote_http_client_timeout(), **(client_kwargs or {})}


def _is_retriable_request_error(error: BaseException) -> bool:
    if isinstance(error, aiohttp.ClientResponseError):
        return error.status == HTTPStatus.TOO_MANY_REQUESTS or error.status >= HTTPStatus.INTERNAL_SERVER_ERROR
    return isinstance(error, RETRIABLE_REQUEST_ERRORS)


def _retry_backoff_seconds(attempt: int) -> int:
    return min(MAXIMUM_RETRY_BACKOFF_SECONDS, FIRST_RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))


async def _request_with_retries(url: str, request: Callable[[], Awaitable[CALLBACK_RESULT]]) -> CALLBACK_RESULT:
    attempts_count = current_runtime_configuration().remote_retries
    for attempt in range(1, attempts_count):
        try:
            return await request()
        except Exception as error:
            if not _is_retriable_request_error(error):
                raise
            backoff_seconds = _retry_backoff_seconds(attempt)
            REMOTE_ZARR_LOGGER.info(
                "Remote request for %s failed (%s/%s): %r. Retrying in %ss.",
                url,
                attempt,
                attempts_count,
                error,
                backoff_seconds,
            )
        await asyncio.sleep(backoff_seconds)
    return await request()


class RetryingHTTPFileSystem(HTTPFileSystem):
    def __init__(self, *args, client_kwargs: dict[str, Any] | None = None, **kwargs):
        super().__init__(*args, client_kwargs=_client_kwargs_with_timeout(client_kwargs), **kwargs)

    async def _cat_file(self, url, start=None, end=None, **kwargs):
        return await _request_with_retries(url, partial(super()._cat_file, url, start=start, end=end, **kwargs))


def require_remote_dataset_dimensions(
    dataset: DATASET,
    expected_dimensions: Iterable[str],
    operation_name: str,
) -> DATASET:
    missing_dimensions = sorted(set(expected_dimensions) - set(dataset.dims))
    if missing_dimensions:
        raise IncompleteRemoteDatasetError(
            f"Remote dataset opened without expected dimensions {missing_dimensions} during {operation_name}. "
            f"Available dimensions: {sorted(dataset.dims)}"
        )
    return dataset


def _file_system_arguments(url: str, storage_options: dict[str, Any] | None) -> dict[str, Any]:
    if url.startswith(("http://", "https://")):
        # The timeout is passed explicitly so it is part of the fsspec instance cache token.
        file_system_options = dict(storage_options or {})
        file_system_options["client_kwargs"] = _client_kwargs_with_timeout(file_system_options.get("client_kwargs"))
        return {"fs": RetryingHTTPFileSystem(**file_system_options)}
    return storage_options or {}


def remote_zarr_store(
    url: str,
    storage_options: dict[str, Any] | None = None,
) -> zarr.storage.FSStore:
    return zarr.storage.FSStore(
        url,
        mode="r",
        exceptions=REMOTE_ZARR_STORE_MISSING_KEY_EXCEPTIONS,
        **_file_system_arguments(url, storage_options),
    )


def open_remote_zarr(
    url: str,
    storage_options: dict[str, Any] | None = None,
    **open_dataset_keyword_arguments: Any,
) -> xarray.Dataset:
    return xarray.open_dataset(
        remote_zarr_store(url, storage_options),
        engine="zarr",
        **open_dataset_keyword_arguments,
    )


def open_remote_multizarr(
    urls: Sequence[str],
    storage_options: dict[str, Any] | None = None,
    **open_mfdataset_keyword_arguments: Any,
) -> xarray.Dataset:
    return xarray.open_mfdataset(
        [remote_zarr_store(url, storage_options) for url in urls],
        engine="zarr",
        **open_mfdataset_keyword_arguments,
    )
