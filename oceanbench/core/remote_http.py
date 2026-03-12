# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable, Iterable
from os import environ
from time import sleep
from typing import TypeVar
import logging

from oceanbench.core.instrumentation import log_event

REMOTE_HTTP_RETRIES_ENVIRONMENT_VARIABLE = "OCEANBENCH_REMOTE_HTTP_RETRIES"
DEFAULT_REMOTE_HTTP_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2
RETRIABLE_HTTP_ERROR_TOKENS = (
    "Server disconnected",
    "Connection reset by peer",
)
RETRIABLE_REMOTE_BACKEND_MODULE_PREFIXES = (
    "aiohttp",
    "botocore",
)

REMOTE_ZARR_LOGGER = logging.getLogger(__name__)

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


def _is_retriable_remote_data_error(error: Exception) -> bool:
    return any(
        isinstance(exception, RetriableRemoteDataError)
        or _originates_from_retriable_remote_backend(exception)
        or any(token in str(exception) for token in RETRIABLE_HTTP_ERROR_TOKENS)
        for exception in _exception_chain(error)
    )


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
    retry_count = int(environ.get(REMOTE_HTTP_RETRIES_ENVIRONMENT_VARIABLE, DEFAULT_REMOTE_HTTP_RETRIES))
    for attempt in range(1, retry_count + 1):
        try:
            return callback()
        except Exception as error:
            is_retriable_error = _is_retriable_remote_data_error(error)
            if not is_retriable_error or attempt == retry_count:
                log_event(
                    "remote_operation_failed",
                    operation=operation_name,
                    attempt=attempt,
                    retry_count=retry_count,
                    retriable=is_retriable_error,
                    error_type=error.__class__.__name__,
                    error_module=error.__class__.__module__,
                    error_message=str(error),
                )
                raise
            backoff_seconds = DEFAULT_RETRY_BACKOFF_SECONDS * attempt
            log_event(
                "remote_retry_scheduled",
                operation=operation_name,
                attempt=attempt,
                retry_count=retry_count,
                backoff_seconds=backoff_seconds,
                error_type=error.__class__.__name__,
                error_module=error.__class__.__module__,
                error_message=str(error),
            )
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
