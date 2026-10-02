# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import logging
from http import HTTPStatus

import aiohttp
import multidict
import pytest
import yarl
from fsspec.implementations.http import HTTPFileSystem

from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.remote_http import RetryingHTTPFileSystem, _retry_backoff_seconds
from oceanbench.core.runtime_configuration import DEFAULT_REMOTE_HTTP_RETRIES

URL = "http://remote.test/store.zarr/zos/0.0"
PAYLOAD = b"chunk bytes"


def _response_error(status: int) -> aiohttp.ClientResponseError:
    request_info = aiohttp.RequestInfo(
        yarl.URL(URL), "GET", multidict.CIMultiDictProxy(multidict.CIMultiDict()), yarl.URL(URL)
    )
    return aiohttp.ClientResponseError(request_info, (), status=status, message="simulated")


def _payload_error() -> aiohttp.ClientPayloadError:
    return aiohttp.ClientPayloadError("Response payload is not completed: Not enough data to satisfy content length")


@pytest.fixture
def attempts_count(monkeypatch) -> int:
    configured_attempts_count = 4
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES.value, str(configured_attempts_count))
    monkeypatch.setattr("oceanbench.core.remote_http._retry_backoff_seconds", lambda _attempt: 0)
    return configured_attempts_count


@pytest.fixture
def parent_request(monkeypatch):
    def install(method_name: str, errors: list[BaseException], result: object) -> list[str]:
        calls: list[str] = []

        async def request(self, url, *arguments, **keyword_arguments):
            calls.append(url)
            if len(calls) <= len(errors):
                raise errors[len(calls) - 1]
            return result

        monkeypatch.setattr(HTTPFileSystem, method_name, request)
        return calls

    return install


@pytest.mark.parametrize(
    "transient_error",
    [
        aiohttp.ServerDisconnectedError(),
        aiohttp.ClientConnectorError(None, OSError(111, "Connection refused")),
        aiohttp.ServerTimeoutError("simulated timeout"),
        TimeoutError("simulated timeout"),
        ConnectionResetError(104, "Connection reset by peer"),
        _payload_error(),
        _response_error(HTTPStatus.INTERNAL_SERVER_ERROR),
        _response_error(HTTPStatus.BAD_GATEWAY),
        _response_error(HTTPStatus.SERVICE_UNAVAILABLE),
        _response_error(HTTPStatus.TOO_MANY_REQUESTS),
    ],
    ids=lambda error: type(error).__name__,
)
def test_transient_failures_are_retried_until_the_request_succeeds(
    attempts_count, parent_request, transient_error
) -> None:
    transient_failures_count = attempts_count - 1
    calls = parent_request("_cat_file", [transient_error] * transient_failures_count, PAYLOAD)

    assert RetryingHTTPFileSystem().cat_file(URL) == PAYLOAD
    assert len(calls) == transient_failures_count + 1


def test_not_found_raises_after_one_request(attempts_count, parent_request) -> None:
    calls = parent_request("_cat_file", [FileNotFoundError(URL)], PAYLOAD)

    with pytest.raises(FileNotFoundError):
        RetryingHTTPFileSystem().cat_file(URL)
    assert len(calls) == 1


@pytest.mark.parametrize("status", [HTTPStatus.BAD_REQUEST, HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN])
def test_client_errors_raise_after_one_request(attempts_count, parent_request, status) -> None:
    calls = parent_request("_cat_file", [_response_error(status)], PAYLOAD)

    with pytest.raises(aiohttp.ClientResponseError) as raised:
        RetryingHTTPFileSystem().cat_file(URL)
    assert raised.value.status == status
    assert len(calls) == 1


def test_exhausted_retries_reraise_the_original_error(attempts_count, parent_request) -> None:
    errors = [aiohttp.ServerDisconnectedError() for _ in range(attempts_count)]
    calls = parent_request("_cat_file", errors, PAYLOAD)

    with pytest.raises(aiohttp.ServerDisconnectedError) as raised:
        RetryingHTTPFileSystem().cat_file(URL)
    assert raised.value is errors[-1]
    assert len(calls) == attempts_count


def test_retries_log_at_info_and_never_at_warning(attempts_count, parent_request, caplog) -> None:
    caplog.set_level(logging.DEBUG)
    parent_request(
        "_cat_file", [_response_error(HTTPStatus.SERVICE_UNAVAILABLE), aiohttp.ServerDisconnectedError()], PAYLOAD
    )

    RetryingHTTPFileSystem().cat_file(URL)

    retry_records = [record for record in caplog.records if record.name == "oceanbench.core.remote_http"]
    assert [record.levelno for record in retry_records] == [logging.INFO, logging.INFO]
    assert all(URL in record.getMessage() for record in retry_records)
    assert [record for record in caplog.records if record.levelno >= logging.WARNING] == []


def test_backoff_doubles_and_is_capped_so_default_retries_ride_out_a_minute() -> None:
    assert [_retry_backoff_seconds(attempt) for attempt in range(1, 7)] == [4, 8, 16, 32, 32, 32]
    assert sum(_retry_backoff_seconds(attempt) for attempt in range(1, DEFAULT_REMOTE_HTTP_RETRIES)) == 60
