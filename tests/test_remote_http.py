# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from oceanbench.core.classIV_support import _compute_with_remote_retries
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.remote_http import (
    _RetryingRemoteMapper,
    _is_retriable_remote_data_error,
    resilient_zarr_store,
    with_remote_http_retries,
)


class FakeAiohttpPayloadError(Exception):
    pass


FakeAiohttpPayloadError.__module__ = "aiohttp.client_exceptions"


class FakeAiohttpResponseError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status


FakeAiohttpResponseError.__module__ = "aiohttp.client_exceptions"


class _ScriptedMapper:
    def __init__(self, scripted_getitems_results: list[dict]) -> None:
        self._scripted_getitems_results = scripted_getitems_results
        self.getitems_calls: list[list[str]] = []

    def getitems(self, keys, on_error="return"):
        self.getitems_calls.append(list(keys))
        return self._scripted_getitems_results.pop(0)


def _configure_fast_retries(monkeypatch) -> None:
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES.value, "2")
    monkeypatch.setattr("oceanbench.core.remote_http.sleep", lambda _seconds: None)


def test_with_remote_http_retries_retries_incomplete_payload_messages(monkeypatch) -> None:
    _configure_fast_retries(monkeypatch)
    attempts = 0

    def callback() -> str:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("Response payload is not completed")
        return "loaded"

    assert with_remote_http_retries("remote read", callback) == "loaded"
    assert attempts == 2


def test_class4_compute_uses_remote_http_retries(monkeypatch) -> None:
    _configure_fast_retries(monkeypatch)

    class RemoteBackedArray:
        def __init__(self) -> None:
            self.calls = 0

        def compute(self) -> str:
            self.calls += 1
            if self.calls == 1:
                raise FakeAiohttpPayloadError("Not enough data to satisfy content length header")
            return "loaded"

    remote_backed_array = RemoteBackedArray()

    assert _compute_with_remote_retries("Class IV model read", remote_backed_array) == "loaded"
    assert remote_backed_array.calls == 2


def test_resilient_mapper_retries_only_the_transiently_failed_chunk(monkeypatch) -> None:
    _configure_fast_retries(monkeypatch)
    transient_error = FakeAiohttpPayloadError("Response payload is not completed")
    scripted_mapper = _ScriptedMapper(
        [
            {"0.0": b"chunk-a", "0.1": transient_error},
            {"0.1": b"chunk-b"},
        ]
    )
    resilient_mapper = _RetryingRemoteMapper(scripted_mapper)

    result = resilient_mapper.getitems(["0.0", "0.1"], on_error="return")

    assert result == {"0.0": b"chunk-a", "0.1": b"chunk-b"}
    assert scripted_mapper.getitems_calls == [["0.0", "0.1"], ["0.1"]]


def test_resilient_mapper_passes_missing_chunk_through_without_retry(monkeypatch) -> None:
    _configure_fast_retries(monkeypatch)
    missing_chunk_error = KeyError("0.2")
    scripted_mapper = _ScriptedMapper([{"0.0": b"chunk-a", "0.2": missing_chunk_error}])
    resilient_mapper = _RetryingRemoteMapper(scripted_mapper)

    result = resilient_mapper.getitems(["0.0", "0.2"], on_error="return")

    assert result["0.0"] == b"chunk-a"
    assert isinstance(result["0.2"], KeyError)
    assert scripted_mapper.getitems_calls == [["0.0", "0.2"]]


def test_resilient_mapper_getitem_retries_transient_failure(monkeypatch) -> None:
    _configure_fast_retries(monkeypatch)

    class FlakyMapper:
        def __init__(self) -> None:
            self.calls = 0

        def __getitem__(self, key: str) -> bytes:
            self.calls += 1
            if self.calls == 1:
                raise FakeAiohttpPayloadError("Server disconnected")
            return b"value"

    flaky_mapper = FlakyMapper()
    resilient_mapper = _RetryingRemoteMapper(flaky_mapper)

    assert resilient_mapper["x"] == b"value"
    assert flaky_mapper.calls == 2


def test_http_status_codes_decide_retriability() -> None:
    assert _is_retriable_remote_data_error(FakeAiohttpResponseError(502)) is True
    assert _is_retriable_remote_data_error(FakeAiohttpResponseError(503)) is True
    assert _is_retriable_remote_data_error(FakeAiohttpResponseError(403)) is False
    assert _is_retriable_remote_data_error(FakeAiohttpResponseError(404)) is False


def test_resilient_zarr_store_wraps_the_chunk_mapper() -> None:
    store = resilient_zarr_store("memory://resilient-zarr-store-test")
    assert isinstance(store.map, _RetryingRemoteMapper)
