# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from oceanbench.core.classIV_support import _compute_with_remote_retries
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.remote_http import with_remote_http_retries


class FakeAiohttpPayloadError(Exception):
    pass


FakeAiohttpPayloadError.__module__ = "aiohttp.client_exceptions"


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
