# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import asyncio
import threading
import time

import aiohttp
import pytest
from aiohttp import web

import oceanbench.core.runtime_configuration as runtime_configuration
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.remote_http import remote_http_client_timeout, remote_zarr_store
from oceanbench.core.runtime_configuration import RuntimeConfiguration

ATTEMPTS_COUNT = 3
SOCK_READ_TIMEOUT_SECONDS = 0.5
PAYLOAD = b"chunk bytes"
CHUNK_KEY = "zos/0.0"


class _SilentServer:
    def __init__(self, silent_requests_count: int) -> None:
        self.silent_requests_count = silent_requests_count
        self.requests_count = 0
        self.url = ""
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._released = asyncio.Event()
        self._runner: web.AppRunner | None = None

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        self.requests_count += 1
        if self.requests_count <= self.silent_requests_count:
            await self._released.wait()
        return web.Response(body=PAYLOAD)

    async def _start(self) -> None:
        application = web.Application()
        application.router.add_get("/{key:.*}", self._handle)
        self._runner = web.AppRunner(application, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}/store.zarr"
        self._ready.set()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start())
        self._loop.run_forever()

    def stop(self) -> None:
        self._loop.call_soon_threadsafe(self._released.set)
        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)


@pytest.fixture
def silent_server(monkeypatch):
    monkeypatch.setenv(
        OceanbenchEnvironmentVariable.OCEANBENCH_HTTP_SOCK_READ_TIMEOUT.value, str(SOCK_READ_TIMEOUT_SECONDS)
    )
    monkeypatch.setattr(
        runtime_configuration, "_runtime_configuration", RuntimeConfiguration(remote_retries=ATTEMPTS_COUNT)
    )
    monkeypatch.setattr("oceanbench.core.remote_http._retry_backoff_seconds", lambda _attempt: 0)
    servers: list[_SilentServer] = []

    def start(silent_requests_count: int) -> _SilentServer:
        server = _SilentServer(silent_requests_count)
        server.start()
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.stop()


def test_unanswered_request_times_out_and_is_retried(silent_server) -> None:
    server = silent_server(silent_requests_count=1)
    store = remote_zarr_store(server.url)

    started = time.monotonic()
    payload = store.fs.cat_file(f"{server.url}/{CHUNK_KEY}")

    assert payload == PAYLOAD
    assert server.requests_count == 2
    assert time.monotonic() - started < 10 * SOCK_READ_TIMEOUT_SECONDS


def test_never_answered_request_fails_after_all_attempts_in_bounded_time(silent_server) -> None:
    server = silent_server(silent_requests_count=ATTEMPTS_COUNT)
    store = remote_zarr_store(server.url)

    started = time.monotonic()
    with pytest.raises((aiohttp.ServerTimeoutError, TimeoutError)):
        store.fs.cat_file(f"{server.url}/{CHUNK_KEY}")

    assert server.requests_count == ATTEMPTS_COUNT
    assert time.monotonic() - started < 10 * ATTEMPTS_COUNT * SOCK_READ_TIMEOUT_SECONDS


def test_client_timeout_has_no_total_limit_and_reads_the_environment(monkeypatch) -> None:
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_HTTP_SOCK_READ_TIMEOUT.value, "7")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_HTTP_SOCK_CONNECT_TIMEOUT.value, "3")

    assert remote_http_client_timeout() == aiohttp.ClientTimeout(total=None, sock_connect=3.0, sock_read=7.0)


def test_client_timeout_defaults(monkeypatch) -> None:
    monkeypatch.delenv(OceanbenchEnvironmentVariable.OCEANBENCH_HTTP_SOCK_READ_TIMEOUT.value, raising=False)
    monkeypatch.delenv(OceanbenchEnvironmentVariable.OCEANBENCH_HTTP_SOCK_CONNECT_TIMEOUT.value, raising=False)

    assert remote_http_client_timeout() == aiohttp.ClientTimeout(total=None, sock_connect=30.0, sock_read=120.0)


def test_remote_store_file_system_carries_the_client_timeout(monkeypatch) -> None:
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_HTTP_SOCK_READ_TIMEOUT.value, "11")

    store = remote_zarr_store("http://remote.test/store.zarr")

    assert store.fs.client_kwargs["timeout"].sock_read == 11.0
