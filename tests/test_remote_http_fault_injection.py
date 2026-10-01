# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import asyncio
import pickle
import threading
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from http import HTTPStatus
from multiprocessing import get_context
from pathlib import Path

import numpy
import pytest
import xarray
import zarr
from aiohttp import web

import oceanbench.core.runtime_configuration as runtime_configuration
from oceanbench.core.challenger_datasets import _open_multizarr_forecasts_as_challenger_dataset
from oceanbench.core.remote_http import RetryingHTTPFileSystem, remote_zarr_store
from oceanbench.core.runtime_configuration import RuntimeConfiguration

ATTEMPTS_COUNT = 3
DISCONNECTED_CHUNK_KEY = "zos/1.0"
UNAVAILABLE_CHUNK_KEY = "zos/3.0"
TRUNCATED_CHUNK_KEY = "zos/5.0"
ABSENT_CHUNK_KEY = "zos/7.0"
ABSENT_CHUNK_INDEX = 7
INJECTED_FAULTS = {
    ".zmetadata": "disconnect",
    DISCONNECTED_CHUNK_KEY: "disconnect",
    UNAVAILABLE_CHUNK_KEY: "unavailable",
    TRUNCATED_CHUNK_KEY: "truncate",
}
FIRST_DAY = datetime(2024, 1, 3)


def _source_dataset() -> xarray.Dataset:
    values = numpy.random.default_rng(seed=7).normal(size=(10, 400))
    return xarray.Dataset({"zos": (("time", "x"), values)})


class _FaultyZarrServer:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.faults: dict[str, list[str]] = {}
        self.request_counts: Counter[str] = Counter()
        self.url = ""
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._runner: web.AppRunner | None = None

    async def _serve_disconnect(self, request: web.Request) -> web.StreamResponse:
        request.transport.abort()
        return web.Response()

    async def _serve_truncated(self, request: web.Request, body: bytes) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Length": str(len(body))})
        await response.prepare(request)
        await response.write(body[: len(body) // 2])
        request.transport.close()
        return response

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        key = request.match_info["key"]
        self.request_counts[key] += 1
        pending_faults = self.faults.get(key, [])
        fault = pending_faults.pop(0) if pending_faults else None
        path = self.directory / key
        if fault == "disconnect":
            return await self._serve_disconnect(request)
        if fault == "unavailable":
            return web.Response(status=HTTPStatus.SERVICE_UNAVAILABLE)
        if not path.is_file():
            return web.Response(status=HTTPStatus.NOT_FOUND)
        if fault == "truncate":
            return await self._serve_truncated(request, path.read_bytes())
        return web.Response(body=path.read_bytes())

    async def _start(self) -> None:
        application = web.Application()
        application.router.add_get("/{key:.*}", self._handle)
        self._runner = web.AppRunner(application, access_log=None)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self._ready.set()

    def start(self) -> None:
        threading.Thread(target=self._run, daemon=True).start()
        self._ready.wait()

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._start())
        self._loop.run_forever()

    def stop(self) -> None:
        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result()
        self._loop.call_soon_threadsafe(self._loop.stop)


@pytest.fixture
def faulty_server(tmp_path, monkeypatch):
    served_directory = tmp_path / "served"
    _source_dataset().to_zarr(
        served_directory / "forecast.zarr",
        consolidated=True,
        encoding={"zos": {"chunks": (1, 400)}},
    )
    (served_directory / "forecast.zarr" / ABSENT_CHUNK_KEY).unlink()
    monkeypatch.setattr(
        runtime_configuration, "_runtime_configuration", RuntimeConfiguration(remote_retries=ATTEMPTS_COUNT)
    )
    monkeypatch.setattr("oceanbench.core.remote_http._retry_backoff_seconds", lambda _attempt: 0)
    server = _FaultyZarrServer(served_directory / "forecast.zarr")
    server.start()
    yield server
    server.stop()


def _expected_values() -> numpy.ndarray:
    expected = _source_dataset().zos.values.copy()
    expected[ABSENT_CHUNK_INDEX] = numpy.nan
    return expected


def _inject_one_fault_per_kind(server: _FaultyZarrServer) -> None:
    server.faults.update({key: [fault] for key, fault in INJECTED_FAULTS.items()})


def test_injected_faults_are_retried_and_values_match_the_source(faulty_server) -> None:
    _inject_one_fault_per_kind(faulty_server)

    values = xarray.open_zarr(remote_zarr_store(faulty_server.url), chunks={}).zos.values

    numpy.testing.assert_array_equal(values, _expected_values())
    assert {key: faulty_server.request_counts[key] for key in (*INJECTED_FAULTS, ABSENT_CHUNK_KEY)} == {
        **{key: 2 for key in INJECTED_FAULTS},
        ABSENT_CHUNK_KEY: 1,
    }


def test_stage_build_with_faults_mid_write_matches_the_source_and_leaves_no_temporary_store(
    faulty_server, monkeypatch, tmp_path
) -> None:
    stage_directory = tmp_path / "stage"
    monkeypatch.setattr(
        runtime_configuration,
        "_runtime_configuration",
        RuntimeConfiguration(
            staged_components=("challenger",), stage_directory=str(stage_directory), remote_retries=ATTEMPTS_COUNT
        ),
    )
    _inject_one_fault_per_kind(faulty_server)

    def _forecast_dataset_path(_start_datetime: datetime) -> str:
        return faulty_server.url

    staged = _open_multizarr_forecasts_as_challenger_dataset(_forecast_dataset_path, first_day_datetimes=[FIRST_DAY])

    numpy.testing.assert_array_equal(staged.zos.values[0], _expected_values())
    assert faulty_server.request_counts[TRUNCATED_CHUNK_KEY] == 2
    stage_week_directory = stage_directory / "challenger-forecast-10d"
    assert sorted(path.name for path in stage_week_directory.iterdir()) == ["20240103.zarr"]


def _read_chunk_in_worker(pickled_store: bytes) -> list[float]:
    return zarr.open_array(pickle.loads(pickled_store), mode="r", path="zos")[1].tolist()


def test_remote_store_pickles_into_a_spawned_worker_with_its_retrying_file_system(faulty_server) -> None:
    store = remote_zarr_store(faulty_server.url)
    assert type(store.fs) is RetryingHTTPFileSystem
    assert pickle.loads(pickle.dumps(store.fs)) is store.fs

    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as executor:
        values = executor.submit(_read_chunk_in_worker, pickle.dumps(store)).result()

    numpy.testing.assert_array_equal(values, _source_dataset().zos.values[1])
