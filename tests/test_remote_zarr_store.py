# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import functools
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

import numpy
import pytest
import xarray
from fsspec.implementations.http import HTTPFileSystem

from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.remote_http import open_remote_zarr, with_remote_http_retries

_FAILING_CHUNK_KEY = "zos/2.0"


class _SilentRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, *arguments) -> None:
        pass


@pytest.fixture(scope="module")
def zarr_store_url(tmp_path_factory) -> str:
    served_directory = tmp_path_factory.mktemp("remote-zarr")
    dataset = xarray.Dataset({"zos": (("time", "x"), numpy.arange(20.0).reshape(4, 5))})
    dataset.to_zarr(
        served_directory / "forecast.zarr",
        consolidated=True,
        encoding={"zos": {"chunks": (1, 5)}},
    )
    handler = functools.partial(_SilentRequestHandler, directory=str(served_directory))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/forecast.zarr"
    server.shutdown()


def _fail_chunk_download(monkeypatch, failure_count: int) -> list[int]:
    remaining_failures = [failure_count]
    original_cat_file = HTTPFileSystem._cat_file

    async def cat_file(self, path, **keyword_arguments):
        if path.endswith(_FAILING_CHUNK_KEY) and remaining_failures[0] > 0:
            remaining_failures[0] -= 1
            raise TimeoutError("simulated chunk download timeout")
        return await original_cat_file(self, path, **keyword_arguments)

    monkeypatch.setattr(HTTPFileSystem, "_cat_file", cat_file)
    return remaining_failures


def test_failed_chunk_download_raises_instead_of_staging_fill_values(monkeypatch, zarr_store_url) -> None:
    _fail_chunk_download(monkeypatch, failure_count=1)

    with pytest.raises(TimeoutError):
        open_remote_zarr(zarr_store_url).zos.load()


def test_with_remote_http_retries_retries_failed_chunk_download(monkeypatch, zarr_store_url) -> None:
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES.value, "2")
    monkeypatch.setattr("oceanbench.core.remote_http.sleep", lambda _seconds: None)
    remaining_failures = _fail_chunk_download(monkeypatch, failure_count=1)

    dataset = with_remote_http_retries(
        "remote chunk read",
        lambda: open_remote_zarr(zarr_store_url).zos.load(),
    )

    assert remaining_failures[0] == 0
    assert not numpy.isnan(dataset.values).any()


def test_absent_chunk_still_reads_as_fill_value(monkeypatch, zarr_store_url) -> None:
    original_cat_file = HTTPFileSystem._cat_file

    async def cat_file(self, path, **keyword_arguments):
        if path.endswith(_FAILING_CHUNK_KEY):
            raise FileNotFoundError(path)
        return await original_cat_file(self, path, **keyword_arguments)

    monkeypatch.setattr(HTTPFileSystem, "_cat_file", cat_file)

    values = open_remote_zarr(zarr_store_url).zos.load().values

    assert numpy.isnan(values[2]).all()
    assert not numpy.isnan(values[[0, 1, 3]]).any()
