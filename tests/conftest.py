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

FAILING_CHUNK_KEY = "zos/2.0"


class _SilentRequestHandler(SimpleHTTPRequestHandler):
    def log_message(self, *arguments) -> None:
        pass


@pytest.fixture(scope="session")
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


@pytest.fixture
def fail_chunk_download(monkeypatch):
    def install(failure_count: int) -> list[int]:
        remaining_failures = [failure_count]
        original_cat_file = HTTPFileSystem._cat_file

        async def cat_file(self, path, **keyword_arguments):
            if path.endswith(FAILING_CHUNK_KEY) and remaining_failures[0] > 0:
                remaining_failures[0] -= 1
                raise TimeoutError("simulated chunk download timeout")
            return await original_cat_file(self, path, **keyword_arguments)

        monkeypatch.setattr(HTTPFileSystem, "_cat_file", cat_file)
        return remaining_failures

    return install
