# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
from fsspec.implementations.http import HTTPFileSystem

from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.remote_http import open_remote_zarr, with_remote_http_retries

_ABSENT_CHUNK_KEY = "zos/2.0"


def test_failed_chunk_download_raises_instead_of_staging_fill_values(fail_chunk_download, zarr_store_url) -> None:
    fail_chunk_download(failure_count=1)

    with pytest.raises(TimeoutError):
        open_remote_zarr(zarr_store_url).zos.load()


def test_with_remote_http_retries_retries_failed_chunk_download(
    monkeypatch, fail_chunk_download, zarr_store_url
) -> None:
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES.value, "2")
    monkeypatch.setattr("oceanbench.core.remote_http.sleep", lambda _seconds: None)
    remaining_failures = fail_chunk_download(failure_count=1)

    dataset = with_remote_http_retries(
        "remote chunk read",
        lambda: open_remote_zarr(zarr_store_url).zos.load(),
    )

    assert remaining_failures[0] == 0
    assert not numpy.isnan(dataset.values).any()


def test_absent_chunk_still_reads_as_fill_value(monkeypatch, zarr_store_url) -> None:
    original_cat_file = HTTPFileSystem._cat_file

    async def cat_file(self, path, **keyword_arguments):
        if path.endswith(_ABSENT_CHUNK_KEY):
            raise FileNotFoundError(path)
        return await original_cat_file(self, path, **keyword_arguments)

    monkeypatch.setattr(HTTPFileSystem, "_cat_file", cat_file)

    values = open_remote_zarr(zarr_store_url).zos.load().values

    assert numpy.isnan(values[2]).all()
    assert not numpy.isnan(values[[0, 1, 3]]).any()
