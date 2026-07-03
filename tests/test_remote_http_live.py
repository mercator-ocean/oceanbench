# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Bounded live smoke check for the resilient chunk-fetch engine against a real public
zarr store. Skipped by default (it needs network); enable with
``OCEANBENCH_RUN_LIVE_TESTS=1 pytest tests/test_remote_http_live.py``."""

import os

import pytest
import xarray

from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.remote_http import resilient_zarr_store

_PUBLIC_GLONET_WEEKLY_ZARR_URL = (
    "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/ml-forecast-outputs/glonet/20240103.zarr"
)

pytestmark = pytest.mark.skipif(
    os.environ.get("OCEANBENCH_RUN_LIVE_TESTS") != "1",
    reason="Live network test; set OCEANBENCH_RUN_LIVE_TESTS=1 to run.",
)


def _small_corner_slice(data_array: xarray.DataArray) -> xarray.DataArray:
    return data_array.isel({dimension: slice(0, 2) for dimension in data_array.dims})


def test_resilient_store_caches_a_live_read_and_reuses_it_without_network(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE.value, str(tmp_path))

    first_dataset = xarray.open_dataset(resilient_zarr_store(_PUBLIC_GLONET_WEEKLY_ZARR_URL), engine="zarr")
    variable_name = next(iter(first_dataset.data_vars))
    first_values = _small_corner_slice(first_dataset[variable_name]).load()

    assert any(path.is_file() for path in tmp_path.rglob("*"))

    def _forbid_network(*_arguments, **_keyword_arguments):
        raise RuntimeError("Second read hit the network instead of the local cache.")

    cache_only_store = resilient_zarr_store(_PUBLIC_GLONET_WEEKLY_ZARR_URL)
    cache_only_store.map._inner_mapper.__getitem__ = _forbid_network
    cache_only_store.map._inner_mapper.getitems = _forbid_network

    second_dataset = xarray.open_dataset(cache_only_store, engine="zarr")
    second_values = _small_corner_slice(second_dataset[variable_name]).load()

    assert second_values.shape == first_values.shape
    xarray.testing.assert_identical(second_values, first_values)
