# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core.computed_dataset_cache import cached_computed_dataset
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable


def _build_three_value_dataset(build_calls: list) -> xarray.Dataset:
    build_calls.append(1)
    return xarray.Dataset({"x": ("a", numpy.arange(3.0))})


def test_cached_computed_dataset_recomputes_without_cache_directory(monkeypatch) -> None:
    monkeypatch.delenv(OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE.value, raising=False)
    build_calls: list = []

    first = cached_computed_dataset("recompute-key", lambda: _build_three_value_dataset(build_calls))
    second = cached_computed_dataset("recompute-key", lambda: _build_three_value_dataset(build_calls))

    assert first["x"].values.tolist() == [0.0, 1.0, 2.0]
    assert second["x"].values.tolist() == [0.0, 1.0, 2.0]
    assert len(build_calls) == 2


def test_cached_computed_dataset_persists_and_reuses(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE.value, str(tmp_path))
    build_calls: list = []

    first = cached_computed_dataset("persisted-key", lambda: _build_three_value_dataset(build_calls))
    second = cached_computed_dataset("persisted-key", lambda: _build_three_value_dataset(build_calls))

    assert first["x"].values.tolist() == [0.0, 1.0, 2.0]
    assert second["x"].values.tolist() == [0.0, 1.0, 2.0]
    assert len(build_calls) == 1
    assert (tmp_path / "computed" / "persisted-key.zarr").exists()
