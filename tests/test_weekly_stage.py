# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime

import numpy
import pytest
import xarray

import oceanbench.core.runtime_configuration as runtime_configuration
from oceanbench.core.challenger_datasets import _open_multizarr_forecasts_as_challenger_dataset
from oceanbench.core.runtime_configuration import RuntimeConfiguration

FIRST_DAY = datetime(2024, 1, 3)


def _use_runtime_configuration(monkeypatch, configuration: RuntimeConfiguration) -> None:
    monkeypatch.setattr(runtime_configuration, "_runtime_configuration", configuration)


def _open_forecast(zarr_store_url: str) -> xarray.Dataset:
    def _forecast_dataset_path(_start_datetime: datetime) -> str:
        return zarr_store_url

    return _open_multizarr_forecasts_as_challenger_dataset(_forecast_dataset_path, first_day_datetimes=[FIRST_DAY])


def test_staged_challenger_week_is_identical_to_the_direct_remote_open(monkeypatch, tmp_path, zarr_store_url) -> None:
    _use_runtime_configuration(monkeypatch, RuntimeConfiguration())
    direct = _open_forecast(zarr_store_url).load()

    _use_runtime_configuration(
        monkeypatch, RuntimeConfiguration(staged_components=("challenger",), stage_directory=str(tmp_path))
    )
    staged = _open_forecast(zarr_store_url).load()

    xarray.testing.assert_equal(staged, direct)
    assert {name: variable.dtype for name, variable in staged.variables.items()} == {
        name: variable.dtype for name, variable in direct.variables.items()
    }
    assert (tmp_path / "challenger-forecast-10d" / "20240103.zarr").is_dir()


def test_failed_chunk_download_during_stage_build_raises_and_leaves_no_finished_stage(
    monkeypatch, tmp_path, zarr_store_url, fail_chunk_download
) -> None:
    _use_runtime_configuration(
        monkeypatch,
        RuntimeConfiguration(staged_components=("challenger",), stage_directory=str(tmp_path), remote_retries=1),
    )
    fail_chunk_download(failure_count=10)

    with pytest.raises(TimeoutError):
        _open_forecast(zarr_store_url)

    stage_directory = tmp_path / "challenger-forecast-10d"
    assert not (stage_directory / "20240103.zarr").exists()
    assert [path.name for path in stage_directory.iterdir()] == ["20240103.zarr.tmp"]


def test_challenger_week_time_axis_becomes_lead_day_index_with_first_day_from_the_file_name(
    monkeypatch, tmp_path
) -> None:
    _use_runtime_configuration(monkeypatch, RuntimeConfiguration())
    xarray.Dataset(
        {"zos": (("time", "x"), numpy.arange(20.0).reshape(10, 2))},
        coords={"time": numpy.arange("2024-01-03", "2024-01-13", dtype="datetime64[D]")},
    ).to_zarr(tmp_path / "20240103.zarr")

    def _persistence_dataset_path(start_datetime: datetime) -> str:
        return str(tmp_path / f"{start_datetime:%Y%m%d}.zarr")

    challenger = _open_multizarr_forecasts_as_challenger_dataset(
        _persistence_dataset_path, first_day_datetimes=[FIRST_DAY]
    )

    assert challenger["lead_day_index"].values.tolist() == list(range(10))
    numpy.testing.assert_array_equal(
        challenger["first_day_datetime"].values, numpy.array(["2024-01-03"], dtype="datetime64[ns]")
    )
    assert "time" not in challenger.variables
    assert challenger["zos"].values[0, :, 0].tolist() == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]
