# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import pytest
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.references import swot_observations


def _write_swot_observation_day_store(
    store_path,
    observation_datetimes: list[str],
    subsample_factor: int = swot_observations.EXPECTED_SWOT_OBSERVATIONS_SUBSAMPLE_FACTOR,
) -> None:
    observations_count = len(observation_datetimes)
    observation_dimension = swot_observations.OBSERVATION_DIMENSION_KEY
    day_store = xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                observation_dimension,
                numpy.linspace(-0.25, 0.25, observations_count),
            ),
            Dimension.LATITUDE.key(): (observation_dimension, numpy.linspace(-10.0, 10.0, observations_count)),
            Dimension.LONGITUDE.key(): (observation_dimension, numpy.linspace(-170.0, 170.0, observations_count)),
            Dimension.TIME.key(): (observation_dimension, pandas.to_datetime(observation_datetimes).values),
        },
        attrs={
            swot_observations.SWOT_OBSERVATIONS_BASIS_VERSION_ATTRIBUTE: (
                swot_observations.EXPECTED_SWOT_OBSERVATIONS_BASIS_VERSION
            ),
            swot_observations.SWOT_OBSERVATIONS_SUBSAMPLE_FACTOR_ATTRIBUTE: subsample_factor,
            "source_pass_files": ["SWOT_L3_LR_SSH_Basic_test.nc"],
        },
    )
    day_store.to_zarr(
        store_path,
        mode="w",
        consolidated=True,
        encoding={
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): {
                "dtype": "int32",
                "scale_factor": 0.0001,
                "_FillValue": 2147483647,
            },
            Dimension.TIME.key(): {"units": "seconds since 2000-01-01 00:00:00.0", "dtype": "float64"},
        },
    )


def _use_local_day_stores(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        swot_observations,
        "swot_observation_path",
        lambda day: str(tmp_path / f"{pandas.Timestamp(day):%Y%m%d}.zarr"),
    )


def _selected_swot_observations(observation_days: list[str], first_days: list[str]) -> xarray.Dataset:
    first_day_datetimes = numpy.array(first_days, dtype="datetime64[ns]")
    return swot_observations._selected_swot_observations_dataset(
        observation_days=numpy.array(observation_days, dtype="datetime64[D]"),
        first_day_timestamps=pandas.to_datetime(first_day_datetimes),
        first_day_datetimes=first_day_datetimes,
        lead_days_count=10,
    )


def test_selected_swot_observations_have_the_shape_expected_by_the_class4(tmp_path, monkeypatch) -> None:
    _write_swot_observation_day_store(tmp_path / "20240103.zarr", ["2024-01-03T01:00", "2024-01-03T23:00"])
    _use_local_day_stores(monkeypatch, tmp_path)

    selected = _selected_swot_observations(["2024-01-03"], ["2024-01-03"])

    assert selected.sizes == {swot_observations.OBSERVATION_DIMENSION_KEY: 2}
    assert set(selected.variables) == {
        Dimension.TIME.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
        Dimension.DEPTH.key(),
        Dimension.FIRST_DAY_DATETIME.key(),
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    }
    assert (selected[Dimension.DEPTH.key()].values == 0.0).all()
    numpy.testing.assert_allclose(selected[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()].values, [-0.25, 0.25])


def test_swot_observations_are_selected_once_per_overlapping_forecast(tmp_path, monkeypatch) -> None:
    _write_swot_observation_day_store(tmp_path / "20240110.zarr", ["2024-01-10T12:00"])
    _use_local_day_stores(monkeypatch, tmp_path)

    selected = _selected_swot_observations(["2024-01-10"], ["2024-01-03", "2024-01-10"])

    assert pandas.to_datetime(selected[Dimension.FIRST_DAY_DATETIME.key()].values).tolist() == [
        pandas.Timestamp("2024-01-03"),
        pandas.Timestamp("2024-01-10"),
    ]


def test_days_without_converted_swot_observations_are_skipped(tmp_path, monkeypatch) -> None:
    _write_swot_observation_day_store(tmp_path / "20240103.zarr", ["2024-01-03T12:00"])
    _write_swot_observation_day_store(tmp_path / "20240105.zarr", ["2024-01-05T12:00"])
    _use_local_day_stores(monkeypatch, tmp_path)

    selected = _selected_swot_observations(["2024-01-03", "2024-01-04", "2024-01-05"], ["2024-01-03"])

    assert selected.sizes[swot_observations.OBSERVATION_DIMENSION_KEY] == 2


def test_no_converted_swot_observation_day_raises_unavailable_error(tmp_path, monkeypatch) -> None:
    _use_local_day_stores(monkeypatch, tmp_path)

    with pytest.raises(swot_observations.ObservationDataUnavailableError, match="No converted SWOT"):
        _selected_swot_observations(["2024-01-03"], ["2024-01-03"])


def test_unexpected_swot_subsample_factor_raises_with_found_factor(tmp_path, monkeypatch) -> None:
    _write_swot_observation_day_store(tmp_path / "20240103.zarr", ["2024-01-03T12:00"], subsample_factor=1)
    _use_local_day_stores(monkeypatch, tmp_path)

    with pytest.raises(swot_observations.ObservationBasisVersionError, match="found 'v2.1' and 1"):
        _selected_swot_observations(["2024-01-03"], ["2024-01-03"])


def test_challenger_starting_before_2024_raises_unavailable_error() -> None:
    challenger_dataset = xarray.Dataset(
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2023-12-27"], dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(10),
        }
    )

    with pytest.raises(swot_observations.ObservationDataUnavailableError, match="2023-12-27"):
        swot_observations.swot_observations(challenger_dataset)
