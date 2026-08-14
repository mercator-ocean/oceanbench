# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import pytest
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.classIV import _create_observations_dataframe
from oceanbench.core.references import observations


def _observation_source() -> xarray.Dataset:
    observation_dimension = "obs"
    variables = {
        Dimension.TIME.key(): (
            observation_dimension,
            pandas.to_datetime(["2024-01-03", "2024-01-10", "2024-01-12", "2024-01-14"]).values,
        ),
        Dimension.LATITUDE.key(): (observation_dimension, [0.0, 1.0, 2.0, 3.0]),
        Dimension.LONGITUDE.key(): (observation_dimension, [10.0, 11.0, 12.0, 13.0]),
        Dimension.DEPTH.key(): (observation_dimension, [0.0, 0.0, 0.0, 0.0]),
    }
    for variable in (
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
        Variable.SEA_WATER_SALINITY,
        Variable.EASTWARD_SEA_WATER_VELOCITY,
        Variable.NORTHWARD_SEA_WATER_VELOCITY,
    ):
        variables[variable.key()] = (observation_dimension, [1.0, 2.0, 3.0, 4.0])
    return xarray.Dataset(variables)


def test_selected_observations_dataset_preserves_overlapping_forecast_windows(monkeypatch) -> None:
    source = _observation_source()
    first_day_datetimes = numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]")

    monkeypatch.setattr(observations, "open_mfdataset", lambda *_, **__: source)
    monkeypatch.setattr(observations, "require_remote_dataset_dimensions", lambda dataset, *_: dataset)

    selected = observations._selected_observations_dataset(
        observation_days=numpy.array(["2024-01-03", "2024-01-10", "2024-01-12", "2024-01-14"], dtype="datetime64[D]"),
        first_day_timestamps=pandas.to_datetime(first_day_datetimes),
        first_day_datetimes=first_day_datetimes,
        lead_days_count=10,
    )

    result = pandas.DataFrame(
        {
            "time": pandas.to_datetime(selected[Dimension.TIME.key()].values).strftime("%Y-%m-%d"),
            "first_day": pandas.to_datetime(selected[Dimension.FIRST_DAY_DATETIME.key()].values).strftime("%Y-%m-%d"),
            "value": selected[Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()].values,
        }
    )

    assert result.to_dict(orient="records") == [
        {"time": "2024-01-03", "first_day": "2024-01-03", "value": 1.0},
        {"time": "2024-01-10", "first_day": "2024-01-03", "value": 2.0},
        {"time": "2024-01-12", "first_day": "2024-01-03", "value": 3.0},
        {"time": "2024-01-10", "first_day": "2024-01-10", "value": 2.0},
        {"time": "2024-01-12", "first_day": "2024-01-10", "value": 3.0},
        {"time": "2024-01-14", "first_day": "2024-01-10", "value": 4.0},
    ]

    observations_dataframe = _create_observations_dataframe(
        selected,
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        lead_days_count=10,
    )

    assert observations_dataframe[["observation_value", "lead_day"]].to_dict(orient="records") == [
        {"observation_value": 1.0, "lead_day": 0},
        {"observation_value": 2.0, "lead_day": 7},
        {"observation_value": 3.0, "lead_day": 9},
        {"observation_value": 2.0, "lead_day": 0},
        {"observation_value": 3.0, "lead_day": 2},
        {"observation_value": 4.0, "lead_day": 4},
    ]


def test_observations_stage_path_uses_overlap_safe_version() -> None:
    assert (
        observations._observations_stage_path("2024-01-03", "2025-01-03", 10).name
        == "observations-v4-20240103-20250103-10d.zarr"
    )


def _write_observation_day_store(store_path, basis_version: str) -> None:
    source = _observation_source()
    day_store = source.assign(
        {
            Dimension.TIME.key(): (
                "obs",
                source[Dimension.TIME.key()].values.astype("datetime64[ns]").astype("int64"),
            ),
            "obs_id": ("obs", numpy.array(["a", "b", "c", "d"], dtype="<U96")),
        }
    )
    day_store.attrs[observations.OBSERVATIONS_BASIS_VERSION_ATTRIBUTE] = basis_version
    day_store.to_zarr(store_path, mode="w")


def test_expected_observation_basis_version_opens_and_drops_extra_columns(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "20240103.zarr"
    _write_observation_day_store(store_path, observations.EXPECTED_OBSERVATIONS_BASIS_VERSION)
    monkeypatch.setattr(observations, "observation_path", lambda _day: str(store_path))
    first_day_datetimes = numpy.array(["2024-01-03"], dtype="datetime64[ns]")

    selected = observations._selected_observations_dataset(
        observation_days=numpy.array(["2024-01-03"], dtype="datetime64[D]"),
        first_day_timestamps=pandas.to_datetime(first_day_datetimes),
        first_day_datetimes=first_day_datetimes,
        lead_days_count=10,
    )

    assert "obs_id" not in selected.variables
    assert set(observations._consumed_observation_variable_keys()) <= set(selected.variables)


def test_unexpected_observation_basis_version_raises_with_found_version(tmp_path, monkeypatch) -> None:
    store_path = tmp_path / "20240103.zarr"
    _write_observation_day_store(store_path, "2024-v2.0.1")
    monkeypatch.setattr(observations, "observation_path", lambda _day: str(store_path))
    first_day_datetimes = numpy.array(["2024-01-03"], dtype="datetime64[ns]")

    with pytest.raises(observations.ObservationBasisVersionError) as raised_error:
        observations._selected_observations_dataset(
            observation_days=numpy.array(["2024-01-03"], dtype="datetime64[D]"),
            first_day_timestamps=pandas.to_datetime(first_day_datetimes),
            first_day_datetimes=first_day_datetimes,
            lead_days_count=10,
        )

    assert "2024-v2.0.1" in str(raised_error.value)
    assert observations.EXPECTED_OBSERVATIONS_BASIS_VERSION in str(raised_error.value)
