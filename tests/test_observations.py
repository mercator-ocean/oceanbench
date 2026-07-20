# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
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
        == "observations-v3-20240103-20250103-10d.zarr"
    )


def _challenger_dataset(first_day_datetimes: list[str], lead_days_count: int) -> xarray.Dataset:
    return xarray.Dataset(
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(first_day_datetimes, dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): range(lead_days_count),
        }
    )


def test_observations_use_overlap_when_challenger_starts_before_observations(monkeypatch) -> None:
    captured = {}

    def fake_selected_observations_dataset(**kwargs):
        captured.update(kwargs)
        return xarray.Dataset()

    monkeypatch.setattr(observations, "_should_stage_observations_locally", lambda: False)
    monkeypatch.setattr(observations, "_selected_observations_dataset", fake_selected_observations_dataset)

    observations.observations(
        _challenger_dataset(
            ["2023-12-27", "2024-01-03"],
            lead_days_count=10,
        )
    )

    assert captured["observation_days"][0] == numpy.datetime64("2024-01-01")
    assert captured["observation_days"][-1] == numpy.datetime64("2024-01-12")
    assert captured["first_day_timestamps"].min() == pandas.Timestamp("2023-12-27")


def test_observations_still_fail_when_no_forecast_window_overlaps_available_observations() -> None:
    try:
        observations.observations(
            _challenger_dataset(
                ["2023-12-01"],
                lead_days_count=7,
            )
        )
    except observations.ObservationDataUnavailableError as error:
        assert "forecast windows end on 2023-12-07" in str(error)
    else:
        raise AssertionError("Expected observation data without overlap to fail.")
