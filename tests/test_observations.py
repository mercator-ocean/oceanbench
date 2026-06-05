# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.classIV import _create_observations_dataframe
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.references import observations


def _observation_source() -> xarray.Dataset:
    observation_dimension = "obs"
    variables = {
        Dimension.TIME.key(): (
            observation_dimension,
            pandas.to_datetime(["2024-01-04", "2024-01-11", "2024-01-14"]).values,
        ),
        Dimension.LATITUDE.key(): (observation_dimension, [0.0, 1.0, 2.0]),
        Dimension.LONGITUDE.key(): (observation_dimension, [10.0, 11.0, 12.0]),
        Dimension.DEPTH.key(): (observation_dimension, [0.0, 0.0, 0.0]),
    }
    for variable in (
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
        Variable.SEA_WATER_SALINITY,
        Variable.EASTWARD_SEA_WATER_VELOCITY,
        Variable.NORTHWARD_SEA_WATER_VELOCITY,
    ):
        variables[variable.key()] = (observation_dimension, [1.0, 2.0, 3.0])
    return xarray.Dataset(variables)


def test_selected_observations_dataset_preserves_overlapping_forecast_windows(monkeypatch) -> None:
    source = _observation_source()
    first_day_datetimes = numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]")

    monkeypatch.setattr(observations, "open_mfdataset", lambda *_, **__: source)
    monkeypatch.setattr(observations, "require_remote_dataset_dimensions", lambda dataset, *_: dataset)

    selected = observations._selected_observations_dataset(
        observation_days=numpy.array(["2024-01-04", "2024-01-11", "2024-01-14"], dtype="datetime64[D]"),
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
        {"time": "2024-01-04", "first_day": "2024-01-03", "value": 1.0},
        {"time": "2024-01-11", "first_day": "2024-01-03", "value": 2.0},
        {"time": "2024-01-11", "first_day": "2024-01-10", "value": 2.0},
        {"time": "2024-01-14", "first_day": "2024-01-10", "value": 3.0},
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
        {"observation_value": 2.0, "lead_day": 0},
        {"observation_value": 3.0, "lead_day": 3},
    ]


def test_observations_stage_path_uses_overlap_safe_version() -> None:
    assert (
        observations._observations_stage_path("2024-01-03", "2025-01-04", 10).name
        == "observations-v2-20240103-20250104-10d.zarr"
    )


def test_observation_path_uses_configurable_template_and_file_urls(monkeypatch) -> None:
    monkeypatch.setenv(
        OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_OBSERVATION_ZARR_TEMPLATE.value,
        "file:///tmp/synthetic_class4/{compact_date}.zarr",
    )

    assert observations.observation_path(numpy.datetime64("2026-05-23")) == "/tmp/synthetic_class4/20260523.zarr"
    assert (
        observations.observation_path(
            numpy.datetime64("2026-05-23"),
            "https://example.invalid/observations/{date}.zarr",
        )
        == "https://example.invalid/observations/2026-05-23.zarr"
    )


def test_observations_uses_configured_last_available_day(monkeypatch) -> None:
    source = _observation_source()
    opened_paths = []
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    challenger_dataset = xarray.Dataset(
        coords={
            first_day_key: numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
            lead_day_key: [0, 1, 2],
        }
    )

    def open_mfdataset(paths, *_, **__):
        opened_paths.extend(paths)
        return source

    monkeypatch.setenv(
        OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_OBSERVATION_ZARR_TEMPLATE.value,
        "file:///tmp/synthetic_class4/{day}.zarr",
    )
    monkeypatch.setenv(
        OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_OBSERVATION_LAST_DAY.value,
        "2024-01-04",
    )
    monkeypatch.setattr(observations, "open_mfdataset", open_mfdataset)
    monkeypatch.setattr(observations, "require_remote_dataset_dimensions", lambda dataset, *_: dataset)

    selected = observations.observations(challenger_dataset)

    assert opened_paths == [
        "/tmp/synthetic_class4/20240104.zarr",
    ]
    assert pandas.to_datetime(selected[Dimension.TIME.key()].values).strftime("%Y-%m-%d").tolist() == ["2024-01-04"]
