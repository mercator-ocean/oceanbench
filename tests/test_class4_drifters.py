# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core import class4_drifters
from oceanbench.core.dataset_utils import Dimension, Variable


def _challenger_dataset(first_day_count: int = 1, lead_day_count: int = 3) -> xarray.Dataset:
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    depth_key = Dimension.DEPTH.key()
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    dimensions = (first_day_key, lead_day_key, depth_key, latitude_key, longitude_key)
    return xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                (first_day_key, lead_day_key, latitude_key, longitude_key),
                numpy.zeros((first_day_count, lead_day_count, 2, 2)),
            ),
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(): (
                dimensions,
                numpy.zeros((first_day_count, lead_day_count, 1, 2, 2)),
            ),
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): (
                dimensions,
                numpy.zeros((first_day_count, lead_day_count, 1, 2, 2)),
            ),
        },
        coords={
            first_day_key: numpy.array(["2024-01-01"], dtype="datetime64[ns]")[:first_day_count],
            lead_day_key: numpy.arange(lead_day_count),
            depth_key: [0.5],
            latitude_key: [0.0, 1.0],
            longitude_key: [10.0, 11.0],
        },
    )


def _observation_dataset() -> xarray.Dataset:
    observation_dimension = "observation"
    first_day = numpy.datetime64("2024-01-01T00:00:00")
    times = numpy.array(
        [
            "2024-01-01T00:00:00",
            "2024-01-01T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-02T00:00:00",
            "2024-01-03T00:00:00",
            "2024-01-03T00:00:00",
        ],
        dtype="datetime64[ns]",
    )
    return xarray.Dataset(
        {
            Dimension.TIME.key(): (observation_dimension, times),
            Dimension.FIRST_DAY_DATETIME.key(): (observation_dimension, numpy.repeat(first_day, len(times))),
            Dimension.DEPTH.key(): (observation_dimension, numpy.full(len(times), 15.0)),
            Dimension.LATITUDE.key(): (observation_dimension, [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]),
            Dimension.LONGITUDE.key(): (observation_dimension, [10.0, 11.0, 10.0, 11.0, 10.0, 11.0]),
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(): (observation_dimension, numpy.zeros(len(times))),
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): (observation_dimension, numpy.zeros(len(times))),
        }
    )


def test_class4_drifter_reference_trajectories_reconstruct_daily_tracks() -> None:
    trajectories = class4_drifters.class4_drifter_reference_trajectories(
        _challenger_dataset(),
        _observation_dataset(),
    )

    assert trajectories.sizes == {"particle": 2, "time": 3}
    assert trajectories["lat"].values.tolist() == [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]
    assert trajectories["lon0"].values.tolist() == [10.0, 11.0]


def test_class4_drifter_score_reports_deviation_and_matched_counts(monkeypatch) -> None:
    reference_trajectories = xarray.Dataset(
        {
            "lat": (("particle", "time"), numpy.array([[0.0, 0.0], [1.0, 1.0]])),
            "lon": (("particle", "time"), numpy.array([[10.0, 10.0], [11.0, 11.0]])),
        },
        coords={
            "particle": [0, 1],
            "time": numpy.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "lat0": ("particle", [0.0, 1.0]),
            "lon0": ("particle", [10.0, 11.0]),
        },
    )
    challenger_trajectories = reference_trajectories.copy(deep=True)
    challenger_trajectories["lat"] = challenger_trajectories["lat"] + xarray.DataArray(
        numpy.array([[0.0, 1.0], [0.0, numpy.nan]]),
        dims=("particle", "time"),
    )

    monkeypatch.setattr(
        class4_drifters,
        "class4_drifter_trajectory_comparison",
        lambda **_: (challenger_trajectories, reference_trajectories),
    )

    score = class4_drifters.deviation_of_lagrangian_trajectories_compared_to_class4_observations(
        _challenger_dataset(lead_day_count=2),
        _observation_dataset(),
    )

    assert score.index.tolist() == [
        "Class-4 drifter trajectory deviation mean (km)",
        "Class-4 matched drifter count",
    ]
    assert score.columns.tolist() == ["Lead day 1", "Lead day 2"]
    assert score.loc["Class-4 drifter trajectory deviation mean (km)", "Lead day 1"] == 0.0
    assert score.loc["Class-4 matched drifter count", "Lead day 1"] == 2.0
    assert score.loc["Class-4 matched drifter count", "Lead day 2"] == 1.0


def test_class4_drifter_score_uses_available_trajectory_lead_days(monkeypatch) -> None:
    reference_trajectories = xarray.Dataset(
        {
            "lat": (("particle", "time"), numpy.array([[0.0, 0.0]])),
            "lon": (("particle", "time"), numpy.array([[10.0, 10.0]])),
        },
        coords={
            "particle": [0],
            "time": numpy.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "lat0": ("particle", [0.0]),
            "lon0": ("particle", [10.0]),
        },
    )
    challenger_trajectories = reference_trajectories.copy(deep=True)

    monkeypatch.setattr(
        class4_drifters,
        "class4_drifter_trajectory_comparison",
        lambda **_: (challenger_trajectories, reference_trajectories),
    )

    score = class4_drifters.deviation_of_lagrangian_trajectories_compared_to_class4_observations(
        _challenger_dataset(lead_day_count=3),
        _observation_dataset(),
    )

    assert score.columns.tolist() == ["Lead day 1", "Lead day 2"]
