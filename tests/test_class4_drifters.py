# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
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


def _multi_depth_challenger_dataset(depths: list[float]) -> xarray.Dataset:
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    lead_day_key = Dimension.LEAD_DAY_INDEX.key()
    depth_key = Dimension.DEPTH.key()
    latitude_key = Dimension.LATITUDE.key()
    longitude_key = Dimension.LONGITUDE.key()
    dimensions = (first_day_key, lead_day_key, depth_key, latitude_key, longitude_key)
    depth_count = len(depths)
    eastward_velocities = numpy.broadcast_to(
        numpy.array([0.1 * (depth_index + 1) for depth_index in range(depth_count)]).reshape(1, 1, depth_count, 1, 1),
        (1, 3, depth_count, 2, 2),
    ).copy()
    northward_velocities = eastward_velocities * 2.0
    return xarray.Dataset(
        {
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(): (
                (first_day_key, lead_day_key, latitude_key, longitude_key),
                numpy.zeros((1, 3, 2, 2)),
            ),
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(): (dimensions, eastward_velocities),
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): (dimensions, northward_velocities),
        },
        coords={
            first_day_key: numpy.array(["2024-01-01"], dtype="datetime64[ns]"),
            lead_day_key: numpy.arange(3),
            depth_key: depths,
            latitude_key: [0.0, 1.0],
            longitude_key: [10.0, 11.0],
        },
    )


def _reference_trajectories() -> xarray.Dataset:
    return xarray.Dataset(
        {
            "lat": (("particle", "time"), numpy.array([[0.0, 0.0, 0.0]])),
            "lon": (("particle", "time"), numpy.array([[10.0, 10.0, 10.0]])),
        },
        coords={
            "particle": [0],
            "time": numpy.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]"),
            "lat0": ("particle", [0.0]),
            "lon0": ("particle", [10.0]),
        },
    )


def _advection_dataset_used(monkeypatch, challenger_dataset: xarray.Dataset) -> xarray.Dataset:
    captured_datasets = []

    def _capture(dataset, latitudes, longitudes):
        captured_datasets.append(dataset)
        return _reference_trajectories()

    monkeypatch.setattr(class4_drifters.lagrangian_trajectory, "_get_particle_dataset", _capture)
    class4_drifters.class4_drifter_challenger_trajectories(
        challenger_dataset=challenger_dataset,
        reference_trajectories=_reference_trajectories(),
    )
    return captured_datasets[0]


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


def test_class4_drifter_distance_wraps_longitude_at_dateline() -> None:
    reference_trajectories = xarray.Dataset(
        {
            "lat": (("particle", "time"), numpy.array([[0.0]])),
            "lon": (("particle", "time"), numpy.array([[179.0]])),
        },
        coords={"particle": [0], "time": numpy.array(["2024-01-01"], dtype="datetime64[ns]")},
    )
    challenger_trajectories = xarray.Dataset(
        {
            "lat": (("particle", "time"), numpy.array([[0.0]])),
            "lon": (("particle", "time"), numpy.array([[-179.0]])),
        },
        coords={"particle": [0], "time": numpy.array(["2024-01-01"], dtype="datetime64[ns]")},
    )

    distances = class4_drifters.class4_drifter_trajectory_distance_km(
        challenger_trajectories,
        reference_trajectories,
    )

    assert numpy.isclose(distances[0, 0], 222.39, atol=0.1)


def test_class4_drifter_score_reports_deviation_and_matched_counts(monkeypatch) -> None:
    reference_trajectories = xarray.Dataset(
        {
            "lat": (("particle", "time"), numpy.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])),
            "lon": (("particle", "time"), numpy.array([[10.0, 10.0, 10.0], [11.0, 11.0, 11.0]])),
        },
        coords={
            "particle": [0, 1],
            "time": numpy.array(["2024-01-01", "2024-01-02", "2024-01-03"], dtype="datetime64[ns]"),
            "lat0": ("particle", [0.0, 1.0]),
            "lon0": ("particle", [10.0, 11.0]),
        },
    )
    challenger_trajectories = reference_trajectories.copy(deep=True)
    challenger_trajectories["lat"] = challenger_trajectories["lat"] + xarray.DataArray(
        numpy.array([[0.0, 1.0, 2.0], [0.0, numpy.nan, numpy.nan]]),
        dims=("particle", "time"),
    )

    monkeypatch.setattr(
        class4_drifters,
        "class4_drifter_trajectory_comparison",
        lambda **_: (challenger_trajectories, reference_trajectories),
    )

    score = class4_drifters.deviation_of_lagrangian_trajectories_compared_to_class4_observations(
        _challenger_dataset(lead_day_count=3),
        _observation_dataset(),
    )

    assert score.index.tolist() == [
        "Class-4 drifter trajectory deviation mean (km)",
        "Class-4 matched drifter count",
    ]
    assert score.columns.tolist() == ["Lead day 1 (init)", "Lead day 2"]
    assert score.loc["Class-4 matched drifter count", "Lead day 1 (init)"] == 2.0
    assert score.loc["Class-4 matched drifter count", "Lead day 2"] == 1.0
    assert numpy.isnan(score.loc["Class-4 drifter trajectory deviation mean (km)", "Lead day 1 (init)"])
    assert numpy.isnan(score.loc["Class-4 drifter trajectory deviation mean (km)", "Lead day 2"])


def test_class4_drifter_score_masks_low_matched_count(monkeypatch) -> None:
    particle_count = 60
    reference_trajectories = xarray.Dataset(
        {
            "lat": (("particle", "time"), numpy.zeros((particle_count, 2))),
            "lon": (("particle", "time"), numpy.zeros((particle_count, 2))),
        },
        coords={
            "particle": numpy.arange(particle_count),
            "time": numpy.array(["2024-01-01", "2024-01-02"], dtype="datetime64[ns]"),
            "lat0": ("particle", numpy.zeros(particle_count)),
            "lon0": ("particle", numpy.zeros(particle_count)),
        },
    )
    challenger_trajectories = reference_trajectories.copy(deep=True)
    challenger_latitudes = challenger_trajectories["lat"].values
    challenger_latitudes[:49, 1] = 1.0
    challenger_latitudes[49:, 1] = numpy.nan
    challenger_trajectories["lat"] = (("particle", "time"), challenger_latitudes)

    monkeypatch.setattr(
        class4_drifters,
        "class4_drifter_trajectory_comparison",
        lambda **_: (challenger_trajectories, reference_trajectories),
    )

    score = class4_drifters.deviation_of_lagrangian_trajectories_compared_to_class4_observations(
        _challenger_dataset(lead_day_count=3),
        _observation_dataset(),
    )

    assert score.loc["Class-4 matched drifter count", "Lead day 2"] == 49.0
    assert numpy.isnan(score.loc["Class-4 drifter trajectory deviation mean (km)", "Lead day 2"])


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

    assert score.columns.tolist() == ["Lead day 1 (init)", "Lead day 2"]


def test_class4_drifter_advection_interpolates_currents_to_drifter_depth(monkeypatch) -> None:
    advection_dataset = _advection_dataset_used(monkeypatch, _multi_depth_challenger_dataset([5.0, 25.0]))

    assert Dimension.DEPTH.key() not in advection_dataset.dims
    assert Dimension.DEPTH.key() not in advection_dataset.coords
    assert numpy.allclose(advection_dataset[Variable.EASTWARD_SEA_WATER_VELOCITY.key()].values, 0.15)
    assert numpy.allclose(advection_dataset[Variable.NORTHWARD_SEA_WATER_VELOCITY.key()].values, 0.30)


def test_class4_drifter_advection_keeps_single_level_currents(monkeypatch) -> None:
    advection_dataset = _advection_dataset_used(monkeypatch, _multi_depth_challenger_dataset([0.5]))

    assert Dimension.DEPTH.key() not in advection_dataset.dims
    assert numpy.allclose(advection_dataset[Variable.EASTWARD_SEA_WATER_VELOCITY.key()].values, 0.1)


def _full_depth_axis_interpolation(dataset: xarray.Dataset) -> xarray.Dataset:
    depth_key = Dimension.DEPTH.key()
    current_keys = [
        Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
        Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
    ]
    return dataset[current_keys].interp({depth_key: class4_drifters.DRIFTER_DEPTH_METERS}).drop_vars(depth_key)


@pytest.mark.parametrize(
    "depths",
    [
        [0.5, 5.0, 15.0, 25.0, 50.0],
        [0.5, 5.0, 12.0, 20.0, 50.0],
        [50.0, 100.0, 200.0],
        [0.5, 2.0, 5.0],
    ],
    ids=["exact_level", "bracketed", "below_axis_range", "above_axis_range"],
)
def test_class4_drifter_depth_currents_match_full_axis_interpolation(depths: list[float]) -> None:
    dataset = _multi_depth_challenger_dataset(depths)

    bracketed = class4_drifters._drifter_depth_current_dataset(dataset)
    full_axis = _full_depth_axis_interpolation(dataset)

    for current_key in (
        Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
        Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
    ):
        bracketed_values = bracketed[current_key].values
        full_axis_values = full_axis[current_key].values
        assert bracketed_values.shape == full_axis_values.shape
        assert numpy.array_equal(bracketed_values, full_axis_values, equal_nan=True)


def test_class4_drifter_depth_currents_read_two_levels_only(monkeypatch) -> None:
    interpolated_depth_counts: list[int] = []
    original_interp = xarray.Dataset.interp

    def counting_interp(self, *arguments, **keyword_arguments):
        interpolated_depth_counts.append(self.sizes[Dimension.DEPTH.key()])
        return original_interp(self, *arguments, **keyword_arguments)

    monkeypatch.setattr(xarray.Dataset, "interp", counting_interp)

    class4_drifters._drifter_depth_current_dataset(_multi_depth_challenger_dataset([0.5, 5.0, 12.0, 20.0, 50.0, 100.0]))

    assert interpolated_depth_counts == [2]


@pytest.mark.parametrize(
    ("depths", "expected_indices"),
    [
        ([0.5, 5.0, 12.0, 20.0, 50.0], [2, 3]),
        ([0.5, 5.0, 15.0, 25.0], [2, 3]),
        ([50.0, 100.0, 200.0], [0, 1]),
        ([0.5, 2.0, 5.0], [1, 2]),
        ([50.0, 20.0, 12.0, 5.0, 0.5], [1, 2]),
    ],
    ids=["bracketed", "exact_level", "below_axis_range", "above_axis_range", "descending"],
)
def test_bracketing_depth_indices(depths: list[float], expected_indices: list[int]) -> None:
    indices = class4_drifters._bracketing_depth_indices(numpy.array(depths), class4_drifters.DRIFTER_DEPTH_METERS)

    assert indices == expected_indices


def test_class4_drifter_advection_keeps_depthless_currents(monkeypatch) -> None:
    depthless_dataset = _multi_depth_challenger_dataset([0.5]).isel({Dimension.DEPTH.key(): 0}, drop=True)

    advection_dataset = _advection_dataset_used(monkeypatch, depthless_dataset)

    assert Dimension.DEPTH.key() not in advection_dataset.dims
    assert numpy.allclose(advection_dataset[Variable.EASTWARD_SEA_WATER_VELOCITY.key()].values, 0.1)


def _hourly_observation_dataset(rows: list[tuple[str, float, float]]) -> xarray.Dataset:
    observation_dimension = "observation"
    first_day = numpy.datetime64("2024-01-01T00:00:00")
    return xarray.Dataset(
        {
            Dimension.TIME.key(): (
                observation_dimension,
                numpy.array([timestamp for timestamp, _, _ in rows], dtype="datetime64[ns]"),
            ),
            Dimension.FIRST_DAY_DATETIME.key(): (observation_dimension, numpy.repeat(first_day, len(rows))),
            Dimension.DEPTH.key(): (observation_dimension, numpy.full(len(rows), 15.0)),
            Dimension.LATITUDE.key(): (observation_dimension, [latitude for _, latitude, _ in rows]),
            Dimension.LONGITUDE.key(): (observation_dimension, [longitude for _, _, longitude in rows]),
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(): (observation_dimension, numpy.zeros(len(rows))),
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): (observation_dimension, numpy.zeros(len(rows))),
        }
    )


HOUR_00 = "2024-01-01T00:00:00.000000000"
HOUR_02 = "2024-01-01T02:00:00.000000000"


def _linked_timestamps(linked_trajectories, track_id: int) -> list[str]:
    track_rows = linked_trajectories[linked_trajectories["track_id"] == track_id]
    return [str(timestamp) for timestamp in track_rows[Dimension.TIME.key()].to_numpy()]


def test_class4_drifter_reliability_threshold_never_drops_below_minimum() -> None:
    assert class4_drifters._minimum_reliable_matched_drifter_count(numpy.array([2.0, 1.0])) == 50
    assert class4_drifters._minimum_reliable_matched_drifter_count(numpy.array([2000.0, 1500.0])) == 100


def test_class4_drifter_linking_survives_a_frame_without_matches() -> None:
    linked_trajectories = class4_drifters._link_hourly_drifter_trajectories(
        _hourly_observation_dataset(
            [
                ("2024-01-01T00:00:00", 0.0, 10.0),
                ("2024-01-01T00:00:00", 1.0, 11.0),
                ("2024-01-01T01:00:00", 5.0, 20.0),
                ("2024-01-01T01:00:00", 6.0, 21.0),
                ("2024-01-01T02:00:00", 0.0, 10.0),
                ("2024-01-01T02:00:00", 1.0, 11.0),
            ]
        ),
        numpy.datetime64("2024-01-01T00:00:00"),
    )

    assert _linked_timestamps(linked_trajectories, 0) == [HOUR_00, HOUR_02]
    assert _linked_timestamps(linked_trajectories, 1) == [HOUR_00, HOUR_02]


def test_class4_drifter_linking_re_matches_a_track_after_a_missing_hour() -> None:
    linked_trajectories = class4_drifters._link_hourly_drifter_trajectories(
        _hourly_observation_dataset(
            [
                ("2024-01-01T00:00:00", 0.0, 10.0),
                ("2024-01-01T00:00:00", 1.0, 11.0),
                ("2024-01-01T01:00:00", 0.0, 10.0),
                ("2024-01-01T02:00:00", 0.0, 10.0),
                ("2024-01-01T02:00:00", 1.0, 11.0),
            ]
        ),
        numpy.datetime64("2024-01-01T00:00:00"),
    )

    assert _linked_timestamps(linked_trajectories, 1) == [HOUR_00, HOUR_02]


def test_class4_drifter_linking_drops_a_track_past_the_maximum_gap() -> None:
    present_track_rows = [(f"2024-01-01T{hour:02d}:00:00", 0.0, 10.0) for hour in range(9)]
    gapped_track_rows = [("2024-01-01T00:00:00", 1.0, 11.0), ("2024-01-01T08:00:00", 1.0, 11.0)]

    linked_trajectories = class4_drifters._link_hourly_drifter_trajectories(
        _hourly_observation_dataset(present_track_rows + gapped_track_rows),
        numpy.datetime64("2024-01-01T00:00:00"),
    )

    assert len(_linked_timestamps(linked_trajectories, 0)) == 9
    assert _linked_timestamps(linked_trajectories, 1) == [HOUR_00]


def test_class4_drifter_linking_excludes_late_arriving_drifters() -> None:
    linked_trajectories = class4_drifters._link_hourly_drifter_trajectories(
        _hourly_observation_dataset(
            [
                ("2024-01-01T00:00:00", 0.0, 10.0),
                ("2024-01-01T01:00:00", 0.0, 10.0),
                ("2024-01-01T01:00:00", 40.0, 40.0),
                ("2024-01-01T02:00:00", 0.0, 10.0),
                ("2024-01-01T02:00:00", 40.0, 40.0),
            ]
        ),
        numpy.datetime64("2024-01-01T00:00:00"),
    )

    assert linked_trajectories["track_id"].unique().tolist() == [0]
