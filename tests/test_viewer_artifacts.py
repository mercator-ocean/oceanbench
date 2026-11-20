# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import subprocess
import sys

import numpy
import pandas
import pyarrow.parquet
import pytest
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.regions import custom_region, padded_region, subset_dataset_to_region
from oceanbench.publish import viewer_artifacts
from tests import synthetic_class4
from tests.test_classiv_matchups import _class4_table, _matchups

_SEA_SURFACE_HEIGHT_KEY = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
_TEMPERATURE_KEY = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
_SALINITY_KEY = Variable.SEA_WATER_SALINITY.key()


def _frame(blocks, start_dates, lead_days, rows_per_block, seed=3, bias=0.0) -> pandas.DataFrame:
    generator = numpy.random.default_rng(seed)
    frames = []
    for start_date in start_dates:
        for lead_day in lead_days:
            for variable, depth_bin in blocks:
                observation_value = generator.normal(0.0, 1.0, rows_per_block)
                frames.append(
                    pandas.DataFrame(
                        {
                            "variable": variable,
                            "depth_bin": depth_bin,
                            "lead_day": lead_day,
                            "first_day": pandas.Timestamp(start_date),
                            "latitude": generator.uniform(-60, 60, rows_per_block),
                            "longitude": generator.uniform(-180, 180, rows_per_block),
                            "observation_value": observation_value,
                            "model_value": observation_value + generator.normal(bias, 0.2, rows_per_block),
                        }
                    )
                )
    return pandas.concat(frames, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _row_group_statistics(path) -> list[dict]:
    metadata = pyarrow.parquet.ParquetFile(path).metadata
    names = viewer_artifacts.MATCHUP_SCHEMA.names
    return [
        {
            name: metadata.row_group(index).column(names.index(name)).statistics
            for name in ("start_date", "lead_day", "variable", "depth_bin")
        }
        | {"rows": metadata.row_group(index).num_rows}
        for index in range(metadata.num_row_groups)
    ]


def test_matchup_parquet_packs_one_start_and_lead_day_per_row_group(tmp_path) -> None:
    path = tmp_path / "class4-matchups.parquet"
    blocks = ((_SEA_SURFACE_HEIGHT_KEY, "surface"), (_TEMPERATURE_KEY, "0-5m"))
    viewer_artifacts.write_matchup_parquet(_frame(blocks, ["2024-01-10", "2024-01-03"], [4, 0], 50), path)

    groups = _row_group_statistics(path)
    assert len(groups) == 4
    for group in groups:
        assert group["start_date"].min == group["start_date"].max
        assert group["lead_day"].min == group["lead_day"].max
        assert group["rows"] == len(blocks) * 50
    pairs = [(group["start_date"].min, group["lead_day"].min) for group in groups]
    assert pairs == [("2024-01-03", 1), ("2024-01-03", 5), ("2024-01-10", 1), ("2024-01-10", 5)]
    metadata = pyarrow.parquet.ParquetFile(path).metadata
    assert {metadata.row_group(0).column(index).compression for index in range(metadata.num_columns)} == {"ZSTD"}
    for name in ("latitude", "longitude", "observation_value", "model_value"):
        assert metadata.row_group(0).column(viewer_artifacts.MATCHUP_SCHEMA.names.index(name)).statistics is None
    assert pyarrow.parquet.read_schema(path).equals(viewer_artifacts.MATCHUP_SCHEMA)


def test_matchup_parquet_splits_a_large_pair_across_row_groups(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(viewer_artifacts, "MAXIMUM_ROW_GROUP_ROWS", 3)
    path = tmp_path / "class4-matchups.parquet"
    viewer_artifacts.write_matchup_parquet(_frame([(_SEA_SURFACE_HEIGHT_KEY, "surface")], ["2024-01-03"], [0], 7), path)

    assert [group["rows"] for group in _row_group_statistics(path)] == [3, 3, 1]


def test_verify_rejects_a_row_group_mixing_two_lead_days(tmp_path) -> None:
    path = tmp_path / "mixed.parquet"
    table = viewer_artifacts._serving_matchups(
        _frame([(_SEA_SURFACE_HEIGHT_KEY, "surface")], ["2024-01-03"], [0, 1], 4)
    )
    pyarrow.parquet.write_table(table, path, row_group_size=1_000)

    with pytest.raises(ValueError, match="mixes several"):
        viewer_artifacts.verify_matchup_parquet(path)


def test_matchup_parquet_pooled_scores_equal_the_class4_score_rows(tmp_path) -> None:
    matchups = _matchups()
    path = tmp_path / "class4-matchups.parquet"
    viewer_artifacts.write_matchup_parquet(matchups, path)

    served = pyarrow.parquet.read_table(path).to_pandas()
    assert len(served) == len(matchups)
    table = _class4_table()
    labels = lead_day_labels(1, synthetic_class4.LEAD_DAYS_COUNT)
    first_lead_day = served["lead_day"].min()
    for (variable, depth_bin), block in served.groupby(["variable", "depth_bin"]):
        row = f"[{variable}]{{{depth_bin}}}"
        label = next(label for label in table.index if row in label)
        for lead_day, lead_block in block.groupby("lead_day"):
            scored = lead_block.dropna(subset=["model_value"])
            error = scored["model_value"].astype(float) - scored["observation_value"].astype(float)
            # The served values are float32, about 4e-6 apart around a salinity of 35.
            numpy.testing.assert_allclose(
                numpy.sqrt((error**2).mean()), table.at[label, labels[lead_day - 1]], rtol=1e-5, atol=1e-5
            )
        first = block[block["lead_day"] == first_lead_day]
        if len(first):
            assert table.at[label, "Observations"] == len(first)
            assert table.at[label, "Missing"] == first["model_value"].isna().sum()


def test_year_artifacts_keep_each_variable_of_a_shared_row_group_apart() -> None:
    matchups = pandas.DataFrame(
        {
            "variable": [_SEA_SURFACE_HEIGHT_KEY] * 2 + [_TEMPERATURE_KEY] * 2,
            "depth_bin": ["surface"] * 4,
            "lead_day": [0] * 4,
            "first_day": [pandas.Timestamp("2024-01-03")] * 4,
            "latitude": [1.0, 1.0, 41.0, 41.0],
            "longitude": [1.0, 1.0, 1.0, 1.0],
            "observation_value": [10.0, 10.0, 20.0, 20.0],
            "model_value": [12.0, 14.0, 25.0, 27.0],
        }
    )
    grid = viewer_artifacts.YEAR_GRIDS["global"]

    geography = viewer_artifacts.year_error_geography(matchups, grid, 1)
    for short, expected in (("SSH", 3.0), ("T", 6.0)):
        assert [value for value in geography["variables"][short]["leads"]["1"] if value is not None] == [expected]
        assert [value for value in geography["variables"][short]["bias"]["1"] if value is not None] == [expected]
    rows_by_start = viewer_artifacts.class4_rmsd_rows_by_start(matchups)
    by_start = viewer_artifacts.year_rmsd_by_start(matchups, rows_by_start, grid, 1)
    assert by_start["variables"]["SSH"]["leads"]["1"]["dates"] == ["2024-01-03"]
    assert by_start["variables"]["SSH"]["leads"]["1"]["bias"] == [3.0]
    assert by_start["variables"]["T"]["depth_bin"] == "surface"


def _year_matchups() -> pandas.DataFrame:
    return _frame([(_SEA_SURFACE_HEIGHT_KEY, "surface")], ["2024-01-03", "2024-01-10"], [0], 200, seed=7, bias=0.5)


def _year_rmsd_by_start(matchups: pandas.DataFrame) -> dict:
    rows_by_start = viewer_artifacts.class4_rmsd_rows_by_start(matchups)
    return viewer_artifacts.year_rmsd_by_start(matchups, rows_by_start, viewer_artifacts.YEAR_GRIDS["global"], 1)


def test_year_rmsd_by_start_is_the_pooled_reduction_with_bracketing_intervals() -> None:
    matchups = _year_matchups()
    by_start = _year_rmsd_by_start(matchups)

    series = by_start["variables"]["SSH"]["leads"]["1"]
    assert series["dates"] == ["2024-01-03", "2024-01-10"]
    for index, start_date in enumerate(series["dates"]):
        start = matchups[matchups["first_day"] == pandas.Timestamp(start_date)]
        error = (start["model_value"] - start["observation_value"]).to_numpy()
        assert series["rmsd"][index] == pytest.approx(numpy.sqrt((error * error).mean()), abs=1e-6)
        assert series["bias"][index] == pytest.approx(error.mean(), abs=1e-6)
        assert series["rmsd_centred"][index] == pytest.approx(
            numpy.sqrt((error * error).mean() - error.mean() ** 2), abs=1e-6
        )
        assert series["n"][index] == len(start)
        assert series["rmsd_ci_low"][index] <= series["rmsd"][index] <= series["rmsd_ci_high"][index]
        assert series["bias_ci_low"][index] <= series["bias"][index] <= series["bias_ci_high"][index]
    assert _year_rmsd_by_start(matchups) == by_start


def test_an_interval_does_not_change_when_another_start_date_is_added() -> None:
    matchups = _year_matchups()
    first_start = matchups[matchups["first_day"] == pandas.Timestamp("2024-01-10")]
    alone = _year_rmsd_by_start(first_start)["variables"]["SSH"]["leads"]["1"]
    together = _year_rmsd_by_start(matchups)["variables"]["SSH"]["leads"]["1"]

    index = together["dates"].index("2024-01-10")
    assert alone["dates"] == ["2024-01-10"]
    for name in ("rmsd_ci_low", "rmsd_ci_high", "bias_ci_low", "bias_ci_high"):
        assert alone[name] == [together[name][index]]
        assert alone[name][0] is not None


def _correlated_block_errors(generator, block_count: int, block_length: int) -> tuple[numpy.ndarray, numpy.ndarray]:
    block = numpy.repeat(numpy.arange(block_count), block_length)
    signed_error = numpy.repeat(generator.normal(0.0, 1.0, block_count), block_length) + generator.normal(
        0.0, 0.3, block.size
    )
    return block, signed_error


def test_block_bootstrap_covers_the_true_bias_where_the_independent_interval_is_too_narrow() -> None:
    generator = numpy.random.default_rng(11)
    block_covers, independent_covers = 0, 0
    repetitions = 200
    for _ in range(repetitions):
        block, signed_error = _correlated_block_errors(generator, block_count=40, block_length=50)
        _, (bias_low, bias_high) = viewer_artifacts._block_bootstrap_intervals(block, signed_error, generator)
        half_width = 1.96 * signed_error.std() / numpy.sqrt(signed_error.size)
        block_covers += bias_low <= 0.0 <= bias_high
        independent_covers += abs(signed_error.mean()) <= half_width
    assert 0.85 <= block_covers / repetitions <= 0.99
    assert independent_covers / repetitions < 0.5


def test_block_bootstrap_with_one_match_up_per_block_matches_the_analytic_interval() -> None:
    signed_error = numpy.random.default_rng(5).normal(0.3, 1.0, size=5_000)
    block = numpy.arange(signed_error.size)

    rmsd_interval, bias_interval = viewer_artifacts._block_bootstrap_intervals(
        block, signed_error, numpy.random.default_rng(1)
    )

    bias_half_width = 1.96 * signed_error.std() / numpy.sqrt(signed_error.size)
    assert (bias_interval[1] - bias_interval[0]) / 2.0 == pytest.approx(bias_half_width, rel=0.10)
    assert numpy.mean(bias_interval) == pytest.approx(signed_error.mean(), abs=0.2 * bias_half_width)
    squared_error = signed_error * signed_error
    mean = squared_error.mean()
    standard_error = squared_error.std() / numpy.sqrt(squared_error.size)
    rmsd_half_width = (numpy.sqrt(mean + 1.96 * standard_error) - numpy.sqrt(mean - 1.96 * standard_error)) / 2.0
    assert (rmsd_interval[1] - rmsd_interval[0]) / 2.0 == pytest.approx(rmsd_half_width, rel=0.10)


def test_block_bootstrap_needs_ten_blocks_and_is_deterministic_with_the_seed() -> None:
    block, signed_error = _correlated_block_errors(numpy.random.default_rng(2), block_count=9, block_length=100)
    assert viewer_artifacts._block_bootstrap_intervals(block, signed_error, numpy.random.default_rng(0)) == (
        None,
        None,
    )

    block, signed_error = _correlated_block_errors(numpy.random.default_rng(2), block_count=10, block_length=100)
    first = viewer_artifacts._block_bootstrap_intervals(block, signed_error, numpy.random.default_rng(3))
    second = viewer_artifacts._block_bootstrap_intervals(block, signed_error, numpy.random.default_rng(3))
    assert first == second
    assert all(interval is not None for interval in first)


def test_year_rmsd_by_start_blocks_follow_the_block_size_of_the_grid() -> None:
    blocks = viewer_artifacts._bootstrap_blocks(
        numpy.array([0.5, 4.5, 5.5, 0.5, 0.5]), numpy.array([0.5, 4.5, 0.5, 179.9, -180.1]), 5
    )
    assert blocks[0] == blocks[1] != blocks[2]
    assert blocks[3] == blocks[4]
    assert {grid["block_size_degrees"] for grid in viewer_artifacts.YEAR_GRIDS.values()} == {5, 2}


def test_year_error_geography_shares_its_counts_with_the_bias_standard_error() -> None:
    geography = viewer_artifacts.year_error_geography(_year_matchups(), viewer_artifacts.YEAR_GRIDS["global"], 1)

    sea_surface_height = geography["variables"]["SSH"]
    grid = geography["grid"]
    assert len(sea_surface_height["n"]["1"]) == grid["nlat"] * grid["nlon"]
    assert sum(sea_surface_height["n"]["1"]) == 400
    for count, start_count, bias in zip(
        sea_surface_height["n"]["1"], sea_surface_height["n_starts"]["1"], sea_surface_height["bias"]["1"]
    ):
        assert (bias is None) == (count == 0)
        assert start_count <= min(count, 2)
    assert set(sea_surface_height["bias_se"]["1"]) == {None}


def test_year_error_geography_bias_standard_error_is_taken_across_starts() -> None:
    start_dates = ["2024-01-03", "2024-01-10", "2024-01-17", "2024-01-24"]
    matchups = _frame([(_SEA_SURFACE_HEIGHT_KEY, "surface")], start_dates, [0], 30, seed=13)
    matchups = matchups.assign(latitude=10.5, longitude=20.5)

    geography = viewer_artifacts.year_error_geography(matchups, viewer_artifacts.YEAR_GRIDS["global"], 1)

    sea_surface_height = geography["variables"]["SSH"]
    cell = sea_surface_height["n"]["1"].index(120)
    error = matchups["model_value"] - matchups["observation_value"]
    start_means = error.groupby(matchups["first_day"]).mean().to_numpy()
    assert sea_surface_height["n_starts"]["1"][cell] == 4
    assert sea_surface_height["bias_se"]["1"][cell] == pytest.approx(start_means.std(ddof=1) / 2.0, abs=1e-4)
    assert sea_surface_height["bias"]["1"][cell] == pytest.approx(error.mean(), abs=1e-4)


def test_rmsd_by_depth_pools_over_all_starts_and_skips_surface_only_variables() -> None:
    blocks = (
        (_TEMPERATURE_KEY, "0-5m"),
        (_TEMPERATURE_KEY, "100-300m"),
        (_SALINITY_KEY, "0-5m"),
        (_SALINITY_KEY, "100-300m"),
        (_SEA_SURFACE_HEIGHT_KEY, "surface"),
    )
    matchups = _frame(blocks, ["2024-01-03", "2024-01-10"], [0, 4], 6, seed=19, bias=0.3)

    variables = viewer_artifacts.rmsd_by_depth(matchups)

    assert set(variables) == {_TEMPERATURE_KEY, _SALINITY_KEY}
    temperature = variables[_TEMPERATURE_KEY]
    assert temperature["depth_bins"] == ["0-5m", "100-300m"]
    assert temperature["leads"] == [1, 5]
    for depth_index, depth_bin in enumerate(temperature["depth_bins"]):
        for lead_index, lead_day in enumerate((0, 4)):
            block = matchups[
                (matchups["variable"] == _TEMPERATURE_KEY)
                & (matchups["depth_bin"] == depth_bin)
                & (matchups["lead_day"] == lead_day)
            ]
            error = (block["model_value"] - block["observation_value"]).to_numpy()
            assert temperature["rmsd"][depth_index][lead_index] == pytest.approx(numpy.sqrt((error * error).mean()))
            assert temperature["bias"][depth_index][lead_index] == pytest.approx(error.mean())
            assert temperature["n"][depth_index][lead_index] == len(block)
    assert viewer_artifacts.rmsd_by_depth(_year_matchups()) == {}


_KILOMETRES_PER_DEGREE = 111.195
_PLANTED_RADIUS_KM = 60.0


def _gaussian_eddy(grid_latitude, grid_longitude, latitude, longitude, amplitude) -> numpy.ndarray:
    longitude_offset = (grid_longitude - longitude + 180.0) % 360.0 - 180.0
    eastward_km = longitude_offset * _KILOMETRES_PER_DEGREE * numpy.cos(numpy.deg2rad(latitude))
    northward_km = (grid_latitude - latitude) * _KILOMETRES_PER_DEGREE
    return amplitude * numpy.exp(-(eastward_km**2 + northward_km**2) / (2 * _PLANTED_RADIUS_KM**2))


def _sea_surface_height_dataset(start_dates=("2024-01-03",), cell_size_degrees: float = 0.25) -> xarray.Dataset:
    latitudes = numpy.arange(20.0, 40.0 + cell_size_degrees / 2, cell_size_degrees)
    longitudes = numpy.arange(-10.0, 10.0 + cell_size_degrees / 2, cell_size_degrees)
    grid_latitude, grid_longitude = numpy.meshgrid(latitudes, longitudes, indexing="ij")
    cyclone = _gaussian_eddy(grid_latitude, grid_longitude, 33.0, -5.0, -0.2)
    # Each start gets its own displaced anticyclone, so a census that ignored the start is detectable.
    values = numpy.stack(
        [
            numpy.stack([cyclone + _gaussian_eddy(grid_latitude, grid_longitude, 30.0, 2.0 * start_index, 0.2)] * 2)
            for start_index in range(len(start_dates))
        ]
    )
    return xarray.Dataset(
        {
            _SEA_SURFACE_HEIGHT_KEY: (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                values,
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(list(start_dates), dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(2),
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


@pytest.fixture
def py_eddy_tracker():
    return pytest.importorskip("py_eddy_tracker")


def test_eddy_census_writes_every_start_and_lead_day_with_distinct_frames(tmp_path, py_eddy_tracker) -> None:
    start_dates = ("2024-01-03", "2024-01-10", "2024-01-17")
    viewer_artifacts.write_eddy_census(
        _sea_surface_height_dataset(start_dates), tmp_path / "eddies.json", "your_model", None
    )

    index = json.loads((tmp_path / "eddies.json").read_text(encoding="utf-8"))
    assert index["kind"] == "eddy-census"
    assert index["schema_version"] == "4"
    assert index["method"] == "py-eddy-tracker"
    assert index["parameters"]["contour_step_meters"] == 0.002
    assert index["parameters"]["py_eddy_tracker_version"]
    assert index["detection_grid"]["block_average_factor"] == [1, 1]
    assert index["dataset"] == "your_model"
    assert [entry["start_date"] for entry in index["starts"]] == list(start_dates)
    assert index["leads"] == index["starts"][0]["leads"]
    centres_by_start = set()
    for start_entry in index["starts"]:
        assert [entry["lead_day"] for entry in start_entry["leads"]] == [1, 2]
        for lead_entry in start_entry["leads"]:
            assert lead_entry["file"] == f"eddies-start-{start_entry['start_date']}-lead-{lead_entry['lead_day']}.json"
            payload = json.loads((tmp_path / lead_entry["file"]).read_text(encoding="utf-8"))
            payload["frame"] = viewer_artifacts.decoded_eddy_frame(payload["frame"])
            assert (payload["start_date"], payload["frame"]["lead_day"]) == (
                start_entry["start_date"],
                lead_entry["lead_day"],
            )
            assert sorted(eddy["polarity"] for eddy in payload["frame"]["detections"]) == ["anticyclone", "cyclone"]
            for eddy in payload["frame"]["detections"]:
                assert len(eddy["contour_latitude"]) == len(eddy["contour_longitude"]) >= 3
                assert eddy["amplitude"] == pytest.approx(0.2, rel=0.2)
                assert eddy["speed_radius_km"] == pytest.approx(_PLANTED_RADIUS_KM, rel=0.15)
                assert eddy["effective_radius_km"] > eddy["speed_radius_km"]
        first_frame = viewer_artifacts.decoded_eddy_frame(
            json.loads((tmp_path / start_entry["leads"][0]["file"]).read_text(encoding="utf-8"))["frame"]
        )
        centres_by_start.add(tuple(sorted((eddy["latitude"], eddy["longitude"]) for eddy in first_frame["detections"])))
    assert len(centres_by_start) == len(start_dates)


def test_a_census_frame_round_trips_through_the_compact_encoding() -> None:
    frame = {
        "lead_day": 3,
        "detections": [
            {
                "id": 0,
                "latitude": -38.589,
                "longitude": 179.9,
                "polarity": "anticyclone",
                "amplitude": 0.0755,
                "effective_radius_km": 41.2,
                "speed_radius_km": 30.5,
                "contour_latitude": [-38.2, -38.9, -39.123],
                "contour_longitude": [180.4, 179.5, 179.95],
            },
            {
                "id": 1,
                "latitude": 10.0,
                "longitude": -0.001,
                "polarity": "cyclone",
                "amplitude": 0.0,
                "effective_radius_km": 0.0,
                "speed_radius_km": 0.0,
                "contour_latitude": [10.5, 9.5, 10.0],
                "contour_longitude": [-0.5, -0.5, 0.6],
            },
        ],
    }
    encoded = viewer_artifacts.encoded_eddy_frame(frame)
    assert encoded["polarity"] == [1, 0]
    assert all(isinstance(contour, str) for contour in encoded["contour"])
    assert viewer_artifacts.decoded_eddy_frame(json.loads(json.dumps(encoded))) == frame


def test_contour_simplification_stays_within_its_tolerance_and_vertex_bound() -> None:
    angles = numpy.linspace(0.0, 2.0 * numpy.pi, 50)
    wobble = 1.0 + 0.1 * numpy.sin(7.0 * angles)
    latitudes = 45.0 + 0.4 * wobble * numpy.sin(angles)
    longitudes = 10.0 + 0.4 * wobble * numpy.cos(angles) / numpy.cos(numpy.radians(45.0))
    kept_latitudes, kept_longitudes = viewer_artifacts.simplified_ring(latitudes, longitudes)
    assert 3 <= kept_latitudes.size <= viewer_artifacts._MAXIMUM_CONTOUR_POINT_COUNT < latitudes.size - 1
    cosine = numpy.cos(numpy.radians(45.0))
    ring = numpy.column_stack([kept_longitudes * cosine, kept_latitudes])
    segments = numpy.stack([ring, numpy.roll(ring, -1, axis=0)], axis=1)
    for point in numpy.column_stack([longitudes * cosine, latitudes]):
        start, chord = segments[:, 0], segments[:, 1] - segments[:, 0]
        fraction = numpy.clip(((point - start) * chord).sum(axis=1) / (chord**2).sum(axis=1), 0.0, 1.0)
        distance = numpy.hypot(*(start + fraction[:, None] * chord - point).T).min()
        assert distance <= viewer_artifacts.CONTOUR_SIMPLIFICATION_TOLERANCE_DEGREES + 1e-9


def test_a_parallel_eddy_census_writes_the_files_of_the_serial_census(tmp_path, py_eddy_tracker) -> None:
    dataset = _sea_surface_height_dataset(("2024-01-03", "2024-01-10", "2024-01-17"))
    serial, parallel = tmp_path / "serial", tmp_path / "parallel"
    serial.mkdir()
    parallel.mkdir()
    viewer_artifacts.write_eddy_census(dataset, serial / "eddies.json", "your_model", None, max_workers=1)
    viewer_artifacts.write_eddy_census(dataset, parallel / "eddies.json", "your_model", None, max_workers=3)

    serial_files = sorted(path.name for path in serial.iterdir())
    assert len(serial_files) == 1 + 3 * 2
    assert sorted(path.name for path in parallel.iterdir()) == serial_files
    for name in serial_files:
        assert (parallel / name).read_bytes() == (serial / name).read_bytes(), name


def test_regional_eddy_census_keeps_only_the_eddies_centred_in_the_region(py_eddy_tracker) -> None:
    dataset = _sea_surface_height_dataset()
    everywhere = viewer_artifacts.eddy_census_frames(dataset, 0, None)[0]["detections"]
    assert len(everywhere) == 2

    def polarities(minimum_latitude: float, maximum_latitude: float) -> list:
        region = custom_region("band", "Band", minimum_latitude, maximum_latitude, -10.0, 10.0)
        padded = subset_dataset_to_region(dataset, padded_region(region, 10.0))
        return sorted(
            eddy["polarity"] for eddy in viewer_artifacts.eddy_census_frames(padded, 0, region)[0]["detections"]
        )

    assert polarities(25.0, 35.0) == ["anticyclone", "cyclone"]
    assert polarities(31.0, 35.0) == ["cyclone"]
    assert polarities(35.0, 40.0) == []


def _custom_region_artifacts(tmp_path, challenger_dataset: xarray.Dataset):
    region = custom_region("band", "Band", 22.0, 38.0, -8.0, 8.0)
    matchups = _frame([(_SEA_SURFACE_HEIGHT_KEY, "surface")], ["2024-01-03"], [0, 1], 20)
    rows_by_start = viewer_artifacts.class4_rmsd_rows_by_start(matchups)
    return viewer_artifacts.write_viewer_artifacts(
        tmp_path, "your_model", challenger_dataset, region, matchups, rows_by_start
    )


def test_viewer_artifacts_of_a_custom_region_have_no_year_files_and_no_maps(tmp_path, py_eddy_tracker) -> None:
    artifacts = _custom_region_artifacts(tmp_path, _sea_surface_height_dataset())

    assert artifacts.has_maps is False
    assert artifacts.insights == {
        "class4_matchups": "./data/insights/your_model/band/class4-matchups.parquet",
        "eddies": "./data/insights/your_model/band/eddies.json",
        "rmsd_by_depth": None,
        "year_error_geography": None,
        "year_rmsd_by_start": None,
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == ["insights"]


def test_no_eddy_census_is_written_on_a_grid_coarser_than_the_census_threshold(tmp_path) -> None:
    artifacts = _custom_region_artifacts(tmp_path, _sea_surface_height_dataset(cell_size_degrees=0.5))

    assert artifacts.insights["eddies"] is None
    assert artifacts.insights["class4_matchups"] is not None
    assert not list(tmp_path.rglob("eddies*.json"))


# Runs in a fresh interpreter where py-eddy-tracker cannot be imported, as in an install without it.
_WITHOUT_PY_EDDY_TRACKER_SCRIPT = """
import importlib.abc
import pathlib
import sys
import tempfile


class BlockPyEddyTracker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] == "py_eddy_tracker":
            raise ModuleNotFoundError(f"No module named {name!r}")
        return None


sys.meta_path.insert(0, BlockPyEddyTracker())

from tests.test_viewer_artifacts import _custom_region_artifacts, _sea_surface_height_dataset

# Longer than the stdout buffer of a pipe, so part of it is still buffered when the warning comes.
print("x" * 10_000)
directory = pathlib.Path(tempfile.mkdtemp())
artifacts = _custom_region_artifacts(directory, _sea_surface_height_dataset())
assert artifacts.insights["eddies"] is None, artifacts.insights
assert artifacts.insights["class4_matchups"] is not None
assert not list(directory.rglob("eddies*.json"))
assert "py_eddy_tracker" not in sys.modules
print("ok")
"""


def test_without_py_eddy_tracker_the_viewer_data_skips_the_eddies_with_one_warning() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _WITHOUT_PY_EDDY_TRACKER_SCRIPT],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split() == ["x" * 10_000, "ok"]
    warnings = [line for line in completed.stderr.splitlines() if "eddy census" in line]
    assert len(warnings) == 1
    assert "py-eddy-tracker" in warnings[0] and "documentation/viewer-data" in warnings[0]


def test_the_eddy_census_skip_warning_starts_on_its_own_line_in_a_combined_log() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", _WITHOUT_PY_EDDY_TRACKER_SCRIPT],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    assert completed.returncode == 0, completed.stdout
    lines = completed.stdout.splitlines()
    assert "x" * 10_000 in lines
    assert [line for line in lines if "eddy census" in line][0].startswith("Skipping the eddy census")


def _global_sea_surface_height_dataset(first_longitude: float) -> xarray.Dataset:
    latitudes = numpy.arange(-50.0, 50.01, 0.25)
    longitudes = first_longitude + numpy.arange(1440) * 0.25
    grid_latitude, grid_longitude = numpy.meshgrid(latitudes, longitudes, indexing="ij")
    field = sum(
        _gaussian_eddy(grid_latitude, grid_longitude, latitude, longitude, amplitude)
        for latitude, longitude, amplitude in ((30.0, 180.0, 0.2), (-35.0, 60.0, -0.2), (25.0, -100.0, 0.2))
    )
    return xarray.Dataset(
        {
            _SEA_SURFACE_HEIGHT_KEY: (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                field[None, None, :, :],
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(1),
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def _census_centres(detections: list[dict]) -> list[tuple]:
    return sorted((eddy["polarity"], round(eddy["latitude"], 2), round(eddy["longitude"], 2)) for eddy in detections)


def test_eddy_census_on_a_0_to_360_grid_equals_the_census_on_a_minus_180_to_180_grid(py_eddy_tracker) -> None:
    eastern = viewer_artifacts.eddy_census_frames(_global_sea_surface_height_dataset(0.0), 0, None)[0]["detections"]
    centred = viewer_artifacts.eddy_census_frames(_global_sea_surface_height_dataset(-180.0), 0, None)[0]["detections"]

    assert len(eastern) == 3
    assert _census_centres(eastern) == _census_centres(centred)
    assert all(-180.0 <= eddy["longitude"] < 180.0 for eddy in eastern)
    assert -100.0 in [round(eddy["longitude"]) for eddy in eastern]


def test_an_eddy_straddling_the_dateline_keeps_a_closed_contour_around_its_centre(py_eddy_tracker) -> None:
    for first_longitude in (0.0, -180.0):
        detections = viewer_artifacts.eddy_census_frames(_global_sea_surface_height_dataset(first_longitude), 0, None)
        dateline_eddy = next(
            eddy
            for eddy in detections[0]["detections"]
            if eddy["polarity"] == "anticyclone" and round(eddy["latitude"]) == 30
        )
        contour_longitudes = numpy.array(dateline_eddy["contour_longitude"])
        assert len(contour_longitudes) >= 3
        assert numpy.abs(numpy.diff(contour_longitudes)).max() < 5.0
        assert abs(abs(dateline_eddy["longitude"]) - 180.0) < 0.01
        assert contour_longitudes.min() < dateline_eddy["longitude"] < contour_longitudes.max()
        assert numpy.abs(contour_longitudes - dateline_eddy["longitude"]).max() < 5.0
        assert numpy.abs(numpy.array(dateline_eddy["contour_latitude"]) - 30.0).max() < 5.0
