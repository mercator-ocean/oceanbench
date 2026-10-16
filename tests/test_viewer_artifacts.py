# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

import numpy
import pandas
import pyarrow.parquet
import pytest
import xarray

from oceanbench.core.classIV import class4_rmsd_table
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.regions import custom_region, padded_region, subset_dataset_to_region
from oceanbench.publish import viewer_artifacts
from tests import synthetic_class4
from tests.test_classiv_matchups import _matchups

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
    table = class4_rmsd_table(matchups, synthetic_class4.LEAD_DAYS_COUNT)
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
        assert series["n"][index] == len(start)
        assert series["rmsd_ci_low"][index] <= series["rmsd"][index] <= series["rmsd_ci_high"][index]
        assert series["bias_ci_low"][index] <= series["bias"][index] <= series["bias_ci_high"][index]
    assert _year_rmsd_by_start(matchups) == by_start


def test_binned_bootstrap_matches_the_analytic_interval_at_large_counts() -> None:
    squared_error = numpy.random.default_rng(5).gamma(2.0, 0.01, size=50_000)

    low, high = viewer_artifacts._bootstrap_rmsd_ci(squared_error, numpy.random.default_rng(1))

    mean = squared_error.mean()
    standard_error = squared_error.std() / numpy.sqrt(squared_error.size)
    analytic_half_width = (numpy.sqrt(mean + 1.96 * standard_error) - numpy.sqrt(mean - 1.96 * standard_error)) / 2.0
    assert low < numpy.sqrt(mean) < high
    assert (high - low) / 2.0 == pytest.approx(analytic_half_width, rel=0.10)


def test_year_error_geography_shares_its_counts_with_the_bias_standard_error() -> None:
    geography = viewer_artifacts.year_error_geography(_year_matchups(), viewer_artifacts.YEAR_GRIDS["global"], 1)

    sea_surface_height = geography["variables"]["SSH"]
    grid = geography["grid"]
    assert len(sea_surface_height["n"]["1"]) == grid["nlat"] * grid["nlon"]
    assert sum(sea_surface_height["n"]["1"]) == 400
    for count, bias, bias_standard_error in zip(
        sea_surface_height["n"]["1"], sea_surface_height["bias"]["1"], sea_surface_height["bias_se"]["1"]
    ):
        assert (bias is None) == (count == 0)
        assert (bias_standard_error is None) == (count < 2)


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


def _sea_surface_height_dataset(start_dates=("2024-01-03",)) -> xarray.Dataset:
    # Centred at 30 N, clear of the equatorial band the detector excludes.
    latitudes = numpy.linspace(20.0, 40.0, 41)
    longitudes = numpy.linspace(-10.0, 10.0, 41)
    grid_y, grid_x = numpy.meshgrid(latitudes - 30.0, longitudes, indexing="ij")
    field = 0.3 * numpy.exp(-(grid_y**2 + grid_x**2) / 8.0)
    values = numpy.broadcast_to(field[None, None, :, :], (len(start_dates), 2, *field.shape)).copy()
    # Each start gets its own displaced eddy, so a census that ignored the start is detectable.
    for start_index in range(len(start_dates)):
        values[start_index] += 0.3 * numpy.exp(-((grid_y - start_index) ** 2 + (grid_x - 2 * start_index) ** 2) / 4.0)
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


def test_eddy_census_writes_every_start_and_lead_day_with_distinct_frames(tmp_path) -> None:
    start_dates = ("2024-01-03", "2024-01-10", "2024-01-17")
    viewer_artifacts.write_eddy_census(
        _sea_surface_height_dataset(start_dates), tmp_path / "eddies.json", "your_model", None
    )

    index = json.loads((tmp_path / "eddies.json").read_text(encoding="utf-8"))
    assert index["kind"] == "eddy-census"
    assert index["dataset"] == "your_model"
    assert [entry["start_date"] for entry in index["starts"]] == list(start_dates)
    assert index["leads"] == index["starts"][0]["leads"]
    centres_by_start = set()
    for start_entry in index["starts"]:
        assert [entry["lead_day"] for entry in start_entry["leads"]] == [1, 2]
        for lead_entry in start_entry["leads"]:
            assert lead_entry["file"] == f"eddies-start-{start_entry['start_date']}-lead-{lead_entry['lead_day']}.json"
            payload = json.loads((tmp_path / lead_entry["file"]).read_text(encoding="utf-8"))
            assert (payload["start_date"], payload["frame"]["lead_day"]) == (
                start_entry["start_date"],
                lead_entry["lead_day"],
            )
            for eddy in payload["frame"]["detections"]:
                assert len(eddy["contour_latitude"]) == len(eddy["contour_longitude"]) >= 3
        first_frame = json.loads((tmp_path / start_entry["leads"][0]["file"]).read_text(encoding="utf-8"))["frame"]
        centres_by_start.add(tuple(sorted((eddy["latitude"], eddy["longitude"]) for eddy in first_frame["detections"])))
    assert len(centres_by_start) == len(start_dates)


def test_regional_eddy_census_keeps_only_the_eddies_centred_in_the_region() -> None:
    dataset = _sea_surface_height_dataset()
    everywhere = viewer_artifacts.eddy_census_frames(dataset, 0, None)[0]["detections"]
    assert everywhere

    def centres(minimum_latitude: float, maximum_latitude: float) -> list:
        region = custom_region("band", "Band", minimum_latitude, maximum_latitude, -10.0, 10.0)
        padded = subset_dataset_to_region(dataset, padded_region(region, 10.0))
        return viewer_artifacts.eddy_census_frames(padded, 0, region)[0]["detections"]

    assert centres(25.0, 35.0) == everywhere
    assert centres(35.0, 40.0) == []


def test_viewer_artifacts_of_a_custom_region_have_no_year_files_and_no_maps(tmp_path) -> None:
    region = custom_region("band", "Band", 22.0, 38.0, -8.0, 8.0)
    matchups = _frame([(_SEA_SURFACE_HEIGHT_KEY, "surface")], ["2024-01-03"], [0, 1], 20)
    rows_by_start = viewer_artifacts.class4_rmsd_rows_by_start(matchups)

    artifacts = viewer_artifacts.write_viewer_artifacts(
        tmp_path, "your_model", _sea_surface_height_dataset(), region, matchups, rows_by_start
    )

    assert artifacts.has_maps is False
    assert artifacts.insights == {
        "class4_matchups": "./data/insights/your_model/band/class4-matchups.parquet",
        "eddies": "./data/insights/your_model/band/eddies.json",
        "rmsd_by_depth": None,
        "year_error_geography": None,
        "year_rmsd_by_start": None,
    }
    assert sorted(path.name for path in tmp_path.iterdir()) == ["insights"]
