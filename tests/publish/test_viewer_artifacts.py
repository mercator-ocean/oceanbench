# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Viewer serving artifact writers (Class-4 match-up parquet, eddy census, year JSON).

Every test uses small synthetic in-memory inputs; no real challenger data is downloaded.
"""

import json

import numpy
import pandas
import pyarrow.parquet
import pytest
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.publish import viewer_artifacts

_SEA_SURFACE_HEIGHT_KEY = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()


def _matchup_frame() -> pandas.DataFrame:
    generator = numpy.random.default_rng(3)
    rows = []
    for start_date in ("2024-01-10", "2024-01-03"):
        for lead_day in (5, 1):
            for variable, depth_bin in (
                ("sea_surface_height_above_geoid", "surface"),
                ("sea_water_potential_temperature", "0-5m"),
            ):
                for _ in range(4):
                    observation_value = float(generator.normal())
                    model_value = float(generator.normal())
                    rows.append(
                        {
                            "variable": variable,
                            "depth_bin": depth_bin,
                            "lead_day": lead_day,
                            "start_date": numpy.datetime64(start_date),
                            "latitude": float(generator.uniform(-60, 60)),
                            "longitude": float(generator.uniform(-180, 180)),
                            "observation_value": observation_value,
                            "model_value": model_value,
                        }
                    )
    return pandas.DataFrame(rows)


def test_matchup_parquet_layout_and_verification(tmp_path) -> None:
    output_path = str(tmp_path / "class4-matchups.parquet")
    viewer_artifacts.write_matchup_parquet(_matchup_frame(), output_path)

    parquet_file = pyarrow.parquet.ParquetFile(output_path)
    metadata = parquet_file.metadata
    assert parquet_file.schema_arrow.names == viewer_artifacts._MATCHUP_TARGET_SCHEMA.names
    # One (start_date, lead_day, variable, depth_bin) group per row group: 2 starts x 2 leads x 2
    # variables = 8 groups, each pure and never straddling a variable boundary, ascending.
    assert metadata.num_row_groups == 8
    names = parquet_file.schema_arrow.names
    group_indices = [names.index(column) for column in viewer_artifacts._MATCHUP_GROUP_COLUMNS]
    previous_key = None
    for group_index in range(metadata.num_row_groups):
        row_group = metadata.row_group(group_index)
        statistics = [row_group.column(column_index).statistics for column_index in group_indices]
        assert all(column.min == column.max for column in statistics)
        key = tuple(column.min for column in statistics)
        assert previous_key is None or key > previous_key
        previous_key = key
    # abs_error is |model - observation|.
    table = parquet_file.read().to_pandas()
    numpy.testing.assert_allclose(
        table["abs_error"].to_numpy(),
        numpy.abs(table["model_value"].to_numpy() - table["observation_value"].to_numpy()),
        rtol=1e-6,
    )


def test_matchup_parquet_splits_a_large_pair_across_row_groups(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(viewer_artifacts, "MAXIMUM_ROW_GROUP_ROWS", 3)
    frame = pandas.DataFrame(
        {
            "variable": ["sea_surface_height_above_geoid"] * 7,
            "depth_bin": ["surface"] * 7,
            "lead_day": [1] * 7,
            "start_date": [numpy.datetime64("2024-01-03")] * 7,
            "latitude": numpy.linspace(-1, 1, 7),
            "longitude": numpy.linspace(-1, 1, 7),
            "observation_value": numpy.zeros(7),
            "model_value": numpy.ones(7),
        }
    )
    output_path = str(tmp_path / "matchups.parquet")
    viewer_artifacts.write_matchup_parquet(frame, output_path)
    metadata = pyarrow.parquet.ParquetFile(output_path).metadata
    assert metadata.num_row_groups == 3  # 3 + 3 + 1
    assert all(metadata.row_group(index).num_rows <= 3 for index in range(metadata.num_row_groups))


def test_verify_rejects_a_mixed_pair_row_group(tmp_path) -> None:
    frame = pandas.DataFrame(
        {
            "variable": ["sea_surface_height_above_geoid"] * 4,
            "depth_bin": ["surface"] * 4,
            "lead_day": [1, 1, 2, 2],
            "start_date": [numpy.datetime64("2024-01-03")] * 4,
            "latitude": [0.0, 1.0, 2.0, 3.0],
            "longitude": [0.0, 1.0, 2.0, 3.0],
            "observation_value": [0.0, 0.0, 0.0, 0.0],
            "model_value": [1.0, 1.0, 1.0, 1.0],
        }
    )
    projected = viewer_artifacts._project_matchups(pyarrow.Table.from_pandas(frame, preserve_index=False))
    output_path = str(tmp_path / "impure.parquet")
    # Write everything as a single row group on purpose: it mixes lead_day 1 and 2.
    pyarrow.parquet.write_table(projected, output_path, compression="snappy", row_group_size=1_000)
    with pytest.raises(ValueError, match="mixes more than one"):
        viewer_artifacts.verify_matchup_parquet(output_path)


def test_year_artifacts_handle_a_mixed_variable_row_group(tmp_path) -> None:
    """A row group pure in (start_date, lead_day) but mixing variables must not be corrupted.

    The buggy assumption that a row group is homogeneous in variable would attribute every row to
    a single variable. Here one row group carries both SSH (surface) and temperature (0-5m) super
    observations at distinct latitudes; each must land in its own variable channel and cell.
    """
    frame = pandas.DataFrame(
        {
            "variable": [
                "sea_surface_height_above_geoid",
                "sea_surface_height_above_geoid",
                "sea_water_potential_temperature",
                "sea_water_potential_temperature",
            ],
            "depth_bin": ["surface", "surface", "0-5m", "0-5m"],
            "lead_day": [1, 1, 1, 1],
            "start_date": [numpy.datetime64("2024-01-03")] * 4,
            # SSH near the equator (grid cell A), temperature near 40N (grid cell B).
            "latitude": [1.0, 1.0, 41.0, 41.0],
            "longitude": [1.0, 1.0, 1.0, 1.0],
            "observation_value": [10.0, 10.0, 20.0, 20.0],
            "model_value": [12.0, 14.0, 25.0, 27.0],
        }
    )
    matchup_path = str(tmp_path / "class4-matchups.parquet")
    viewer_artifacts.write_matchup_parquet(frame, matchup_path)

    geography_path = str(tmp_path / "year-error-geography.json")
    rmsd_path = str(tmp_path / "year-rmsd-by-start.json")
    viewer_artifacts._write_year_artifacts(matchup_path, "global", geography_path, rmsd_path, source="synthetic")

    geography = json.loads(open(geography_path).read())
    ssh_cells = [value for value in geography["variables"]["SSH"]["leads"]["1"] if value is not None]
    temperature_cells = [value for value in geography["variables"]["T"]["leads"]["1"] if value is not None]
    # SSH abs errors 2 and 4 -> mean 3; temperature abs errors 5 and 7 -> mean 6.
    assert ssh_cells == [pytest.approx(3.0)]
    assert temperature_cells == [pytest.approx(6.0)]
    # Signed bias (model - obs): SSH +2 and +4 -> +3; temperature +5 and +7 -> +6.
    ssh_bias = [value for value in geography["variables"]["SSH"]["bias"]["1"] if value is not None]
    temperature_bias = [value for value in geography["variables"]["T"]["bias"]["1"] if value is not None]
    assert ssh_bias == [pytest.approx(3.0)]
    assert temperature_bias == [pytest.approx(6.0)]

    rmsd = json.loads(open(rmsd_path).read())
    assert rmsd["variables"]["SSH"]["leads"]["1"]["dates"] == ["2024-01-03"]
    assert rmsd["variables"]["SSH"]["leads"]["1"]["bias"] == [pytest.approx(3.0)]
    assert rmsd["variables"]["T"]["depth_bin"] == "0-5m"


def _year_ci_frame() -> pandas.DataFrame:
    """Many SSH super-obs spread across cells over two starts, so per-start CIs are well defined."""
    generator = numpy.random.default_rng(7)
    rows = []
    for start_date in ("2024-01-03", "2024-01-10"):
        for _ in range(200):
            observation_value = float(generator.normal())
            model_value = observation_value + float(generator.normal(0.5, 0.2))
            rows.append(
                {
                    "variable": "sea_surface_height_above_geoid",
                    "depth_bin": "surface",
                    "lead_day": 1,
                    "start_date": numpy.datetime64(start_date),
                    "latitude": float(generator.uniform(-60, 60)),
                    "longitude": float(generator.uniform(-180, 180)),
                    "observation_value": observation_value,
                    "model_value": model_value,
                }
            )
    return pandas.DataFrame(rows)


def _write_year_pair(tmp_path):
    matchup_path = str(tmp_path / "class4-matchups.parquet")
    viewer_artifacts.write_matchup_parquet(_year_ci_frame(), matchup_path)
    geography_path = str(tmp_path / "year-error-geography.json")
    rmsd_path = str(tmp_path / "year-rmsd-by-start.json")
    viewer_artifacts._write_year_artifacts(matchup_path, "global", geography_path, rmsd_path, source="synthetic")
    return (json.loads(open(geography_path).read()), json.loads(open(rmsd_path).read()))


def test_year_rmsd_by_start_carries_bracketing_confidence_intervals(tmp_path) -> None:
    _, rmsd = _write_year_pair(tmp_path)
    series = rmsd["variables"]["SSH"]["leads"]["1"]
    n_dates = len(series["dates"])
    for key in ("rmsd_ci_low", "rmsd_ci_high", "bias_ci_low", "bias_ci_high"):
        assert len(series[key]) == n_dates
    for index in range(n_dates):
        assert series["rmsd_ci_low"][index] <= series["rmsd"][index] <= series["rmsd_ci_high"][index]
        assert series["bias_ci_low"][index] <= series["bias"][index] <= series["bias_ci_high"][index]
    assert "seed" in rmsd["meta"]["ci_method"] and "bootstrap" in rmsd["meta"]["ci_method"]


def test_year_ci_is_deterministic_under_fixed_seed(tmp_path) -> None:
    first = _write_year_pair(tmp_path / "a")
    second = _write_year_pair(tmp_path / "b")
    assert first[1]["variables"]["SSH"]["leads"]["1"] == second[1]["variables"]["SSH"]["leads"]["1"]
    assert first[0]["variables"]["SSH"]["bias_se"] == second[0]["variables"]["SSH"]["bias_se"]


def test_year_rmsd_by_start_is_the_pooled_reduction(tmp_path) -> None:
    frame = _year_ci_frame()
    matchup_path = str(tmp_path / "class4-matchups.parquet")
    viewer_artifacts.write_matchup_parquet(frame, matchup_path)
    geography_path = str(tmp_path / "year-error-geography.json")
    rmsd_path = str(tmp_path / "year-rmsd-by-start.json")
    viewer_artifacts._write_year_artifacts(matchup_path, "global", geography_path, rmsd_path, source="synthetic")
    series = json.loads(open(rmsd_path).read())["variables"]["SSH"]["leads"]["1"]
    for index, start_date in enumerate(series["dates"]):
        subset = frame[frame["start_date"] == numpy.datetime64(start_date)]
        error = (subset["model_value"] - subset["observation_value"]).to_numpy()
        assert series["rmsd"][index] == pytest.approx(float(numpy.sqrt((error * error).mean())), rel=2e-5)
        assert series["bias"][index] == pytest.approx(float(error.mean()), abs=2e-5)
        assert series["n"][index] == len(subset)


def test_binned_bootstrap_matches_analytic_interval_at_large_n() -> None:
    generator = numpy.random.default_rng(5)
    squared = generator.gamma(2.0, 0.01, size=50_000)
    low, high = viewer_artifacts._bootstrap_rmsd_ci(squared, numpy.random.default_rng(1))
    point = float(numpy.sqrt(squared.mean()))
    assert low < point < high
    # At this n the bootstrap interval of the mean of squares must agree with the analytic normal
    # interval within 10% of the half-width.
    mean = squared.mean()
    standard_error = squared.std() / numpy.sqrt(squared.size)
    analytic_half = float(numpy.sqrt(mean + 1.96 * standard_error) - numpy.sqrt(mean - 1.96 * standard_error)) / 2.0
    assert (high - low) / 2.0 == pytest.approx(analytic_half, rel=0.10)


def test_year_geography_carries_shared_counts_and_bias_standard_error(tmp_path) -> None:
    geography, _ = _write_year_pair(tmp_path)
    ssh = geography["variables"]["SSH"]
    assert set(ssh) == {"leads", "bias", "n", "bias_se"}
    counts = ssh["n"]["1"]
    bias_se = ssh["bias_se"]["1"]
    bias = ssh["bias"]["1"]
    grid = geography["grid"]
    assert len(counts) == len(bias_se) == grid["nlat"] * grid["nlon"]
    for cell, count in enumerate(counts):
        if count >= 2:
            assert bias_se[cell] is not None and bias_se[cell] >= 0.0
            assert bias[cell] is not None
        else:
            assert bias_se[cell] is None


def test_class4_bias_per_start_records_are_signed_means() -> None:
    from oceanbench.runner.records import RunContext

    frame = pandas.DataFrame(
        {
            "variable": ["sea_water_salinity"] * 3,
            "depth_bin": ["0-5m"] * 3,
            "lead_day": [1, 1, 1],
            "start_date": [numpy.datetime64("2024-01-03")] * 3,
            "observation_value": [10.0, 10.0, 10.0],
            "model_value": [11.0, 12.0, 13.0],
        }
    )
    context = RunContext(
        challenger="your_model", challenger_version="local", year=2024, region="global", oceanbench_version="test"
    )
    records = viewer_artifacts.class4_bias_per_start_records(frame, context=context)
    assert len(records) == 1
    assert records[0]["metric"] == "class4_bias"
    assert records[0]["value"] == pytest.approx(2.0)  # mean(1, 2, 3)
    assert records[0]["n"] == 3
    assert records[0]["depth"] == "0-5m"


def _depth_matchup_frame() -> pandas.DataFrame:
    """3D temperature/salinity (multiple depth bins) plus surface-only SSH, over two starts."""
    generator = numpy.random.default_rng(19)
    rows = []
    for start_date in ("2024-01-03", "2024-01-10"):
        for lead_day in (1, 5):
            for variable, depth_bin in (
                ("sea_water_potential_temperature", "0-5m"),
                ("sea_water_potential_temperature", "100-300m"),
                ("sea_water_salinity", "0-5m"),
                ("sea_water_salinity", "100-300m"),
                ("sea_surface_height_above_geoid", "surface"),
            ):
                for _ in range(6):
                    observation_value = float(generator.normal())
                    model_value = observation_value + float(generator.normal(0.3, 0.2))
                    rows.append(
                        {
                            "variable": variable,
                            "depth_bin": depth_bin,
                            "lead_day": lead_day,
                            "start_date": numpy.datetime64(start_date),
                            "latitude": float(generator.uniform(-60, 60)),
                            "longitude": float(generator.uniform(-180, 180)),
                            "observation_value": observation_value,
                            "model_value": model_value,
                        }
                    )
    return pandas.DataFrame(rows)


def test_rmsd_by_depth_pools_over_all_starts_and_skips_surface_only(tmp_path) -> None:
    frame = _depth_matchup_frame()
    matchup_path = str(tmp_path / "class4-matchups.parquet")
    viewer_artifacts.write_matchup_parquet(frame, matchup_path)
    output_path = str(tmp_path / "rmsd-by-depth.json")
    returned = viewer_artifacts.write_rmsd_by_depth(
        matchup_path, output_path, challenger="your_model", region="global", source="synthetic"
    )
    assert returned == output_path
    payload = json.loads(open(output_path).read())
    assert payload["schema_version"] == 1
    assert payload["challenger"] == "your_model"
    assert payload["region"] == "global"
    assert payload["provenance"]["source"] == "synthetic"

    # Only the 3D multi-depth variables appear; surface-only SSH is skipped.
    assert set(payload["variables"]) == {"sea_water_potential_temperature", "sea_water_salinity"}
    block = payload["variables"]["sea_water_potential_temperature"]
    assert block["depth_bins"] == ["0-5m", "100-300m"]  # surface -> deep
    assert block["leads"] == [1, 5]

    # Every cell pools ALL match-ups of that (variable, depth_bin, lead) over both starts.
    for depth_index, depth_bin in enumerate(block["depth_bins"]):
        for lead_index, lead in enumerate(block["leads"]):
            subset = frame[
                (frame["variable"] == "sea_water_potential_temperature")
                & (frame["depth_bin"] == depth_bin)
                & (frame["lead_day"] == lead)
            ]
            error = (subset["model_value"] - subset["observation_value"]).to_numpy()
            assert block["rmsd"][depth_index][lead_index] == pytest.approx(
                float(numpy.sqrt((error * error).mean())), rel=1e-5
            )
            assert block["bias"][depth_index][lead_index] == pytest.approx(float(error.mean()), abs=1e-5)
            assert block["n"][depth_index][lead_index] == len(subset)


def test_rmsd_by_depth_returns_none_when_no_multi_depth_variable(tmp_path) -> None:
    matchup_path = str(tmp_path / "class4-matchups.parquet")
    viewer_artifacts.write_matchup_parquet(_matchup_frame(), matchup_path)  # SSH surface + T single 0-5m bin
    output_path = str(tmp_path / "rmsd-by-depth.json")
    assert viewer_artifacts.write_rmsd_by_depth(matchup_path, output_path, challenger="m", region="global") is None
    assert not (tmp_path / "rmsd-by-depth.json").exists()


def _sea_surface_height_dataset() -> xarray.Dataset:
    latitudes = numpy.linspace(-10.0, 10.0, 41)
    longitudes = numpy.linspace(-10.0, 10.0, 41)
    grid_y, grid_x = numpy.meshgrid(latitudes, longitudes, indexing="ij")
    field = 0.3 * numpy.exp(-((grid_y) ** 2 + (grid_x) ** 2) / 8.0)
    values = numpy.broadcast_to(field[None, None, :, :], (1, 5, field.shape[0], field.shape[1])).copy()
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
                {"standard_name": _SEA_SURFACE_HEIGHT_KEY},
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(5),
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def test_dataset_eddy_census_payload_shape_and_stamp(tmp_path) -> None:
    census = viewer_artifacts.dataset_eddy_census(
        _sea_surface_height_dataset(), dataset_slug="your_model", lead_days=(1,)
    )
    assert census["kind"] == "eddy-census"
    assert census["schema_version"] == "1"
    assert census["dataset"] == "your_model"
    assert census["parameters"]["apply_contour_filtering"] is True
    assert "oceanbench_version" in census["parameters"]
    assert census["provenance"]["source"] == "your_model"
    assert census["provenance"]["parameters"] == census["parameters"]
    assert [frame["lead_day"] for frame in census["frames"]] == [1]
    for frame in census["frames"]:
        for eddy in frame["detections"]:
            assert set(eddy) == {"id", "latitude", "longitude", "polarity", "contour_latitude", "contour_longitude"}
            assert -90.0 <= eddy["latitude"] <= 90.0
            assert -180.0 <= eddy["longitude"] <= 180.0

    output_path = str(tmp_path / "eddies.json")
    viewer_artifacts.write_eddy_census(
        _sea_surface_height_dataset(), output_path, dataset_slug="your_model", lead_days=(1, 5)
    )
    index = json.loads(open(output_path).read())
    assert index["kind"] == "eddy-census"
    # The index lists one file per lead and each lead file carries just that lead's frame.
    assert [entry["lead_day"] for entry in index["leads"]] == [1, 5]
    for entry in index["leads"]:
        lead_payload = json.loads(open(tmp_path / entry["file"]).read())
        assert lead_payload["frame"]["lead_day"] == entry["lead_day"]
        assert lead_payload["kind"] == "eddy-census"


def test_matchup_parquet_carries_provenance_metadata(tmp_path) -> None:
    output_path = str(tmp_path / "class4-matchups.parquet")
    source = "insights/model/global/class4-matchups.parquet"
    viewer_artifacts.write_matchup_parquet(_matchup_frame(), output_path, source=source)

    metadata = pyarrow.parquet.ParquetFile(output_path).schema_arrow.metadata
    provenance = json.loads(metadata[viewer_artifacts._MATCHUP_PROVENANCE_METADATA_KEY])
    assert provenance["oceanbench_version"]
    assert provenance["source"] == "insights/model/global/class4-matchups.parquet"
    assert provenance["generated_at"].endswith("Z")
    assert "git_commit" in provenance


def test_year_artifacts_carry_provenance(tmp_path) -> None:
    matchup_path = str(tmp_path / "class4-matchups.parquet")
    viewer_artifacts.write_matchup_parquet(_matchup_frame(), matchup_path)
    geography_path = str(tmp_path / "year-error-geography.json")
    rmsd_path = str(tmp_path / "year-rmsd-by-start.json")
    source = "insights/model/global/class4-matchups.parquet"
    viewer_artifacts._write_year_artifacts(matchup_path, "global", geography_path, rmsd_path, source=source)

    for path in (geography_path, rmsd_path):
        provenance = json.loads(open(path).read())["provenance"]
        assert provenance["oceanbench_version"]
        assert provenance["source"] == "insights/model/global/class4-matchups.parquet"
        assert provenance["parameters"]["region"] == "global"


def test_provenance_block_shape() -> None:
    block = viewer_artifacts.provenance_block(source="thing", parameters={"a": 1})
    assert set(block) == {"oceanbench_version", "git_commit", "generated_at", "source", "parameters"}
    assert block["parameters"] == {"a": 1}
