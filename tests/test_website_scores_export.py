# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

import pandas
import pytest
import xarray

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.scores_export import ExportFormat, scores_table, write_scores_exports  # noqa: E402

METRIC_TITLES = {
    "rmsd_variables_glorys": "Forecasted variables",
    "rmsd_mld_glorys": "RMSD of Mixed Layer Depth",
}
SECTIONS = {
    "reanalysis": {"depth_metric": "rmsd_variables_glorys", "flat_metrics": ["rmsd_mld_glorys"]},
}


def _challenger_scores(name: str, score: float) -> dict:
    return {
        "rmsd_variables_glorys": {
            "name": name,
            "depths": {
                "Surface": {
                    "variables": {
                        "temperature": {
                            "standard_name": "sea_water_potential_temperature",
                            "unit": "°C",
                            "data": {"1": score, "2": score * 2, "3": None},
                        }
                    }
                }
            },
        },
        "rmsd_mld_glorys": {
            "name": name,
            "depths": {
                "flat": {
                    "variables": {
                        "mixed layer depth": {
                            "standard_name": "ocean_mixed_layer_thickness",
                            "unit": "m",
                            "data": {"1": score},
                        }
                    }
                }
            },
        },
    }


def _version_bundle() -> dict:
    return {
        "regions": {
            "global": {
                "challengers": {
                    "glonet": _challenger_scores("glonet", 0.5),
                    "glonet_1_degree": _challenger_scores("glonet_1_degree", 0.7),
                },
                "challenger_names": ["glonet", "glonet_1_degree"],
            },
            "ibi": {
                "challengers": {"glonet": _challenger_scores("glonet", 0.9)},
                "challenger_names": ["glonet"],
            },
        },
        "region_order": ["global", "ibi"],
    }


def _write(output_directory: Path) -> dict:
    return write_scores_exports({"0.6.0": _version_bundle()}, METRIC_TITLES, SECTIONS, output_directory)


def test_table_has_one_row_per_challenger_metric_depth_variable_and_lead_day():
    table = scores_table("0.6.0", "global", "high_resolution", _version_bundle(), METRIC_TITLES, SECTIONS)

    assert list(table.columns) == [
        "version",
        "region",
        "track",
        "challenger",
        "comparison",
        "metric",
        "depth",
        "variable",
        "standard_name",
        "unit",
        "lead_day",
        "value",
    ]
    assert len(table) == 4
    assert set(table["challenger"]) == {"glonet"}
    assert set(table["comparison"]) == {"reanalysis"}
    assert table["lead_day"].tolist() == [1, 2, 3, 1]
    assert table["value"].fillna(-1).tolist() == [0.5, 1.0, -1, 0.5]


def test_one_degree_challengers_are_in_the_one_degree_track():
    table = scores_table("0.6.0", "global", "one_degree", _version_bundle(), METRIC_TITLES, SECTIONS)

    assert set(table["challenger"]) == {"glonet_1_degree"}


@pytest.mark.parametrize("export_format", [ExportFormat.CSV, ExportFormat.PARQUET])
def test_flat_formats_expose_the_same_scores_for_each_version_region_and_track(tmp_path, export_format):
    _write(tmp_path)
    expected = scores_table("0.6.0", "global", "one_degree", _version_bundle(), METRIC_TITLES, SECTIONS)

    path = tmp_path / "0.6.0" / "global" / f"one_degree.{export_format.value}"
    opened = {
        ExportFormat.CSV: lambda: pandas.read_csv(path, keep_default_na=False, na_values=[""]),
        ExportFormat.PARQUET: lambda: pandas.read_parquet(path),
    }[export_format]()

    pandas.testing.assert_frame_equal(opened.fillna(""), expected.fillna(""), check_dtype=False)


def test_zarr_is_a_multi_dimensional_dataset_of_scores(tmp_path):
    _write(tmp_path)

    dataset = xarray.open_zarr(tmp_path / "0.6.0" / "global" / "high_resolution.zarr", consolidated=True)

    assert set(dataset.dims) == {"challenger", "comparison", "metric", "depth", "variable", "lead_day"}
    assert dataset["value"].dims == ("challenger", "comparison", "metric", "depth", "variable", "lead_day")
    surface_temperature = dataset["value"].sel(
        challenger="glonet",
        comparison="reanalysis",
        metric="Forecasted variables",
        depth="Surface",
        variable="temperature",
    )
    assert surface_temperature.values.tolist()[:2] == [0.5, 1.0]
    assert (
        dataset["unit"].sel(
            comparison="reanalysis", metric="Forecasted variables", depth="Surface", variable="temperature"
        )
        == "°C"
    )
    assert (
        dataset["standard_name"].sel(
            comparison="reanalysis", metric="RMSD of Mixed Layer Depth", depth="", variable="mixed layer depth"
        )
        == "ocean_mixed_layer_thickness"
    )


def test_tracks_without_challengers_are_not_exported(tmp_path):
    _write(tmp_path)

    assert not (tmp_path / "0.6.0" / "ibi" / "one_degree.csv").exists()
    assert (tmp_path / "0.6.0" / "ibi" / "high_resolution.csv").exists()


def test_index_lists_every_exported_file(tmp_path):
    index = _write(tmp_path)

    assert json.loads(json.dumps(index)) == index
    assert index["versions"]["0.6.0"]["global"]["one_degree"] == {
        "csv": "api/scores/0.6.0/global/one_degree.csv",
        "parquet": "api/scores/0.6.0/global/one_degree.parquet",
        "zarr": "api/scores/0.6.0/global/one_degree.zarr",
    }
    assert list(index["versions"]["0.6.0"]["ibi"]) == ["high_resolution"]
