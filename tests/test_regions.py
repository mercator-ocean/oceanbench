# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

import dask.array
import nbformat
import pytest
import xarray

import oceanbench
from oceanbench.core.python2jupyter import generate_evaluation_notebook_file


def test_custom_region_roundtrip_and_subset() -> None:
    region = oceanbench.regions.custom(
        identifier="western_med",
        display_name="Western Mediterranean",
        minimum_latitude=5.0,
        maximum_latitude=15.0,
        minimum_longitude=5.0,
        maximum_longitude=15.0,
    )

    region_dict = oceanbench.regions.region_to_dict(region)
    loaded_region = oceanbench.regions.region_from_dict(region_dict)

    assert loaded_region == region
    assert loaded_region.official is False

    dataset = xarray.Dataset(
        coords={
            "latitude": [0.0, 10.0, 20.0],
            "longitude": [0.0, 10.0, 20.0],
        }
    )
    subset = oceanbench.regions.subset(dataset, region)

    assert subset.sizes["latitude"] == 1
    assert subset.sizes["longitude"] == 1
    assert float(subset["latitude"].values[0]) == 10.0
    assert float(subset["longitude"].values[0]) == 10.0


def test_region_subset_accepts_dask_backed_coordinates() -> None:
    region = oceanbench.regions.custom(
        identifier="western_med",
        display_name="Western Mediterranean",
        minimum_latitude=5.0,
        maximum_latitude=15.0,
        minimum_longitude=5.0,
        maximum_longitude=15.0,
    )
    dataset = xarray.Dataset(
        coords={
            "latitude": ("points", dask.array.from_array([0.0, 10.0, 20.0], chunks=2)),
            "longitude": ("points", dask.array.from_array([0.0, 10.0, 20.0], chunks=2)),
        }
    )

    subset = oceanbench.regions.subset(dataset, region)

    assert subset.sizes["points"] == 1
    assert float(subset["latitude"].values[0]) == 10.0
    assert float(subset["longitude"].values[0]) == 10.0


def test_load_region_file_and_reject_reserved_official_id(tmp_path) -> None:
    region_path = tmp_path / "region.json"
    region_path.write_text(
        json.dumps(
            {
                "id": "western_med",
                "display_name": "Western Mediterranean",
                "bounds": {
                    "minimum_latitude": 5.0,
                    "maximum_latitude": 15.0,
                    "minimum_longitude": 5.0,
                    "maximum_longitude": 15.0,
                },
            }
        ),
        encoding="utf-8",
    )

    loaded_region = oceanbench.regions.load_region_file(region_path)

    assert loaded_region.id == "western_med"
    assert loaded_region.display_name == "Western Mediterranean"
    assert loaded_region.official is False

    reserved_region_path = tmp_path / "reserved.json"
    reserved_region_path.write_text(
        json.dumps(
            {
                "id": "ibi",
                "display_name": "Fake IBI",
                "bounds": {
                    "minimum_latitude": 0.0,
                    "maximum_latitude": 1.0,
                    "minimum_longitude": 0.0,
                    "maximum_longitude": 1.0,
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reserved for the built-in official region"):
        oceanbench.regions.load_region_file(reserved_region_path)


def test_load_region_file_reports_missing_path_cleanly(tmp_path) -> None:
    missing_region_path = tmp_path / "missing.json"

    with pytest.raises(ValueError, match="Unable to read region file"):
        oceanbench.regions.load_region_file(missing_region_path)


def test_generate_evaluation_notebook_embeds_custom_region_configuration(tmp_path) -> None:
    challenger_path = tmp_path / "challenger.py"
    challenger_path.write_text(
        "import xarray\n\nchallenger_dataset = xarray.Dataset()\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "custom.report.ipynb"
    custom_region = oceanbench.regions.custom(
        identifier="western_med",
        display_name="Western Mediterranean",
        minimum_latitude=5.0,
        maximum_latitude=15.0,
        minimum_longitude=5.0,
        maximum_longitude=15.0,
    )

    generate_evaluation_notebook_file(
        str(challenger_path),
        str(output_path),
        region=custom_region,
    )

    notebook = nbformat.read(output_path, as_version=4)

    assert "region = oceanbench.regions.region_from_dict(" in notebook.cells[4].source
    assert notebook.metadata["oceanbench"]["region"]["id"] == "western_med"
    assert notebook.metadata["oceanbench"]["region"]["official"] is False


def test_generate_evaluation_notebook_keeps_official_region_string(tmp_path) -> None:
    challenger_path = tmp_path / "challenger.py"
    challenger_path.write_text(
        "import xarray\n\nchallenger_dataset = xarray.Dataset()\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "official.report.ipynb"

    generate_evaluation_notebook_file(
        str(challenger_path),
        str(output_path),
        region="ibi",
    )

    notebook = nbformat.read(output_path, as_version=4)

    assert notebook.cells[4].source == "region = 'ibi'"
    assert notebook.metadata["oceanbench"]["region"]["id"] == "ibi"
    assert notebook.metadata["oceanbench"]["region"]["official"] is True
