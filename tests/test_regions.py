# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path

import dask.array
import pytest
import xarray

import oceanbench
from oceanbench.core.regions import region_from_dict, region_to_dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WESTERN_MED_REGION_FILE = PROJECT_ROOT / "assets" / "western_med_region.json"


def test_custom_region_roundtrip_and_subset() -> None:
    region = oceanbench.regions.custom(
        identifier="western_med",
        display_name="Western Mediterranean",
        minimum_latitude=5.0,
        maximum_latitude=15.0,
        minimum_longitude=5.0,
        maximum_longitude=15.0,
    )

    region_dict = region_to_dict(region)
    loaded_region = region_from_dict(region_dict)

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


def test_example_custom_region_file_loads_as_a_custom_region() -> None:
    custom_region = oceanbench.regions.load_region_file(WESTERN_MED_REGION_FILE)

    assert custom_region.id == "western_med"
    assert custom_region.display_name == "Western Mediterranean"
    assert custom_region.official is False
