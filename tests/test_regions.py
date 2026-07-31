# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path

import dask.array
import pytest
import xarray

import oceanbench
from oceanbench.core import regions as regions_core
from oceanbench.core.regions import (
    normalize_region_name,
    realism_battery_region_ids,
    region_from_dict,
    region_to_dict,
    resolve_battery_region,
    subset_dataset_to_region,
)

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


def test_realism_battery_regions_are_excluded_from_the_official_score_paths() -> None:
    assert set(realism_battery_region_ids()) == {"gulfstream", "kuroshio"}
    for realism_region_id in realism_battery_region_ids():
        assert realism_region_id not in regions_core.official_region_ids()
        # The gridded / Class-4 paths resolve their region through ``normalize_region_name``
        # (OFFICIAL_REGIONS only); a WBC box must be rejected there.
        with pytest.raises(ValueError, match="Unsupported region"):
            normalize_region_name(realism_region_id)


def test_realism_battery_regions_resolve_and_subset() -> None:
    gulfstream = resolve_battery_region("gulfstream")
    assert gulfstream.bounds.minimum_latitude == 30.0
    assert gulfstream.bounds.maximum_longitude == -50.0

    kuroshio = resolve_battery_region("kuroshio")
    assert kuroshio.bounds.minimum_longitude == 130.0

    dataset = xarray.Dataset(
        coords={
            "latitude": [20.0, 35.0, 50.0],
            "longitude": [-70.0, -60.0, 0.0],
        }
    )
    subset = subset_dataset_to_region(dataset, gulfstream)
    assert subset.sizes["latitude"] == 1
    assert float(subset["latitude"].values[0]) == 35.0
    assert subset.sizes["longitude"] == 2
