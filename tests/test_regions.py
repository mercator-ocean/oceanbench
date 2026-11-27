# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path

import dask.array
import numpy
import pytest
import xarray

import oceanbench
from oceanbench.core.regions import IBI, padded_region, region_from_dict, region_to_dict

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


def test_example_custom_region_file_loads_a_custom_region() -> None:
    custom_region = oceanbench.regions.load_region_file(WESTERN_MED_REGION_FILE)

    assert region_to_dict(custom_region)["id"] == "western_med"
    assert region_to_dict(custom_region)["display_name"] == "Western Mediterranean"
    assert custom_region.official is False


def _global_grid(longitudes: numpy.ndarray) -> xarray.Dataset:
    return xarray.Dataset(coords={"latitude": [0.0], "longitude": longitudes})


def test_region_longitude_box_spanning_greenwich_selects_the_same_cells_on_a_0_360_grid() -> None:
    region = oceanbench.regions.custom("greenwich", "Greenwich", -1.0, 1.0, -10.0, 10.0)

    subset = oceanbench.regions.subset(_global_grid(numpy.arange(0.0, 360.0)), region)

    assert subset["longitude"].values.tolist() == list(range(0, 11)) + list(range(350, 360))


def test_region_longitude_box_crossing_the_dateline_selects_cells_on_both_grid_conventions() -> None:
    region = oceanbench.regions.custom("dateline", "Dateline", -1.0, 1.0, 170.0, -170.0)

    positive_subset = oceanbench.regions.subset(_global_grid(numpy.arange(0.0, 360.0)), region)
    signed_subset = oceanbench.regions.subset(_global_grid(numpy.arange(-180.0, 180.0)), region)

    assert positive_subset["longitude"].values.tolist() == list(range(170, 191))
    assert signed_subset["longitude"].values.tolist() == list(range(-180, -169)) + list(range(170, 180))


def test_padded_region_widens_every_side_and_keeps_global_unbounded() -> None:
    padded = padded_region(IBI, 10.0)

    assert padded.bounds.minimum_latitude == IBI.bounds.minimum_latitude - 10.0
    assert padded.bounds.maximum_latitude == IBI.bounds.maximum_latitude + 10.0
    assert padded.bounds.minimum_longitude == IBI.bounds.minimum_longitude - 10.0
    assert padded.bounds.maximum_longitude == IBI.bounds.maximum_longitude + 10.0
    assert padded_region("global", 10.0).bounds is None
