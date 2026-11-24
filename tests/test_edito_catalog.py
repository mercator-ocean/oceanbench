# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import ast
import base64
import sys
from pathlib import Path

REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIRECTORY / "edito" / "catalog"))

from registry import CATALOG_ENTRIES, BucketStores, Collection, CopernicusMarineDatasets  # noqa: E402
from stac import stac_collection, stac_item  # noqa: E402
import numpy  # noqa: E402
import pandas  # noqa: E402
import xarray  # noqa: E402
from store_metadata import SpatialExtent, StoreMetadata, Variable, store_metadata  # noqa: E402

DATASETS_DIRECTORY = REPOSITORY_DIRECTORY / "oceanbench" / "datasets"
EXAMPLE_PREFIX = "data:text/x-python;base64,"
QUARTER_DEGREE_STORE_PATH_FRAGMENTS = ("glorys14", "glo14")


def _library_functions() -> set[str]:
    return {
        f"oceanbench.datasets.{module.stem}.{node.name}"
        for module in DATASETS_DIRECTORY.glob("*.py")
        for node in ast.parse(module.read_text()).body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }


def _example_snippets(item: dict) -> list[str]:
    return [
        base64.b64decode(link["href"].removeprefix(EXAMPLE_PREFIX)).decode()
        for link in item["links"]
        if link["rel"] == "example"
    ]


def _referenced_urls(item: dict) -> list[str]:
    return [asset["href"] for asset in item["assets"].values()] + [link["href"] for link in item["links"]]


def _points_to_quarter_degree_store(item: dict) -> bool:
    return any(fragment in url for url in _referenced_urls(item) for fragment in QUARTER_DEGREE_STORE_PATH_FRAGMENTS)


def test_catalog_lists_every_library_dataset_function():
    listed = {function for entry in CATALOG_ENTRIES for function in entry.library_functions}
    assert _library_functions() - listed == set()
    assert listed - _library_functions() == set()


def test_catalog_item_identifiers_are_unique():
    identifiers = [entry.identifier for entry in CATALOG_ENTRIES]
    assert len(identifiers) == len(set(identifiers))


def test_bucket_item_describes_store_extent_variables_and_how_to_open_it():
    entry = next(entry for entry in CATALOG_ENTRIES if entry.identifier == "challenger-glonet")
    metadata = StoreMetadata(
        spatial_extent=SpatialExtent(west=-180.0, south=-78.0, east=179.75, north=90.0),
        variables=(
            Variable(
                "thetao",
                ("lead_day_index", "depth", "lat", "lon"),
                "degrees_C",
                "Temperature",
                "sea_water_potential_temperature",
            ),
        ),
        groups=(),
    )
    item = stac_item(entry, metadata)
    assert item["bbox"] == [-180.0, -78.0, 179.75, 90.0]
    assert item["properties"]["start_datetime"] == "2024-01-03T00:00:00Z"
    assert item["properties"]["cube:variables"]["thetao"]["unit"] == "degrees_C"
    assert "sea_water_potential_temperature" in item["properties"]["keywords"]
    assert item["assets"]["sample"]["href"].endswith("public/ml-forecast-outputs/glonet/20240103.zarr")
    assert _example_snippets(item) == [
        "import oceanbench\n\n"
        "dataset = oceanbench.datasets.challenger.glonet()\n"
        "dataset_1_degree = oceanbench.datasets.challenger.glonet_1_degree()\n"
    ]


def test_grouped_store_without_library_function_is_opened_by_merging_its_groups():
    entry = next(
        entry for entry in CATALOG_ENTRIES if not entry.library_functions and isinstance(entry.source, BucketStores)
    )
    metadata = StoreMetadata(SpatialExtent(-180.0, -90.0, 180.0, 90.0), (), ("surface", "depth"))
    [snippet] = _example_snippets(stac_item(entry, metadata))
    assert "group=group) for group in ['surface', 'depth']" in snippet


def test_collection_temporal_extent_spans_its_items():
    entries = tuple(entry for entry in CATALOG_ENTRIES if entry.collection == Collection.TRAINING)
    collection = stac_collection(Collection.TRAINING, entries)
    assert collection["extent"]["temporal"]["interval"][0][0].startswith("1993-01-0")
    assert collection["extent"]["temporal"]["interval"][0][1] == "2019-12-31T00:00:00Z"


def test_store_on_zero_to_360_longitudes_is_described_as_global_datacube(tmp_path):
    store = str(tmp_path / "forcings.zarr")
    xarray.Dataset(
        {"u10": (("time", "lat", "lon"), numpy.zeros((2, 3, 4)), {"units": "m s-1"})},
        coords={
            "time": pandas.date_range("2024-01-01", periods=2),
            "lat": [-90.0, 0.0, 90.0],
            "lon": [0.0, 90.0, 180.0, 270.0],
        },
    ).to_zarr(store, consolidated=True)
    entry = next(entry for entry in CATALOG_ENTRIES if isinstance(entry.source, BucketStores))
    item = stac_item(entry, store_metadata(store))
    assert item["bbox"] == [-180.0, -90.0, 90.0, 90.0]
    assert item["properties"]["cube:dimensions"] == {
        "time": {"type": "temporal", "extent": ["2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"]},
        "lat": {"type": "spatial", "axis": "y", "extent": [-90.0, 90.0], "reference_system": 4326},
        "lon": {"type": "spatial", "axis": "x", "extent": [-180.0, 90.0], "reference_system": 4326},
    }


def test_every_item_has_assets_and_a_time_range():
    items = tuple(stac_item(entry, None) for entry in CATALOG_ENTRIES)
    assert all(item["assets"] for item in items)
    assert all(item["properties"]["start_datetime"] and item["properties"]["end_datetime"] for item in items)


def test_no_catalog_item_points_to_quarter_degree_store():
    items = tuple(stac_item(entry, None) for entry in CATALOG_ENTRIES)
    assert [item["id"] for item in items if _points_to_quarter_degree_store(item)] == []


def _assigned_names(snippet: str) -> list[str]:
    return [
        target.id
        for node in ast.walk(ast.parse(snippet))
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]


def test_no_example_overwrites_a_dataset_it_opened():
    items = tuple(stac_item(entry, None) for entry in CATALOG_ENTRIES)
    snippets = tuple(snippet for item in items for snippet in _example_snippets(item))
    assert all(len(_assigned_names(snippet)) == len(set(_assigned_names(snippet))) for snippet in snippets)


def test_copernicus_marine_references_span_the_benchmark_year():
    items = tuple(
        stac_item(entry, None)
        for entry in CATALOG_ENTRIES
        if entry.collection == Collection.REFERENCES and isinstance(entry.source, CopernicusMarineDatasets)
    )
    assert {(item["properties"]["start_datetime"], item["properties"]["end_datetime"]) for item in items} == {
        ("2024-01-01T00:00:00Z", "2024-12-31T00:00:00Z")
    }
