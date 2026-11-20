# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import ast
import base64
import re
import sys
from pathlib import Path

REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_DIRECTORY / "edito" / "catalog"))

from registry import CATALOG_ENTRIES, BucketStores, Collection, CopernicusMarineDatasets  # noqa: E402
from stac import stac_collection, stac_item  # noqa: E402
from store_metadata import SpatialExtent, StoreMetadata, Variable  # noqa: E402

DATASETS_DIRECTORY = REPOSITORY_DIRECTORY / "oceanbench" / "datasets"
COPERNICUS_MARINE_IDENTIFIER_PATTERN = re.compile(r'"(cmems_[^"]+)"')
EXAMPLE_PREFIX = "data:text/x-python;base64,"


def _library_functions() -> set[str]:
    return {
        f"oceanbench.datasets.{module.stem}.{node.name}"
        for module in DATASETS_DIRECTORY.glob("*.py")
        for node in ast.parse(module.read_text()).body
        if isinstance(node, ast.FunctionDef) and not node.name.startswith("_")
    }


def _library_copernicus_marine_identifiers() -> set[str]:
    return {
        identifier
        for module in (REPOSITORY_DIRECTORY / "oceanbench").rglob("*.py")
        for identifier in COPERNICUS_MARINE_IDENTIFIER_PATTERN.findall(module.read_text())
    }


def _example_snippets(item: dict) -> list[str]:
    return [
        base64.b64decode(link["href"].removeprefix(EXAMPLE_PREFIX)).decode()
        for link in item["links"]
        if link["rel"] == "example"
    ]


def test_catalog_lists_every_library_dataset_function():
    listed = {function for entry in CATALOG_ENTRIES for function in entry.library_functions}
    assert _library_functions() - listed == set()
    assert listed - _library_functions() == set()


def test_catalog_lists_every_copernicus_marine_dataset_used_by_the_library():
    listed = {
        identifier
        for entry in CATALOG_ENTRIES
        if isinstance(entry.source, CopernicusMarineDatasets)
        for identifier in entry.source.dataset_identifiers
    }
    assert _library_copernicus_marine_identifiers() - listed == set()


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


def test_copernicus_marine_item_links_its_product_and_opens_with_the_toolbox():
    entry = next(
        entry
        for entry in CATALOG_ENTRIES
        if isinstance(entry.source, CopernicusMarineDatasets) and not entry.library_functions
    )
    item = stac_item(entry, None)
    assert item["bbox"] == [-180.0, -90.0, 180.0, 90.0]
    assert any(entry.source.product_identifier in link["href"] for link in item["links"] if link["rel"] == "about")
    assert all("copernicusmarine.open_dataset" in snippet for snippet in _example_snippets(item))


def test_collection_temporal_extent_spans_its_items():
    entries = tuple(entry for entry in CATALOG_ENTRIES if entry.collection == Collection.TRAINING)
    collection = stac_collection(Collection.TRAINING, entries)
    assert collection["extent"]["temporal"]["interval"][0][0].startswith("1993-01-0")
    assert collection["extent"]["temporal"]["interval"][0][1] == "2019-12-31T00:00:00Z"
