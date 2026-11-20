# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Create or update the public OceanBench STAC catalog on EDITO from the catalog registry.
"""

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from edito_data_api import (
    EditoSession,
    delete_item,
    item_identifiers,
    session_from_environment,
    upsert_collection_in_project_catalog,
    upsert_item,
    upsert_project_catalog,
)
from registry import CATALOG_ENTRIES, BucketStores, CatalogEntry, Collection, bucket_url
from stac import CATALOG_IDENTIFIER, Visibility, stac_catalog, stac_collection, stac_item
from store_metadata import StoreMetadata, store_metadata

LOGGER = logging.getLogger("oceanbench-catalog")


@dataclass(frozen=True)
class CollectionTree:
    collection: dict
    items: tuple[dict, ...]


def catalog_trees() -> tuple[CollectionTree, ...]:
    return tuple(map(_collection_tree, Collection))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = _parsed_arguments()
    trees = catalog_trees()
    match arguments.output_directory:
        case None:
            _publish(session_from_environment(), trees)
        case output_directory:
            _write(trees, Path(output_directory))


def _parsed_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", help="Write the STAC JSON files there instead of publishing them.")
    return parser.parse_args()


def _collection_tree(collection: Collection) -> CollectionTree:
    entries = tuple(entry for entry in CATALOG_ENTRIES if entry.collection == collection)
    return CollectionTree(
        collection=stac_collection(collection, entries),
        items=tuple(stac_item(entry, _entry_metadata(entry)) for entry in entries),
    )


def _entry_metadata(entry: CatalogEntry) -> StoreMetadata | None:
    match entry.source:
        case BucketStores(sample_path=sample_path):
            LOGGER.info("Read %s", sample_path)
            return store_metadata(bucket_url(sample_path))
        case _:
            return None


def _publish(session: EditoSession, trees: tuple[CollectionTree, ...]) -> None:
    upsert_project_catalog(session, stac_catalog(), Visibility.PUBLIC.value)
    tuple(_publish_collection(session, tree) for tree in trees)


def _publish_collection(session: EditoSession, tree: CollectionTree) -> None:
    identifier = tree.collection["id"]
    LOGGER.info("Publish %s", identifier)
    upsert_collection_in_project_catalog(session, CATALOG_IDENTIFIER, tree.collection)
    tuple(upsert_item(session, identifier, item) for item in tree.items)
    published = {item["id"] for item in tree.items}
    tuple(
        delete_item(session, identifier, stale)
        for stale in item_identifiers(session, identifier)
        if stale not in published
    )


def _write(trees: tuple[CollectionTree, ...], output_directory: Path) -> None:
    _write_json(output_directory / "catalog.json", stac_catalog())
    tuple(_write_json(output_directory / f"{tree.collection['id']}.json", tree.collection) for tree in trees)
    tuple(
        _write_json(output_directory / tree.collection["id"] / f"{item['id']}.json", item)
        for tree in trees
        for item in tree.items
    )


def _write_json(path: Path, content: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content, indent=2) + "\n")


if __name__ == "__main__":
    main()
