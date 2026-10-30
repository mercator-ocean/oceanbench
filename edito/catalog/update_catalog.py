# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Create or update the OceanBench STAC catalog on EDITO from the OceanBench bucket and Copernicus Marine.
"""

import argparse
import json
import logging
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from dataset_metadata import MetadataUnavailable
from dataset_resolution import BucketDataset, ResolvedDataset, resolved_datasets
from datasets import DATASET_DESCRIPTIONS, CollectionKind
from edito_data_api import (
    delete_item,
    item_identifiers,
    session_from_environment,
    upsert_collection_in_project_catalog,
    upsert_item,
    upsert_project_catalog,
)
from stac import CatalogEnvironment, stac_catalog, stac_collection, stac_item

CATALOG_ENVIRONMENT_VARIABLE = "OCEANBENCH_CATALOG_ENVIRONMENT"
LOGGER = logging.getLogger("oceanbench-catalog")


class DuplicateItemIdentifiersError(ValueError):
    pass


@dataclass(frozen=True)
class CollectionTree:
    collection: dict
    items: tuple[dict, ...]


@dataclass(frozen=True)
class CatalogTree:
    catalog: dict
    collections: tuple[CollectionTree, ...]
    visibility: tuple[str, ...]


def main() -> None:
    """
    Resolve every OceanBench dataset, then publish the catalog on EDITO or write it in a local directory.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = _parsed_arguments()
    environment = CatalogEnvironment(os.environ.get(CATALOG_ENVIRONMENT_VARIABLE) or None)
    datasets = resolved_datasets(DATASET_DESCRIPTIONS)
    _log_unavailable_metadata(datasets)
    catalog_tree = _catalog_tree(datasets, environment, datetime.now(timezone.utc))
    match arguments.output_directory:
        case None:
            _publish(catalog_tree)
        case output_directory:
            _write(catalog_tree, Path(output_directory))


def _parsed_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        help="Write the STAC JSON files in this directory instead of publishing them on EDITO.",
    )
    return parser.parse_args()


def _log_unavailable_metadata(datasets: tuple[ResolvedDataset, ...]) -> None:
    for dataset in datasets:
        if isinstance(dataset, BucketDataset) and isinstance(dataset.metadata, MetadataUnavailable):
            LOGGER.warning("Metadata unavailable for %s: %s", dataset.description.identifier, dataset.metadata.reason)


def _catalog_tree(
    datasets: tuple[ResolvedDataset, ...], environment: CatalogEnvironment, publication_datetime: datetime
) -> CatalogTree:
    collections = tuple(
        _collection_tree(kind, _datasets_of(kind, datasets), environment, publication_datetime)
        for kind in CollectionKind
    )
    _require_unique_item_identifiers(collections)
    return CatalogTree(
        catalog=stac_catalog(environment),
        collections=collections,
        visibility=environment.visibility().value,
    )


def _datasets_of(kind: CollectionKind, datasets: tuple[ResolvedDataset, ...]) -> tuple[ResolvedDataset, ...]:
    return tuple(dataset for dataset in datasets if dataset.description.collection == kind)


def _collection_tree(
    kind: CollectionKind,
    datasets: tuple[ResolvedDataset, ...],
    environment: CatalogEnvironment,
    publication_datetime: datetime,
) -> CollectionTree:
    return CollectionTree(
        collection=stac_collection(kind, datasets, environment),
        items=tuple(stac_item(dataset, environment, publication_datetime) for dataset in datasets),
    )


def _require_unique_item_identifiers(collections: tuple[CollectionTree, ...]) -> None:
    duplicates = tuple(
        identifier
        for identifier, count in Counter(item["id"] for tree in collections for item in tree.items).items()
        if count > 1
    )
    if duplicates:
        raise DuplicateItemIdentifiersError(f"Duplicate item identifiers: {', '.join(duplicates)}")


def _publish(catalog_tree: CatalogTree) -> None:
    session = session_from_environment()
    catalog_identifier = catalog_tree.catalog["id"]
    upsert_project_catalog(session, catalog_tree.catalog, catalog_tree.visibility)
    for tree in catalog_tree.collections:
        _publish_collection(session, catalog_identifier, tree)
    LOGGER.info("Published catalog %s", catalog_identifier)


def _publish_collection(session, catalog_identifier: str, tree: CollectionTree) -> None:
    collection_identifier = tree.collection["id"]
    upsert_collection_in_project_catalog(session, catalog_identifier, tree.collection)
    for item in tree.items:
        upsert_item(session, collection_identifier, item)
        LOGGER.info("Published item %s", item["id"])
    current_identifiers = {item["id"] for item in tree.items}
    for stale_identifier in set(item_identifiers(session, collection_identifier)) - current_identifiers:
        delete_item(session, collection_identifier, stale_identifier)
        LOGGER.info("Deleted stale item %s", stale_identifier)
    LOGGER.info("Published collection %s with %d items", collection_identifier, len(tree.items))


def _write(catalog_tree: CatalogTree, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_json(output_directory / "catalog.json", catalog_tree.catalog)
    for tree in catalog_tree.collections:
        collection_directory = output_directory / tree.collection["id"]
        collection_directory.mkdir(exist_ok=True)
        _write_json(collection_directory / "collection.json", tree.collection)
        for item in tree.items:
            _write_json(collection_directory / f"{item['id']}.json", item)
    LOGGER.info("Wrote catalog %s in %s", catalog_tree.catalog["id"], output_directory)


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
