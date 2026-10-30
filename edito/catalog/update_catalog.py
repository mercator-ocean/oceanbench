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

from bucket import list_bucket_directories
from catalog_health import (
    CatalogHealth,
    CollectionChanges,
    UnreadableDataset,
    collection_changes,
    health_metadata,
    unmapped_prefixes,
    unreadable_datasets,
)
from dataset_metadata import MetadataUnavailable
from dataset_resolution import BucketDataset, ResolvedDataset, resolved_datasets
from datasets import DATASET_DESCRIPTIONS, NOT_DATASET_PREFIXES, CollectionKind
from edito_data_api import (
    delete_item,
    ensure_project_catalog,
    item_identifiers,
    session_from_environment,
    upsert_collection_in_project_catalog,
    upsert_item,
    upsert_project_catalog,
)
from stac import HEALTH_PROPERTY, CatalogEnvironment, item_identifier, stac_catalog, stac_collection, stac_item

CATALOG_ENVIRONMENT_VARIABLE = "OCEANBENCH_CATALOG_ENVIRONMENT"
UNMAPPED_PREFIX_SEARCH_DEPTH = 3
LOGGER = logging.getLogger("oceanbench-catalog")


class DuplicateItemIdentifiersError(ValueError):
    pass


@dataclass(frozen=True)
class CollectionTree:
    collection: dict
    items: tuple[dict, ...]


@dataclass(frozen=True)
class CatalogTree:
    environment: CatalogEnvironment
    collections: tuple[CollectionTree, ...]
    unreadable_datasets: tuple[UnreadableDataset, ...]
    unmapped_prefixes: tuple[str, ...]
    publication_datetime: datetime

    def health(self, changes: tuple[CollectionChanges, ...]) -> CatalogHealth:
        return CatalogHealth(
            updated=self.publication_datetime,
            item_count=sum(len(tree.items) for tree in self.collections),
            changes=changes,
            unreadable_datasets=self.unreadable_datasets,
            unmapped_prefixes=self.unmapped_prefixes,
        )

    def catalog(self, changes: tuple[CollectionChanges, ...] | None = None) -> dict:
        return stac_catalog(self.environment, health_metadata(self.health(changes or ())))


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
        environment=environment,
        collections=collections,
        unreadable_datasets=unreadable_datasets(
            datasets, tuple(item_identifier(dataset, environment) for dataset in datasets)
        ),
        unmapped_prefixes=unmapped_prefixes(
            list_bucket_directories("", UNMAPPED_PREFIX_SEARCH_DEPTH), DATASET_DESCRIPTIONS, NOT_DATASET_PREFIXES
        ),
        publication_datetime=publication_datetime,
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
    catalog_identifier = catalog_tree.catalog()["id"]
    ensure_project_catalog(session, stac_catalog(catalog_tree.environment))
    changes = tuple(_publish_collection(session, catalog_identifier, tree) for tree in catalog_tree.collections)
    catalog = catalog_tree.catalog(changes)
    upsert_project_catalog(session, catalog, catalog_tree.environment.visibility().value)
    LOGGER.info("Published catalog %s: %s", catalog_identifier, json.dumps(catalog[HEALTH_PROPERTY]))


def _publish_collection(session, catalog_identifier: str, tree: CollectionTree) -> CollectionChanges:
    collection_identifier = tree.collection["id"]
    upsert_collection_in_project_catalog(session, catalog_identifier, tree.collection)
    published_identifiers = item_identifiers(session, collection_identifier)
    for item in tree.items:
        upsert_item(session, collection_identifier, item)
        LOGGER.info("Published item %s", item["id"])
    changes = collection_changes(collection_identifier, tuple(item["id"] for item in tree.items), published_identifiers)
    for stale_identifier in changes.deleted_items:
        delete_item(session, collection_identifier, stale_identifier)
        LOGGER.info("Deleted stale item %s", stale_identifier)
    for withheld_identifier in changes.withheld_deletions:
        LOGGER.warning("Withheld deletion of stale item %s", withheld_identifier)
    LOGGER.info("Published collection %s with %d items", collection_identifier, len(tree.items))
    return changes


def _write(catalog_tree: CatalogTree, output_directory: Path) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_json(output_directory / "catalog.json", catalog_tree.catalog())
    for tree in catalog_tree.collections:
        collection_directory = output_directory / tree.collection["id"]
        collection_directory.mkdir(exist_ok=True)
        _write_json(collection_directory / "collection.json", tree.collection)
        for item in tree.items:
            _write_json(collection_directory / f"{item['id']}.json", item)
    LOGGER.info("Wrote catalog %s in %s", catalog_tree.catalog()["id"], output_directory)


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
