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

from bucket import count_bucket_objects, list_dataset_directories
from catalog_health import (
    BucketObjectCount,
    CatalogHealth,
    CollectionChanges,
    UnreadableDataset,
    collection_changes,
    health_metadata,
    public_health_metadata,
    reusable_bucket_object_count,
    stored_bucket_object_count,
    unreadable_datasets,
)
from dataset_metadata import MetadataUnavailable
from dataset_resolution import BucketDataset, ResolvedDataset, resolved_datasets
from datasets import dataset_descriptions
from library_usage import library_usage
from edito_data_api import (
    EditoSession,
    delete_item,
    ensure_project_catalog,
    item_identifiers,
    project_catalog,
    session_from_environment,
    upsert_collection_in_project_catalog,
    upsert_item,
    upsert_project_catalog,
)
from stac import (
    HEALTH_PROPERTY,
    CatalogAudience,
    CatalogEnvironment,
    catalog_datasets,
    catalog_identifier,
    item_identifier,
    stac_catalog,
    stac_collection,
    stac_item,
)

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
    environment: CatalogEnvironment
    collections: tuple[CollectionTree, ...]
    unreadable_datasets: tuple[UnreadableDataset, ...]
    bucket_object_count: BucketObjectCount | None
    publication_datetime: datetime

    def health(self, changes: tuple[CollectionChanges, ...]) -> CatalogHealth:
        return CatalogHealth(
            updated=self.publication_datetime,
            item_count=sum(len(tree.items) for tree in self.collections),
            bucket_object_count=self.bucket_object_count,
            changes=changes,
            unreadable_datasets=self.unreadable_datasets,
        )

    def catalog(self, changes: tuple[CollectionChanges, ...] | None = None) -> dict:
        health = self.health(changes or ())
        shown_health = (
            public_health_metadata(health)
            if self.environment.audience == CatalogAudience.PUBLIC
            else health_metadata(health)
        )
        return stac_catalog(self.environment, shown_health)


def main() -> None:
    """
    Resolve every OceanBench dataset, then publish the internal and the public catalogs on EDITO or write them in a
    local directory.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    arguments = _parsed_arguments()
    environment = CatalogEnvironment(os.environ.get(CATALOG_ENVIRONMENT_VARIABLE) or None)
    datasets = resolved_datasets(dataset_descriptions(list_dataset_directories(), library_usage()))
    _log_unavailable_metadata(datasets)
    publication_datetime = datetime.now(timezone.utc)
    match arguments.output_directory:
        case None:
            session = session_from_environment()
            bucket_object_count = _bucket_object_count(session, environment, publication_datetime)
            catalog_trees = _catalog_trees(datasets, environment, bucket_object_count, publication_datetime)
            tuple(_publish(session, catalog_tree) for catalog_tree in catalog_trees)
        case output_directory:
            catalog_trees = _catalog_trees(datasets, environment, None, publication_datetime)
            tuple(_write(catalog_tree, Path(output_directory)) for catalog_tree in catalog_trees)


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


def _bucket_object_count(session: EditoSession, environment: CatalogEnvironment, now: datetime) -> BucketObjectCount:
    internal_catalog = project_catalog(session, catalog_identifier(environment.with_audience(CatalogAudience.INTERNAL)))
    reusable = reusable_bucket_object_count(stored_bucket_object_count(internal_catalog), now)
    if reusable is not None:
        LOGGER.info("Reusing the count of %d bucket objects made on %s", reusable.count, reusable.counted)
        return reusable
    count = BucketObjectCount(count_bucket_objects(), datetime.now(timezone.utc))
    LOGGER.info("Counted %d bucket objects", count.count)
    return count


def _catalog_trees(
    datasets: tuple[ResolvedDataset, ...],
    environment: CatalogEnvironment,
    bucket_object_count: BucketObjectCount | None,
    publication_datetime: datetime,
) -> tuple[CatalogTree, ...]:
    internal_environment = environment.with_audience(CatalogAudience.INTERNAL)
    unreadable = unreadable_datasets(
        datasets, tuple(item_identifier(dataset, internal_environment) for dataset in datasets)
    )
    return tuple(
        _catalog_tree(
            catalog_datasets(datasets, audience_environment),
            audience_environment,
            unreadable,
            bucket_object_count,
            publication_datetime,
        )
        for audience_environment in (internal_environment, environment.with_audience(CatalogAudience.PUBLIC))
    )


def _catalog_tree(
    datasets: tuple[ResolvedDataset, ...],
    environment: CatalogEnvironment,
    unreadable: tuple[UnreadableDataset, ...],
    bucket_object_count: BucketObjectCount | None,
    publication_datetime: datetime,
) -> CatalogTree:
    collections = tuple(
        _collection_tree(collection, collection_datasets, environment, publication_datetime)
        for collection in sorted({dataset.description.collection for dataset in datasets})
        if (collection_datasets := _datasets_of(collection, datasets))
    )
    _require_unique_item_identifiers(collections)
    return CatalogTree(
        environment=environment,
        collections=collections,
        unreadable_datasets=unreadable,
        bucket_object_count=bucket_object_count,
        publication_datetime=publication_datetime,
    )


def _datasets_of(collection: str, datasets: tuple[ResolvedDataset, ...]) -> tuple[ResolvedDataset, ...]:
    return tuple(dataset for dataset in datasets if dataset.description.collection == collection)


def _collection_tree(
    collection: str,
    datasets: tuple[ResolvedDataset, ...],
    environment: CatalogEnvironment,
    publication_datetime: datetime,
) -> CollectionTree:
    return CollectionTree(
        collection=stac_collection(collection, datasets, environment),
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


def _publish(session: EditoSession, catalog_tree: CatalogTree) -> None:
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
    catalog = catalog_tree.catalog()
    catalog_directory = output_directory / catalog["id"]
    catalog_directory.mkdir(parents=True, exist_ok=True)
    _write_json(catalog_directory / "catalog.json", catalog)
    for tree in catalog_tree.collections:
        collection_directory = catalog_directory / tree.collection["id"]
        collection_directory.mkdir(exist_ok=True)
        _write_json(collection_directory / "collection.json", tree.collection)
        for item in tree.items:
            _write_json(collection_directory / f"{item['id']}.json", item)
    LOGGER.info("Wrote catalog %s in %s", catalog["id"], catalog_directory)


def _write_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
