# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from itertools import chain

from dataset_metadata import MetadataUnavailable, UnavailabilityCause
from dataset_resolution import BucketDataset, CopernicusMarineCatalogDataset, ResolvedDataset
from datasets import BucketObjects, DatasetDescription

MAXIMUM_DELETIONS_PER_COLLECTION = 3


class HealthStatus(Enum):
    HEALTHY = "healthy"
    NEEDS_ATTENTION = "needs_attention"


@dataclass(frozen=True)
class CollectionChanges:
    collection_identifier: str
    added_items: tuple[str, ...]
    deleted_items: tuple[str, ...]
    withheld_deletions: tuple[str, ...]


@dataclass(frozen=True)
class UnreadableDataset:
    item_identifier: str
    reason: str


@dataclass(frozen=True)
class CatalogHealth:
    updated: datetime
    item_count: int
    changes: tuple[CollectionChanges, ...]
    unreadable_datasets: tuple[UnreadableDataset, ...]
    unmapped_prefixes: tuple[str, ...]

    def status(self) -> HealthStatus:
        needs_attention = (
            any(change.withheld_deletions for change in self.changes)
            or self.unreadable_datasets
            or self.unmapped_prefixes
        )
        return HealthStatus.NEEDS_ATTENTION if needs_attention else HealthStatus.HEALTHY


def collection_changes(
    collection_identifier: str, current_items: tuple[str, ...], published_items: tuple[str, ...]
) -> CollectionChanges:
    stale_items = tuple(sorted(set(published_items) - set(current_items)))
    withheld = len(stale_items) > MAXIMUM_DELETIONS_PER_COLLECTION
    return CollectionChanges(
        collection_identifier=collection_identifier,
        added_items=tuple(sorted(set(current_items) - set(published_items))),
        deleted_items=() if withheld else stale_items,
        withheld_deletions=stale_items if withheld else (),
    )


def unreadable_datasets(
    datasets: tuple[ResolvedDataset, ...], item_identifiers: tuple[str, ...]
) -> tuple[UnreadableDataset, ...]:
    return tuple(
        UnreadableDataset(item_identifier, unavailable.reason)
        for dataset, item_identifier in zip(datasets, item_identifiers)
        if (unavailable := _read_failure(dataset)) is not None
    )


def _read_failure(dataset: ResolvedDataset) -> MetadataUnavailable | None:
    match dataset:
        case BucketDataset(metadata=MetadataUnavailable(cause=UnavailabilityCause.READ_FAILURE) as unavailable):
            return unavailable
        case CopernicusMarineCatalogDataset(
            datasets=MetadataUnavailable(cause=UnavailabilityCause.READ_FAILURE) as unavailable
        ):
            return unavailable
        case _:
            return None


def unmapped_prefixes(
    bucket_directories: tuple[str, ...],
    descriptions: tuple[DatasetDescription, ...],
    ignored_prefixes: tuple[str, ...],
) -> tuple[str, ...]:
    sources = tuple(description.source for description in descriptions if isinstance(description.source, BucketObjects))
    return tuple(
        directory
        for directory in bucket_directories
        if not any(_is_covered(directory, source) for source in sources)
        and not any(directory.startswith(prefix) for prefix in ignored_prefixes)
    )


def _is_covered(directory: str, source: BucketObjects) -> bool:
    if source.prefix.startswith(directory):
        return True
    return directory.startswith(source.prefix) and directory.removeprefix(source.prefix).count("/") <= source.depth


def health_metadata(health: CatalogHealth) -> dict:
    return {
        "status": health.status().value,
        "updated": health.updated.isoformat(),
        "item_count": health.item_count,
        "added_items": list(chain.from_iterable(change.added_items for change in health.changes)),
        "deleted_items": list(chain.from_iterable(change.deleted_items for change in health.changes)),
        "withheld_deletions": list(chain.from_iterable(change.withheld_deletions for change in health.changes)),
        "maximum_deletions_per_collection": MAXIMUM_DELETIONS_PER_COLLECTION,
        "unreadable_datasets": [
            {"item": dataset.item_identifier, "reason": dataset.reason} for dataset in health.unreadable_datasets
        ],
        "unmapped_prefixes": list(health.unmapped_prefixes),
    }
