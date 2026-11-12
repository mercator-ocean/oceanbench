# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import chain

from bucket import BucketObject, list_dataset_objects, with_last_modified
from dataset_metadata import (
    CopernicusMarineDataset,
    DatasetMetadata,
    MetadataUnavailable,
    bucket_objects_metadata,
    copernicus_marine_datasets,
)
from datasets import BucketObjects, CopernicusMarineDatasets, DatasetDescription

MAXIMUM_PARALLEL_METADATA_READS = 8
LOGGER = logging.getLogger("oceanbench-catalog")


@dataclass(frozen=True)
class BucketDataset:
    description: DatasetDescription
    objects: tuple[BucketObject, ...]
    metadata: DatasetMetadata | MetadataUnavailable


@dataclass(frozen=True)
class CopernicusMarineCatalogDataset:
    description: DatasetDescription
    datasets: tuple[CopernicusMarineDataset, ...] | MetadataUnavailable


ResolvedDataset = BucketDataset | CopernicusMarineCatalogDataset


def resolved_datasets(descriptions: tuple[DatasetDescription, ...]) -> tuple[ResolvedDataset, ...]:
    with ThreadPoolExecutor(max_workers=MAXIMUM_PARALLEL_METADATA_READS) as executor:
        return tuple(
            chain.from_iterable(
                executor.map(_resolved_description_datasets, descriptions),
            )
        )


def _resolved_description_datasets(description: DatasetDescription) -> tuple[ResolvedDataset, ...]:
    start = time.monotonic()
    datasets = _resolved_source_datasets(description)
    LOGGER.info("Resolved %s in %.1f s", description.identifier, time.monotonic() - start)
    return datasets


def _resolved_source_datasets(description: DatasetDescription) -> tuple[ResolvedDataset, ...]:
    match description.source:
        case BucketObjects(prefix=prefix):
            return (_bucket_dataset(description, list_dataset_objects(prefix)),)
        case CopernicusMarineDatasets(dataset_identifiers=dataset_identifiers):
            return (CopernicusMarineCatalogDataset(description, copernicus_marine_datasets(dataset_identifiers)),)


def _bucket_dataset(description: DatasetDescription, dataset_objects: tuple[BucketObject, ...]) -> BucketDataset:
    objects = (*dataset_objects[:-1], with_last_modified(dataset_objects[-1]))
    return BucketDataset(description, objects, bucket_objects_metadata(objects))
