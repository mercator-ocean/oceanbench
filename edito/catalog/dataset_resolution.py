# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from itertools import chain, groupby

from bucket import BucketObject, list_bucket_objects, with_last_modified
from dataset_metadata import (
    CopernicusMarineDataset,
    DatasetMetadata,
    MetadataUnavailable,
    bucket_objects_metadata,
    copernicus_marine_datasets,
)
from datasets import VARIANT_GROUP, BucketObjects, CopernicusMarineDatasets, DatasetDescription

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


@dataclass(frozen=True)
class _MatchedObjects:
    description: DatasetDescription
    objects: tuple[BucketObject, ...]


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
        case BucketObjects():
            return tuple(map(_bucket_dataset, _matched_bucket_objects(description)))
        case CopernicusMarineDatasets(dataset_identifiers=dataset_identifiers):
            return (CopernicusMarineCatalogDataset(description, copernicus_marine_datasets(dataset_identifiers)),)


def _bucket_dataset(matched: _MatchedObjects) -> BucketDataset:
    objects = (*matched.objects[:-1], with_last_modified(matched.objects[-1]))
    return BucketDataset(matched.description, objects, bucket_objects_metadata(objects))


def _matched_bucket_objects(description: DatasetDescription) -> tuple[_MatchedObjects, ...]:
    source = description.source
    pattern = re.compile(source.relative_key_pattern)
    variant_and_objects = tuple(
        (_variant(match), bucket_object)
        for bucket_object in list_bucket_objects(source.prefix, source.depth)
        if (match := pattern.fullmatch(bucket_object.key.removeprefix(source.prefix))) is not None
    )
    return tuple(
        _MatchedObjects(_variant_description(description, variant), tuple(bucket_object for _, bucket_object in group))
        for variant, group in groupby(sorted(variant_and_objects, key=_variant_key), key=_variant_key)
    )


def _variant(match: re.Match) -> str | None:
    return match.groupdict().get(VARIANT_GROUP)


def _variant_key(variant_and_object: tuple[str | None, BucketObject]) -> str:
    return variant_and_object[0] or ""


def _variant_description(description: DatasetDescription, variant: str | None) -> DatasetDescription:
    if variant is None:
        return description
    return replace(
        description,
        identifier=description.identifier.format(variant=_identifier_part(variant)),
        title=description.title.format(variant=variant),
    )


def _identifier_part(variant: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", variant.lower()).strip("-")
