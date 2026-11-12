# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import re
from collections import Counter
from hashlib import sha1
from dataclasses import dataclass, replace
from enum import Enum

from library_usage import BucketUsage, CopernicusMarineUsage, LibraryUsage as LibraryUsages

COPERNICUS_MARINE_COLLECTION = "copernicus-marine"
SHORT_HASH_LENGTH = 6


class LibraryUsage(Enum):
    USED = "used"
    UNUSED = "unused"


@dataclass(frozen=True)
class BucketObjects:
    prefix: str


@dataclass(frozen=True)
class CopernicusMarineDatasets:
    dataset_identifiers: tuple[str, ...]


@dataclass(frozen=True)
class DatasetDescription:
    identifier: str
    title: str
    collection: str
    source: BucketObjects | CopernicusMarineDatasets
    library_usage: LibraryUsage
    library_functions: tuple[str, ...]


def dataset_descriptions(
    dataset_directories: tuple[str, ...], library_usages: LibraryUsages
) -> tuple[DatasetDescription, ...]:
    descriptions = (
        *(_bucket_description(directory, library_usages.bucket) for directory in dataset_directories),
        *(_copernicus_marine_description(usage) for usage in library_usages.copernicus_marine),
    )
    identifier_counts = Counter(description.identifier for description in descriptions)
    return tuple(_unambiguous(description, identifier_counts) for description in descriptions)


def _unambiguous(description: DatasetDescription, identifier_counts: Counter) -> DatasetDescription:
    if identifier_counts[description.identifier] == 1:
        return description
    return replace(description, identifier=f"{description.identifier}-{_short_hash(description.title)}")


def _short_hash(text: str) -> str:
    return sha1(text.encode()).hexdigest()[:SHORT_HASH_LENGTH]


def _bucket_description(directory: str, usages: tuple[BucketUsage, ...]) -> DatasetDescription:
    using = tuple(usage for usage in usages if _refers_to(usage.path, directory))
    return DatasetDescription(
        identifier=_identifier_part(directory),
        title=directory.rstrip("/"),
        collection=_identifier_part(directory.split("/")[0]),
        source=BucketObjects(directory),
        library_usage=LibraryUsage.USED if using else LibraryUsage.UNUSED,
        library_functions=tuple(sorted({function for usage in using for function in usage.public_functions})),
    )


def _refers_to(path: str, directory: str) -> bool:
    return f"{path}/" == directory or (path.startswith(directory) and "/" not in path.removeprefix(directory))


def _copernicus_marine_description(usage: CopernicusMarineUsage) -> DatasetDescription:
    return DatasetDescription(
        identifier=_identifier_part(usage.dataset_identifier),
        title=usage.dataset_identifier,
        collection=COPERNICUS_MARINE_COLLECTION,
        source=CopernicusMarineDatasets((usage.dataset_identifier,)),
        library_usage=LibraryUsage.USED,
        library_functions=usage.public_functions,
    )


def _identifier_part(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
