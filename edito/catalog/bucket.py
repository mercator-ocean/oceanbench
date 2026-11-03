# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import datetime
from email.utils import parsedate_to_datetime
from enum import Enum
from functools import cache, partial, reduce
from itertools import chain
from xml.etree import ElementTree

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BUCKET_NAME = "oceanbench-bucket"
BUCKET_URL = f"https://s3.waw3-1.cloudferro.com/{BUCKET_NAME}"
ZARR_SUFFIX = ".zarr"
ZARR_METADATA_FILE_NAME = ".zmetadata"
LISTING_PAGE_SIZE = 1000
LISTING_TIMEOUT_SECONDS = 60
LISTING_RETRIES = 5
LISTING_RETRY_BACKOFF_SECONDS = 1
LISTING_RETRIED_STATUSES = (500, 502, 503, 504)
OBJECT_COUNT_SPLIT_DEPTH = 4
OBJECT_COUNT_PARALLEL_LISTINGS = 32
_OBJECT_KEY_TAG = b"<Key>"
_TRUNCATED_TAG = b"<IsTruncated>true</IsTruncated>"
_NEXT_TOKEN_PATTERN = re.compile(rb"<NextContinuationToken>([^<]*)</NextContinuationToken>")
_S3_NAMESPACE = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


class ObjectKind(Enum):
    ZARR_STORE = "zarr_store"
    FILE = "file"


@dataclass(frozen=True)
class BucketObject:
    key: str
    kind: ObjectKind
    size_in_bytes: int | None
    last_modified: datetime | None


@dataclass(frozen=True)
class _ListingPage:
    directory_keys: tuple[str, ...]
    files: tuple[BucketObject, ...]
    next_continuation_token: str | None


@cache
def _session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(
        pool_maxsize=OBJECT_COUNT_PARALLEL_LISTINGS,
        max_retries=Retry(
            total=LISTING_RETRIES,
            backoff_factor=LISTING_RETRY_BACKOFF_SECONDS,
            status_forcelist=LISTING_RETRIED_STATUSES,
        ),
    )
    session.mount("https://", adapter)
    return session


def public_url(key: str) -> str:
    return f"{BUCKET_URL}/{key}"


def s3_uri(key: str) -> str:
    return f"s3://{BUCKET_NAME}/{key}"


def listing_url(prefix: str) -> str:
    return f"{BUCKET_URL}?list-type=2&delimiter=/&prefix={prefix}"


def with_last_modified(bucket_object: BucketObject) -> BucketObject:
    if bucket_object.last_modified is not None or bucket_object.kind != ObjectKind.ZARR_STORE:
        return bucket_object
    response = _session().head(
        public_url(f"{bucket_object.key}/{ZARR_METADATA_FILE_NAME}"), timeout=LISTING_TIMEOUT_SECONDS
    )
    last_modified = response.headers.get("Last-Modified") if response.ok else None
    return replace(
        bucket_object,
        last_modified=parsedate_to_datetime(last_modified) if last_modified else None,
    )


@cache
def list_bucket_objects(prefix: str, depth: int) -> tuple[BucketObject, ...]:
    directory_keys, files = _list_directory(prefix)
    stores = tuple(_zarr_store(key) for key in directory_keys if _is_zarr_store(key))
    subdirectory_objects = (
        chain.from_iterable(list_bucket_objects(key, depth - 1) for key in directory_keys if not _is_zarr_store(key))
        if depth > 0
        else ()
    )
    return tuple(sorted((*stores, *files, *subdirectory_objects), key=lambda bucket_object: bucket_object.key))


@cache
def list_bucket_directories(prefix: str, depth: int) -> tuple[str, ...]:
    directory_keys = tuple(key for key in _list_directory(prefix)[0] if not _is_zarr_store(key))
    subdirectory_keys = (
        chain.from_iterable(list_bucket_directories(key, depth - 1) for key in directory_keys) if depth > 1 else ()
    )
    return tuple(sorted((*directory_keys, *subdirectory_keys)))


def _is_zarr_store(directory_key: str) -> bool:
    return directory_key.rstrip("/").endswith(ZARR_SUFFIX)


def _zarr_store(directory_key: str) -> BucketObject:
    return BucketObject(
        key=directory_key.rstrip("/"), kind=ObjectKind.ZARR_STORE, size_in_bytes=None, last_modified=None
    )


@cache
def _list_directory(prefix: str) -> tuple[tuple[str, ...], tuple[BucketObject, ...]]:
    pages = tuple(_listing_pages(prefix))
    return (
        tuple(chain.from_iterable(page.directory_keys for page in pages)),
        tuple(chain.from_iterable(page.files for page in pages)),
    )


def _listing_pages(prefix: str, continuation_token: str | None = None) -> Iterator[_ListingPage]:
    page = _listing_page(prefix, continuation_token)
    yield page
    if page.next_continuation_token is not None:
        yield from _listing_pages(prefix, page.next_continuation_token)


def _listing_page(prefix: str, continuation_token: str | None) -> _ListingPage:
    response = _session().get(
        BUCKET_URL,
        params={
            "list-type": "2",
            "delimiter": "/",
            "prefix": prefix,
            "max-keys": str(LISTING_PAGE_SIZE),
            **({"continuation-token": continuation_token} if continuation_token else {}),
        },
        timeout=LISTING_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    return _ListingPage(
        directory_keys=tuple(element.text for element in root.findall("s3:CommonPrefixes/s3:Prefix", _S3_NAMESPACE)),
        files=tuple(_file(element) for element in root.findall("s3:Contents", _S3_NAMESPACE)),
        next_continuation_token=(
            _text(root, "s3:NextContinuationToken") if _text(root, "s3:IsTruncated") == "true" else None
        ),
    )


def count_bucket_objects() -> int:
    """
    Count every object of the bucket, listing in parallel the directories down to the Zarr stores.
    """
    with ThreadPoolExecutor(max_workers=OBJECT_COUNT_PARALLEL_LISTINGS) as pool:
        files_count, prefixes = reduce(partial(_split_directories, pool), range(OBJECT_COUNT_SPLIT_DEPTH), (0, ("",)))
        return files_count + sum(pool.map(_count_prefix_objects, prefixes))


def _split_directories(
    pool: ThreadPoolExecutor, counted_and_prefixes: tuple[int, tuple[str, ...]], _level: int
) -> tuple[int, tuple[str, ...]]:
    counted_files_count, prefixes = counted_and_prefixes
    listings = tuple(pool.map(_list_directory, prefixes))
    return (
        counted_files_count + sum(len(files) for _, files in listings),
        tuple(chain.from_iterable(directory_keys for directory_keys, _ in listings)),
    )


def _count_prefix_objects(prefix: str) -> int:
    return sum(_flat_page_object_counts(prefix))


def _flat_page_object_counts(prefix: str) -> Iterator[int]:
    continuation_token = None
    while True:
        response = _session().get(
            BUCKET_URL,
            params={
                "list-type": "2",
                "prefix": prefix,
                "max-keys": str(LISTING_PAGE_SIZE),
                **({"continuation-token": continuation_token} if continuation_token else {}),
            },
            timeout=LISTING_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        yield response.content.count(_OBJECT_KEY_TAG)
        next_token = _NEXT_TOKEN_PATTERN.search(response.content) if _TRUNCATED_TAG in response.content else None
        if next_token is None:
            return
        continuation_token = next_token.group(1).decode()


def _file(element: ElementTree.Element) -> BucketObject:
    return BucketObject(
        key=_text(element, "s3:Key"),
        kind=ObjectKind.FILE,
        size_in_bytes=int(_text(element, "s3:Size")),
        last_modified=datetime.fromisoformat(_text(element, "s3:LastModified")),
    )


def _text(element: ElementTree.Element, path: str) -> str | None:
    found = element.find(path, _S3_NAMESPACE)
    return found.text if found is not None else None
