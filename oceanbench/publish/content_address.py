# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Content-addressed blob naming for published insight artifacts (contracts.md §4, §8).

Insight blobs are named by the SHA-256 of their bytes so they are immutable and
CDN-cacheable: republishing identical content reuses the same name, and any change
produces a new one. The human-readable identity of a blob lives in the manifest key
that points at it, not in its filename.
"""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil

_HASH_READ_CHUNK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class PublishedBlob:
    name: str
    path: str
    bytes: int


def sha256_hex(source_path: str) -> str:
    """SHA-256 hex digest of a file's bytes, read in chunks."""
    digest = hashlib.sha256()
    with open(source_path, "rb") as source_file:
        for chunk in iter(lambda: source_file.read(_HASH_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_addressed_name(source_path: str) -> str:
    """Content-hash filename ``<sha256><suffix>`` preserving the source extension."""
    return f"{sha256_hex(source_path)}{Path(source_path).suffix}"


def publish_blob(source_path: str, destination_directory: str) -> PublishedBlob:
    """Copy a blob into ``destination_directory`` under its content-addressed name."""
    destination_directory_path = Path(destination_directory)
    destination_directory_path.mkdir(parents=True, exist_ok=True)
    name = content_addressed_name(source_path)
    destination_path = destination_directory_path / name
    if not destination_path.exists():
        shutil.copyfile(source_path, destination_path)
    return PublishedBlob(name=name, path=str(destination_path), bytes=destination_path.stat().st_size)
