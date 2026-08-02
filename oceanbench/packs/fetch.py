# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Download a published evaluation pack from an anonymous HTTPS prefix.

Packs are published, not built, by users: the maintainers build an official pack per region
and year with ``oceanbench build-pack`` and upload the resulting directory tree. A pack is a
tree of zarr stores, so it has no single archive to download; ``pack-files.json`` (written by
:func:`oceanbench.packs.fetch.write_pack_file_index` at the end of the build) is the flat index
of every object in the tree with its byte size, and is what ``fetch_pack`` enumerates.

Downloads are plain anonymous HTTPS GETs. A file already present at the expected size is
skipped, which makes a re-run of an interrupted fetch cheap without any resume protocol.
"""

from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
import shutil
from urllib.parse import quote, urljoin, urlparse
from urllib.request import urlopen

PACK_FILE_INDEX_FILENAME = "pack-files.json"
PACK_FILE_INDEX_SCHEMA_VERSION = "1.0"
DEFAULT_PACK_CACHE_ENVIRONMENT_VARIABLE = "OCEANBENCH_PACK_CACHE"
_DOWNLOAD_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class PackFetchSummary:
    pack_name: str
    destination: str
    downloaded_count: int
    downloaded_bytes: int
    skipped_count: int
    total_count: int
    total_bytes: int


def default_pack_cache_root() -> Path:
    """Root directory the fetched packs are cached under (``~/.cache/oceanbench/packs``)."""
    configured = os.environ.get(DEFAULT_PACK_CACHE_ENVIRONMENT_VARIABLE)
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".cache" / "oceanbench" / "packs"


def default_pack_cache_directory(pack_name: str) -> Path:
    return default_pack_cache_root() / pack_name


def pack_name_from_source(source: str) -> str:
    """The pack directory name a source URL or bare name denotes."""
    if "://" not in source:
        return source.strip("/")
    name = Path(urlparse(source).path.rstrip("/")).name
    if not name:
        raise ValueError(f"cannot derive a pack name from {source!r}; the URL has no final path segment")
    return name


def _pack_relative_files(pack_directory: Path) -> list[dict]:
    entries = [
        {"path": path.relative_to(pack_directory).as_posix(), "size": path.stat().st_size}
        for path in pack_directory.rglob("*")
        if path.is_file() and path.name != PACK_FILE_INDEX_FILENAME
    ]
    return sorted(entries, key=lambda entry: entry["path"])


def write_pack_file_index(pack_directory: str) -> str:
    """Write ``pack-files.json``, the flat download index of every file in a built pack.

    Call this last in a pack build: the index lists every other file in the tree, so anything
    written afterwards would be invisible to ``fetch_pack``.
    """
    directory = Path(pack_directory)
    files = _pack_relative_files(directory)
    payload = {
        "schema_version": PACK_FILE_INDEX_SCHEMA_VERSION,
        "pack_name": directory.name,
        "file_count": len(files),
        "total_bytes": sum(entry["size"] for entry in files),
        "files": files,
    }
    index_path = directory / PACK_FILE_INDEX_FILENAME
    index_path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return str(index_path)


def read_json_url(url: str, *, timeout: int = 30) -> dict:
    """Read a JSON document over HTTPS, inflating a gzip-encoded body.

    Published JSON objects are stored gzip-compressed and served with
    ``Content-Encoding: gzip``. urllib does not act on that header, so the body
    arrives as gzip bytes and has to be inflated before it can be parsed.
    """
    with urlopen(url, timeout=timeout) as response:  # noqa: S310
        body = response.read()
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    return json.loads(body)


def _read_pack_file_index(base_url: str) -> dict:
    index_url = urljoin(base_url, PACK_FILE_INDEX_FILENAME)
    try:
        return read_json_url(index_url, timeout=_DOWNLOAD_TIMEOUT_SECONDS)
    except OSError as error:
        raise ValueError(
            f"no downloadable pack at {base_url}: unable to read {PACK_FILE_INDEX_FILENAME} ({error}). "
            "A published pack carries that index at its root; packs built before it existed must be rebuilt."
        ) from error


def _resolved_target(destination: Path, relative_path: str) -> Path:
    target = (destination / relative_path).resolve()
    if not target.is_relative_to(destination.resolve()):
        raise ValueError(f"the pack index entry {relative_path!r} escapes the destination directory")
    return target


def _download_file(file_url: str, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(file_url, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as response:  # noqa: S310
        with target.open("wb") as output:
            shutil.copyfileobj(response, output)
    return target.stat().st_size


def fetch_pack(source: str, destination: str | None = None) -> PackFetchSummary:
    """Download the pack published at ``source`` into ``destination`` (default: the local cache).

    ``source`` is the https:// prefix of a published pack directory. Files already present at
    the size the index states are skipped, so re-running a partial fetch only pulls what is
    missing. No retries and no byte-range resume: a truncated file has the wrong size and is
    simply downloaded again.
    """
    if "://" not in source:
        raise ValueError(f"{source!r} is not a pack URL; pass the https:// prefix of a published pack")
    base_url = source.rstrip("/") + "/"
    pack_name = pack_name_from_source(base_url)
    destination_path = Path(destination).expanduser() if destination else default_pack_cache_directory(pack_name)
    destination_path.mkdir(parents=True, exist_ok=True)

    index = _read_pack_file_index(base_url)
    entries = index["files"]

    downloaded_count = 0
    downloaded_bytes = 0
    skipped_count = 0
    for entry in entries:
        target = _resolved_target(destination_path, entry["path"])
        if target.is_file() and target.stat().st_size == entry["size"]:
            skipped_count += 1
            continue
        downloaded_bytes += _download_file(urljoin(base_url, quote(entry["path"])), target)
        downloaded_count += 1

    write_pack_file_index(str(destination_path))
    return PackFetchSummary(
        pack_name=index.get("pack_name", pack_name),
        destination=str(destination_path),
        downloaded_count=downloaded_count,
        downloaded_bytes=downloaded_bytes,
        skipped_count=skipped_count,
        total_count=len(entries),
        total_bytes=sum(entry["size"] for entry in entries),
    )


def resolve_offline_references(offline_references: str | None) -> str | None:
    """Turn an ``--offline-references`` value into a local pack directory.

    A local path is returned untouched. An https:// prefix is fetched into the local pack cache
    first and the cached directory is returned, so the same flag serves a downloaded pack and a
    published one.
    """
    if offline_references is None or not offline_references.startswith("https://"):
        return offline_references
    summary = fetch_pack(offline_references)
    print(
        f"pack {summary.pack_name}: {summary.total_count} files, {summary.total_bytes:,} bytes "
        f"({summary.downloaded_count} downloaded, {summary.skipped_count} already present) -> {summary.destination}"
    )
    return summary.destination
