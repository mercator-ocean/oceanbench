# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Publish a local catalog tree to S3-compatible object storage (contracts.md §8).

The benchmark publish stage (``publish/benchmark.py``) writes a catalog tree to a
local ``output_root``. This module uploads that tree, preserving its layout, under
``s3://<bucket>/<prefix>/`` on an S3-compatible endpoint (CloudFerro by default).

Credentials come from the standard ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY``
environment variables (plus optional ``AWS_SESSION_TOKEN``). ``AWS_S3_ENDPOINT``
overrides the default endpoint.

Secret values are never logged, printed or returned in any human-facing summary.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import os
import re
import time

CLOUDFERRO_ENDPOINT = "https://s3.waw3-1.cloudferro.com"
ENDPOINT_ENVIRONMENT_VARIABLE = "AWS_S3_ENDPOINT"

DEFAULT_MAX_WORKERS = 24

_CONTENT_TYPE_BY_SUFFIX = {
    ".css": "text/css",
    ".html": "text/html",
    ".ico": "image/x-icon",
    ".js": "text/javascript",
    ".json": "application/json",
    ".md": "text/markdown",
    ".mjs": "text/javascript",
    ".parquet": "application/vnd.apache.parquet",
    ".png": "image/png",
    ".svg": "image/svg+xml",
}
_DEFAULT_CONTENT_TYPE = "application/octet-stream"

IMMUTABLE_CACHE_CONTROL = "public, max-age=31536000, immutable"
MUTABLE_INDEX_CACHE_CONTROL = "public, max-age=60"
_MUTABLE_INDEX_NAMES = frozenset({"datasets.json", "scores-summary.json"})
_MUTABLE_INDEX_SUFFIX = ".viewer-manifest.json"
_ZARR_CHUNK_NAME = re.compile(r"^\d+(\.\d+)*$")


def content_type_for_path(path: str | os.PathLike) -> str:
    """Return the Content-Type to store an object under, keyed on file extension.

    Browser-facing site assets and known data/document extensions get explicit
    MIME types. Extensionless zarr pyramid chunks stay ``application/octet-stream``.
    """
    return _CONTENT_TYPE_BY_SUFFIX.get(Path(path).suffix, _DEFAULT_CONTENT_TYPE)


def cache_control_for_path(path: str | os.PathLike) -> str | None:
    """Return the Cache-Control to store an object under, or ``None`` to omit the header.

    Content-addressed artifacts the viewer never re-reads after a republish (zarr
    pyramid chunks, parquet match-ups) are cached for a year and marked immutable.
    The few fixed-name indexes the viewer polls to discover everything else get a
    short max-age instead. Anything not in either class keeps the storage default.
    """
    name = Path(path).name
    if name in _MUTABLE_INDEX_NAMES or name.endswith(_MUTABLE_INDEX_SUFFIX):
        return MUTABLE_INDEX_CACHE_CONTROL
    if Path(path).suffix == ".parquet" or _ZARR_CHUNK_NAME.match(name):
        return IMMUTABLE_CACHE_CONTROL
    return None


@dataclass(frozen=True)
class AwsCredentials:
    """Resolved S3 credentials plus a non-secret ``source`` label for reporting."""

    access_key_id: str
    secret_access_key: str
    session_token: str | None
    source: str

    def boto3_client_keyword_arguments(self) -> dict:
        keyword_arguments = {
            "aws_access_key_id": self.access_key_id,
            "aws_secret_access_key": self.secret_access_key,
        }
        if self.session_token is not None:
            keyword_arguments["aws_session_token"] = self.session_token
        return keyword_arguments


@dataclass(frozen=True)
class UploadPlanItem:
    """One local file mapped to its remote key, with the local file size."""

    local_path: Path
    key: str
    size: int


@dataclass(frozen=True)
class UploadSummary:
    """Human-facing result of an upload run. Carries no secret values."""

    planned_count: int
    uploaded_count: int
    skipped_count: int
    uploaded_bytes: int
    total_bytes: int
    elapsed_seconds: float


def default_endpoint(environment: dict[str, str] | None = None) -> str:
    """Return the S3 endpoint to publish to: ``AWS_S3_ENDPOINT`` when set, else CloudFerro."""
    resolved_environment = os.environ if environment is None else environment
    return resolved_environment.get(ENDPOINT_ENVIRONMENT_VARIABLE) or CLOUDFERRO_ENDPOINT


def resolve_credentials(environment: dict[str, str] | None = None) -> AwsCredentials:
    """Resolve S3 credentials from the AWS environment variables.

    ``environment`` defaults to ``os.environ``. ``AWS_SESSION_TOKEN`` is carried
    through when present so temporary credentials also work.
    """
    resolved_environment = dict(os.environ if environment is None else environment)

    access_key_id = resolved_environment.get("AWS_ACCESS_KEY_ID")
    secret_access_key = resolved_environment.get("AWS_SECRET_ACCESS_KEY")
    if not access_key_id or not secret_access_key:
        raise RuntimeError("No S3 credentials: export AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")
    return AwsCredentials(
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        session_token=resolved_environment.get("AWS_SESSION_TOKEN"),
        source="aws-env",
    )


def _walk_files_following_symlinks(root: Path) -> list[Path]:
    """Yield every file under ``root``, descending into symlinked directories.

    ``Path.rglob`` does not follow directory symlinks on Python 3.12, so the viewer
    pyramid directories (which are symlinks) would be silently skipped. ``os.walk``
    with ``followlinks=True`` descends into them; a set of visited real directory
    paths guards against symlink cycles walking forever.
    """
    files: list[Path] = []
    visited_directories: set[str] = set()
    for directory, subdirectories, filenames in os.walk(root, followlinks=True):
        real_directory = os.path.realpath(directory)
        if real_directory in visited_directories:
            subdirectories[:] = []
            continue
        visited_directories.add(real_directory)
        subdirectories[:] = [
            name
            for name in subdirectories
            if os.path.realpath(os.path.join(directory, name)) not in visited_directories
        ]
        for filename in filenames:
            candidate = Path(directory) / filename
            if candidate.is_file():
                files.append(candidate)
    return files


def build_upload_plan(local_root: str | os.PathLike, prefix: str) -> list[UploadPlanItem]:
    """Walk ``local_root`` and map every file to ``<prefix>/<relative-posix-path>``.

    Directory symlinks are followed (with a cycle guard) so symlinked subtrees such
    as the viewer zarr pyramid directories are published, not silently skipped.
    """
    root = Path(local_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Local root is not a directory: {root}")
    normalized_prefix = prefix.strip("/")
    plan = []
    for path in sorted(_walk_files_following_symlinks(root)):
        relative = path.relative_to(root).as_posix()
        key = f"{normalized_prefix}/{relative}" if normalized_prefix else relative
        plan.append(UploadPlanItem(local_path=path, key=key, size=path.stat().st_size))
    return plan


def _remote_size(s3_client, bucket: str, key: str) -> int | None:
    from botocore.exceptions import ClientError

    try:
        response = s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    return response["ContentLength"]


def should_skip_upload(s3_client, bucket: str, item: UploadPlanItem, *, force: bool) -> bool:
    """Skip when a remote object of the same size already exists (unless ``force``).

    Content-addressed blob keys make size+key a sufficient cheap idempotency check:
    a differing blob yields a different key. For the few fixed-name files
    (``catalog.json``, ``scores.parquet``, ``challengers.json``) a same-size edit
    would not be detected, pass ``force`` to guarantee an overwrite of those.
    """
    if force:
        return False
    return _remote_size(s3_client, bucket, item.key) == item.size


def _remote_manifest_fingerprint(s3_client, bucket: str, key: str) -> str | None:
    """Fingerprint stamped in the pyramid manifest already published at ``key``, if any."""
    import gzip
    import json

    from botocore.exceptions import ClientError

    from oceanbench.publish import viewer_artifacts

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise
    body = response["Body"].read()
    if body[:2] == b"\x1f\x8b":
        body = gzip.decompress(body)
    try:
        manifest = json.loads(body)
    except ValueError:
        return None
    provenance = manifest.get(viewer_artifacts.PROVENANCE_KEY)
    if not isinstance(provenance, dict):
        return None
    fingerprint = provenance.get(viewer_artifacts.FINGERPRINT_KEY)
    return fingerprint if isinstance(fingerprint, str) else None


def unchanged_dataset_slugs(s3_client, bucket: str, local_root: str | os.PathLike, prefix: str) -> dict[str, str]:
    """Datasets whose published fingerprint already matches the local one, as ``{slug: fingerprint}``.

    A dataset is identified by its ``<slug>.viewer-manifest.json`` in ``local_root``; the comparison
    reads the fingerprint stamped in the local manifest and in the manifest already published under
    ``prefix``. Datasets with no local fingerprint (older artifacts) never match.
    """
    from oceanbench.publish import viewer_artifacts

    root = Path(local_root)
    normalized_prefix = prefix.strip("/")
    unchanged = {}
    for manifest_path in sorted(root.rglob(f"*{viewer_artifacts.PYRAMID_MANIFEST_SUFFIX}")):
        slug = manifest_path.name[: -len(viewer_artifacts.PYRAMID_MANIFEST_SUFFIX)]
        local_fingerprint = viewer_artifacts.read_manifest_fingerprint(manifest_path)
        if local_fingerprint is None:
            continue
        relative = manifest_path.relative_to(root).as_posix()
        key = f"{normalized_prefix}/{relative}" if normalized_prefix else relative
        if _remote_manifest_fingerprint(s3_client, bucket, key) == local_fingerprint:
            unchanged[slug] = local_fingerprint
    return unchanged


def _plan_without_datasets(plan: list[UploadPlanItem], local_root: str | os.PathLike, slugs) -> list[UploadPlanItem]:
    from oceanbench.publish import viewer_artifacts

    root = Path(local_root)
    return [
        item
        for item in plan
        if not any(
            viewer_artifacts.relative_path_belongs_to_dataset(item.local_path.relative_to(root).as_posix(), slug)
            for slug in slugs
        )
    ]


def _build_s3_client(endpoint: str, credentials: AwsCredentials, max_workers: int):
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        config=Config(
            signature_version="s3v4",
            max_pool_connections=max(max_workers, DEFAULT_MAX_WORKERS),
            retries={"max_attempts": 5, "mode": "standard"},
        ),
        **credentials.boto3_client_keyword_arguments(),
    )


def _gzip_bytes(raw: bytes) -> bytes:
    import gzip

    # mtime=0 keeps the compressed bytes deterministic for a given input.
    return gzip.compress(raw, mtime=0)


def _upload_one(
    s3_client, bucket: str, item: UploadPlanItem, *, force: bool, compress_json: bool
) -> tuple[UploadPlanItem, bool]:
    is_json = str(item.local_path).endswith(".json")
    cache_control = cache_control_for_path(item.local_path)
    if compress_json and is_json:
        body = _gzip_bytes(Path(item.local_path).read_bytes())
        stored_item = UploadPlanItem(local_path=item.local_path, key=item.key, size=len(body))
        if should_skip_upload(s3_client, bucket, stored_item, force=force):
            return stored_item, False
        s3_client.put_object(
            Bucket=bucket,
            Key=item.key,
            Body=body,
            ContentType="application/json",
            ContentEncoding="gzip",
            **({"CacheControl": cache_control} if cache_control else {}),
        )
        return stored_item, True
    if should_skip_upload(s3_client, bucket, item, force=force):
        return item, False
    extra_arguments = {"ContentType": content_type_for_path(item.local_path)}
    if cache_control:
        extra_arguments["CacheControl"] = cache_control
    s3_client.upload_file(str(item.local_path), bucket, item.key, ExtraArgs=extra_arguments)
    return item, True


def upload_tree(
    local_root: str | os.PathLike,
    *,
    bucket: str,
    prefix: str,
    endpoint: str | None = None,
    credentials: AwsCredentials | None = None,
    force: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    compress_json: bool = False,
) -> UploadSummary:
    """Upload the catalog tree at ``local_root`` to ``s3://<bucket>/<prefix>/``.

    Uploads run in parallel across ``max_workers`` threads. Objects whose remote
    size already matches the local size are skipped unless ``force`` is set (see
    ``should_skip_upload``). Resolves credentials via ``resolve_credentials`` when
    ``credentials`` is not supplied. Returns an ``UploadSummary`` (no secrets).

    A whole dataset is skipped before any of its objects is considered when the fingerprint in its
    local ``<slug>.viewer-manifest.json`` matches the one already published (see
    ``unchanged_dataset_slugs``); ``force`` bypasses that too.

    When ``compress_json`` is set, every ``.json`` object is stored gzip-compressed with
    ``Content-Encoding: gzip`` and ``Content-Type: application/json`` so the browser decompresses
    it transparently (large viewer JSON compresses roughly 7-15x); other objects are unchanged.
    """
    plan = build_upload_plan(local_root, prefix)
    resolved_endpoint = endpoint or default_endpoint()
    resolved_credentials = credentials or resolve_credentials()
    s3_client = _build_s3_client(resolved_endpoint, resolved_credentials, max_workers)

    if not force:
        unchanged = unchanged_dataset_slugs(s3_client, bucket, local_root, prefix)
        for slug, fingerprint in unchanged.items():
            print(f"skip {slug}: unchanged ({fingerprint[:8]})")
        if unchanged:
            plan = _plan_without_datasets(plan, local_root, unchanged)
    total_bytes = sum(item.size for item in plan)

    start = time.monotonic()
    uploaded_count = 0
    uploaded_bytes = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_upload_one, s3_client, bucket, item, force=force, compress_json=compress_json)
            for item in plan
        ]
        for future in as_completed(futures):
            item, was_uploaded = future.result()
            if was_uploaded:
                uploaded_count += 1
                uploaded_bytes += item.size
    elapsed_seconds = time.monotonic() - start

    return UploadSummary(
        planned_count=len(plan),
        uploaded_count=uploaded_count,
        skipped_count=len(plan) - uploaded_count,
        uploaded_bytes=uploaded_bytes,
        total_bytes=total_bytes,
        elapsed_seconds=elapsed_seconds,
    )
