# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Publish a local catalog tree to S3-compatible object storage (contracts.md §8).

The benchmark publish stage (``publish/benchmark.py``) writes a catalog tree to a
local ``output_root``. This module uploads that tree, preserving its layout, under
``s3://<bucket>/<prefix>/`` on an S3-compatible endpoint (EDITO MinIO by default).

Credentials resolve in this order:

1. Standard ``AWS_ACCESS_KEY_ID`` / ``AWS_SECRET_ACCESS_KEY`` (plus optional
   ``AWS_SESSION_TOKEN``) environment variables when they are set.
2. Otherwise, temporary STS credentials minted from an EDITO offline token
   (``EDITO_MINIO_OFFLINE_TOKEN``) via Keycloak + ``AssumeRoleWithWebIdentity``.

Secret values are never logged, printed or returned in any human-facing summary.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import os
import time
import xml.etree.ElementTree as ElementTree

import requests

EDITO_KEYCLOAK_TOKEN_URL = "https://auth.dive.edito.eu/auth/realms/datalab/protocol/openid-connect/token"
EDITO_KEYCLOAK_MINIO_CLIENT_ID = "onyxia-minio"
EDITO_MINIO_ENDPOINT = "https://minio.dive.edito.eu"
EDITO_OFFLINE_TOKEN_ENVIRONMENT_VARIABLE = "EDITO_MINIO_OFFLINE_TOKEN"

DEFAULT_MAX_WORKERS = 24
DEFAULT_ASSUME_ROLE_DURATION_SECONDS = 86400

_CONTENT_TYPE_BY_SUFFIX = {".json": "application/json"}
_DEFAULT_CONTENT_TYPE = "application/octet-stream"


def content_type_for_path(path: str | os.PathLike) -> str:
    """Return the Content-Type to store an object under, keyed on file extension.

    ``.json`` maps to ``application/json``; everything else — ``.parquet`` and the
    zarr pyramid chunk/metadata files that make up the bulk of the tree — maps to
    ``application/octet-stream``. Browser range GETs and zlib chunk decoding do not
    depend on the stored Content-Type, so a coarse extension map is sufficient.
    """
    return _CONTENT_TYPE_BY_SUFFIX.get(Path(path).suffix, _DEFAULT_CONTENT_TYPE)


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


def _parse_environment_file(env_file: str | os.PathLike) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in Path(env_file).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        values[key] = value
    return values


def _resolve_offline_token(environment: dict[str, str], env_file: str | os.PathLike | None) -> str:
    token = environment.get(EDITO_OFFLINE_TOKEN_ENVIRONMENT_VARIABLE)
    if not token and env_file is not None:
        token = _parse_environment_file(env_file).get(EDITO_OFFLINE_TOKEN_ENVIRONMENT_VARIABLE)
    if not token:
        raise RuntimeError(
            "No AWS_* credentials in the environment and no "
            f"{EDITO_OFFLINE_TOKEN_ENVIRONMENT_VARIABLE} available to mint STS credentials. "
            "Export AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or provide the offline token "
            "(source the EDITO env or pass --env-file)."
        )
    return token


def _keycloak_access_token(offline_token: str) -> str:
    response = requests.post(
        EDITO_KEYCLOAK_TOKEN_URL,
        data={
            "client_id": EDITO_KEYCLOAK_MINIO_CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": offline_token,
            "scope": "openid email profile",
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    response.raise_for_status()
    access_token = response.json().get("access_token")
    if not access_token:
        raise RuntimeError("Keycloak did not return an access_token for the offline token.")
    return access_token


def _local_tag(element: ElementTree.Element) -> str:
    return element.tag.rsplit("}", 1)[-1]


def _assume_role_with_web_identity(access_token: str, *, endpoint: str, duration_seconds: int) -> AwsCredentials:
    response = requests.post(
        endpoint,
        params={
            "Action": "AssumeRoleWithWebIdentity",
            "WebIdentityToken": access_token,
            "DurationSeconds": str(duration_seconds),
            "Version": "2011-06-15",
        },
        timeout=60,
    )
    response.raise_for_status()
    root = ElementTree.fromstring(response.text)
    fields = {
        _local_tag(element): element.text
        for element in root.iter()
        if _local_tag(element) in ("AccessKeyId", "SecretAccessKey", "SessionToken")
    }
    if "AccessKeyId" not in fields or "SecretAccessKey" not in fields:
        raise RuntimeError("AssumeRoleWithWebIdentity response did not contain credentials.")
    return AwsCredentials(
        access_key_id=fields["AccessKeyId"],
        secret_access_key=fields["SecretAccessKey"],
        session_token=fields.get("SessionToken"),
        source="edito-sts",
    )


def mint_sts_credentials(
    offline_token: str,
    *,
    endpoint: str = EDITO_MINIO_ENDPOINT,
    duration_seconds: int = DEFAULT_ASSUME_ROLE_DURATION_SECONDS,
) -> AwsCredentials:
    """Mint temporary S3 credentials from an EDITO offline token.

    Exchanges the offline (refresh) token for a Keycloak access token, then calls
    ``AssumeRoleWithWebIdentity`` on the MinIO endpoint. The returned credentials
    carry a ``SessionToken`` and expire after ``duration_seconds``. The offline
    token and minted secrets are never logged.
    """
    access_token = _keycloak_access_token(offline_token)
    return _assume_role_with_web_identity(access_token, endpoint=endpoint, duration_seconds=duration_seconds)


def resolve_credentials(
    *,
    endpoint: str = EDITO_MINIO_ENDPOINT,
    env_file: str | os.PathLike | None = None,
    environment: dict[str, str] | None = None,
) -> AwsCredentials:
    """Resolve S3 credentials following the documented order (AWS_* env, then STS).

    ``environment`` defaults to ``os.environ``. AWS_* variables are honoured only
    from the environment, never from ``env_file`` — the env file is consulted
    solely to locate the EDITO offline token (so a stale AWS_* pair left in a
    ``.env`` cannot shadow the STS flow that MinIO writes actually require).
    """
    resolved_environment = dict(os.environ if environment is None else environment)

    access_key_id = resolved_environment.get("AWS_ACCESS_KEY_ID")
    secret_access_key = resolved_environment.get("AWS_SECRET_ACCESS_KEY")
    if access_key_id and secret_access_key:
        return AwsCredentials(
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=resolved_environment.get("AWS_SESSION_TOKEN"),
            source="aws-env",
        )

    offline_token = _resolve_offline_token(resolved_environment, env_file)
    return mint_sts_credentials(offline_token, endpoint=endpoint)


def build_upload_plan(local_root: str | os.PathLike, prefix: str) -> list[UploadPlanItem]:
    """Walk ``local_root`` and map every file to ``<prefix>/<relative-posix-path>``."""
    root = Path(local_root)
    if not root.is_dir():
        raise NotADirectoryError(f"Local root is not a directory: {root}")
    normalized_prefix = prefix.strip("/")
    plan = []
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
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
    would not be detected — pass ``force`` to guarantee an overwrite of those.
    """
    if force:
        return False
    return _remote_size(s3_client, bucket, item.key) == item.size


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


def _upload_one(s3_client, bucket: str, item: UploadPlanItem, *, force: bool) -> tuple[UploadPlanItem, bool]:
    if should_skip_upload(s3_client, bucket, item, force=force):
        return item, False
    s3_client.upload_file(
        str(item.local_path),
        bucket,
        item.key,
        ExtraArgs={"ContentType": content_type_for_path(item.local_path)},
    )
    return item, True


def upload_tree(
    local_root: str | os.PathLike,
    *,
    bucket: str,
    prefix: str,
    endpoint: str = EDITO_MINIO_ENDPOINT,
    credentials: AwsCredentials | None = None,
    force: bool = False,
    max_workers: int = DEFAULT_MAX_WORKERS,
    env_file: str | os.PathLike | None = None,
) -> UploadSummary:
    """Upload the catalog tree at ``local_root`` to ``s3://<bucket>/<prefix>/``.

    Uploads run in parallel across ``max_workers`` threads. Objects whose remote
    size already matches the local size are skipped unless ``force`` is set (see
    ``should_skip_upload``). Resolves credentials via ``resolve_credentials`` when
    ``credentials`` is not supplied. Returns an ``UploadSummary`` (no secrets).
    """
    plan = build_upload_plan(local_root, prefix)
    total_bytes = sum(item.size for item in plan)
    resolved_credentials = credentials or resolve_credentials(endpoint=endpoint, env_file=env_file)
    s3_client = _build_s3_client(endpoint, resolved_credentials, max_workers)

    start = time.monotonic()
    uploaded_count = 0
    uploaded_bytes = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_upload_one, s3_client, bucket, item, force=force) for item in plan]
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
