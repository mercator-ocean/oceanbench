# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Per-challenger insights ``manifest.json`` writer (contracts.md §4).

Given a set of insight artifacts for one (challenger, year, region), this writer
content-addresses each blob into the insights directory and emits a
``manifest.json`` mapping the semantic key to ``{kind, schema_version, url, bytes}``.
The manifest is validated against ``insights-manifest.schema.json`` before it is
written; an invalid manifest is never emitted.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.publish.content_address import publish_blob

INSIGHTS_MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class InsightArtifact:
    """One insight blob to publish (contracts.md §4 kinds)."""

    semantic_key: str
    kind: str
    schema_version: str
    source_path: str


@dataclass(frozen=True)
class InsightsManifestResult:
    manifest: dict
    manifest_path: str


def _blob_url(blob_name: str, base_url: str | None) -> str:
    return blob_name if base_url is None else f"{base_url.rstrip('/')}/{blob_name}"


def write_insights_manifest(
    artifacts: list[InsightArtifact],
    output_directory: str,
    *,
    base_url: str | None = None,
) -> InsightsManifestResult:
    """Publish each artifact's blob content-addressed and write the validated manifest.

    ``base_url`` prefixes the blob names to form the manifest ``url`` values; when
    omitted the urls are the bare content-addressed blob names (relative references).
    """
    if not artifacts:
        raise ValueError("An insights manifest needs at least one artifact.")
    output_directory_path = Path(output_directory)
    manifest: dict[str, dict] = {}
    for artifact in artifacts:
        blob = publish_blob(artifact.source_path, str(output_directory_path))
        manifest[artifact.semantic_key] = {
            "kind": artifact.kind,
            "schema_version": artifact.schema_version,
            "url": _blob_url(blob.name, base_url),
            "bytes": blob.bytes,
        }
    validate_against_schema(manifest, "insights-manifest")
    manifest_path = output_directory_path / INSIGHTS_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return InsightsManifestResult(manifest=manifest, manifest_path=str(manifest_path))
