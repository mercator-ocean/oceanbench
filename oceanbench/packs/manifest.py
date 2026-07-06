# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Evaluation-pack ``pack-manifest.json`` writer (contracts.md §7).

The manifest is the pack's self-describing index: ``oceanbench evaluate`` reads it
alone to locate every bundled reference, the observation store and the mean-dynamic-topography.
It stamps (contracts.md §1) the upstream product identifiers and retrieval dates the pack
derives from and carries the Copernicus Marine attribution/disclaimer (contracts.md §11). The
manifest is validated against ``pack-manifest.schema.json`` before it is written; an invalid
manifest is never emitted.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from oceanbench.core.schema_validation import validate_against_schema

PACK_MANIFEST_FILENAME = "pack-manifest.json"
PACK_MANIFEST_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class PackManifestResult:
    manifest: dict
    manifest_path: str


def write_pack_manifest(manifest: dict, output_directory: str) -> PackManifestResult:
    """Validate and write ``pack-manifest.json`` into ``output_directory``."""
    validate_against_schema(manifest, "pack-manifest")
    manifest_path = Path(output_directory) / PACK_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return PackManifestResult(manifest=manifest, manifest_path=str(manifest_path))
