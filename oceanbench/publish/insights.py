# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Insight-artifact writers for the realism battery (contracts.md §4).

Serializes the ``spectra`` and ``eddies`` payloads computed by
``oceanbench.runner.realism`` into schema-validated JSON blobs, content-addresses
them, and registers them in the per-(challenger, year, region) insights
``manifest.json`` through ``oceanbench.publish.insights_manifest``. Every payload
is validated against its schema (``schemas/spectra.schema.json`` /
``schemas/eddies.schema.json``) before it is written; an invalid payload is never
emitted.
"""

from dataclasses import dataclass
import json
from pathlib import Path

from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.publish.insights_manifest import InsightArtifact, InsightsManifestResult, write_insights_manifest

SPECTRA_KIND = "spectra"
EDDIES_KIND = "eddies"
SPECTRA_SCHEMA_VERSION = "1"
EDDIES_SCHEMA_VERSION = "1"

_SPECTRA_FILENAME = "spectra.json"
_EDDIES_FILENAME = "eddies.json"


@dataclass(frozen=True)
class RealismInsightsResult:
    spectra_payload: dict
    eddies_payload: dict
    manifest_result: InsightsManifestResult


def build_spectra_payload(spectra_entries: list[dict]) -> dict:
    """Assemble and validate a ``spectra`` payload from realism spectrum entries."""
    payload = {
        "kind": SPECTRA_KIND,
        "schema_version": SPECTRA_SCHEMA_VERSION,
        "entries": [_spectrum_entry_payload(entry) for entry in spectra_entries],
    }
    validate_against_schema(payload, SPECTRA_KIND)
    return payload


def build_eddies_payload(
    eddy_census: list[dict],
    *,
    variable: str | None = None,
    bounds: dict | None = None,
) -> dict:
    """Assemble and validate an ``eddies`` payload from a realism eddy census."""
    payload: dict = {
        "kind": EDDIES_KIND,
        "schema_version": EDDIES_SCHEMA_VERSION,
        "references": eddy_census,
    }
    if variable is not None:
        payload["variable"] = variable
    if bounds is not None:
        payload["bounds"] = bounds
    validate_against_schema(payload, EDDIES_KIND)
    return payload


def _spectrum_entry_payload(entry: dict) -> dict:
    payload = {
        "variable": entry["variable"],
        "region": entry["region"],
        "lead_day": entry["lead_day"],
        "wavelength": entry["wavelength"],
        "challenger_power": entry["challenger_power"],
        "reference_power": entry["reference_power"],
        "error_power": entry["error_power"],
    }
    if entry.get("reference") is not None:
        payload["reference"] = entry["reference"]
    if entry.get("unit") is not None:
        payload["unit"] = entry["unit"]
    return payload


def write_realism_insights(
    spectra_entries: list[dict],
    eddy_census: list[dict],
    output_directory: str,
    *,
    variable: str | None = None,
    bounds: dict | None = None,
    base_url: str | None = None,
) -> RealismInsightsResult:
    """Write the spectra and eddies insight blobs and register them in the insights manifest.

    Both payloads are validated against their schemas, written as content-addressed JSON
    blobs, and mapped from their semantic keys (``spectra``, ``eddies``) in ``manifest.json``.
    """
    spectra_payload = build_spectra_payload(spectra_entries)
    eddies_payload = build_eddies_payload(eddy_census, variable=variable, bounds=bounds)

    output_directory_path = Path(output_directory)
    output_directory_path.mkdir(parents=True, exist_ok=True)
    spectra_source_path = output_directory_path / _SPECTRA_FILENAME
    eddies_source_path = output_directory_path / _EDDIES_FILENAME
    spectra_source_path.write_text(json.dumps(spectra_payload, sort_keys=True, indent=2), encoding="utf-8")
    eddies_source_path.write_text(json.dumps(eddies_payload, sort_keys=True, indent=2), encoding="utf-8")

    manifest_result = write_insights_manifest(
        [
            InsightArtifact(
                semantic_key=SPECTRA_KIND,
                kind=SPECTRA_KIND,
                schema_version=SPECTRA_SCHEMA_VERSION,
                source_path=str(spectra_source_path),
            ),
            InsightArtifact(
                semantic_key=EDDIES_KIND,
                kind=EDDIES_KIND,
                schema_version=EDDIES_SCHEMA_VERSION,
                source_path=str(eddies_source_path),
            ),
        ],
        str(output_directory_path),
        base_url=base_url,
    )
    return RealismInsightsResult(
        spectra_payload=spectra_payload,
        eddies_payload=eddies_payload,
        manifest_result=manifest_result,
    )


def read_insight_payload(path: str) -> dict:
    """Read an insight JSON blob back into a dict (round-trip helper)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))
