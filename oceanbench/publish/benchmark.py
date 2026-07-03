# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Minimal end-to-end publication of the benchmark artifact tree (contracts.md §5, §8).

Lays out, under an output root, the S3-style tree: per (release, year, region,
challenger) an ``insights/manifest.json`` (with content-addressed blobs) and a
catalog entry pointing at that manifest and the challenger's viewer zarr; finally a
root ``catalog.json`` indexing everything alongside the single ``scores.parquet``.
Every manifest and the catalog are schema-validated by their writers.
"""

import json
from pathlib import Path

import pandas

from oceanbench.core.schema_validation import validate_against_schema
from oceanbench.publish.aggregate import aggregate_scores, summary_to_json_records
from oceanbench.publish.catalog import CatalogEntry, write_catalog
from oceanbench.publish.compact import SCORES_FILENAME, compact_runs_directory
from oceanbench.publish.insights_manifest import (
    INSIGHTS_MANIFEST_FILENAME,
    InsightArtifact,
    write_insights_manifest,
)

SCORES_SUMMARY_FILENAME = "scores-summary.json"
CHALLENGERS_REGISTRY_FILENAME = "challengers.json"


def _repository_challengers_registry_path() -> Path:
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / CHALLENGERS_REGISTRY_FILENAME
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate the repository {CHALLENGERS_REGISTRY_FILENAME}.")


def publish_challengers_registry(output_root: str, *, registry_path: str | None = None) -> str:
    """Emit the challenger registry (``challengers.json``) into the catalog root.

    Reads the in-repo, versioned registry mapping each canonical challenger slug to its
    metadata (display name, ``is_baseline`` flag used to pin baselines, resolution ...),
    validates it against ``challengers.schema.json`` and copies it next to
    ``scores.parquet`` so the static score page can read display names and pin baselines.
    ``registry_path`` overrides the repository-root registry. Returns the written path.
    """
    source_path = Path(registry_path) if registry_path is not None else _repository_challengers_registry_path()
    registry = json.loads(source_path.read_text(encoding="utf-8"))
    validate_against_schema(registry, "challengers")
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    destination_path = output_root_path / CHALLENGERS_REGISTRY_FILENAME
    destination_path.write_text(
        json.dumps(registry, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return str(destination_path)


def publish_challenger_insights(
    artifacts: list[InsightArtifact],
    *,
    output_root: str,
    base_url: str,
    release: str,
    year: str,
    region: str,
    challenger: str,
    viewer_zarr_url: str,
) -> CatalogEntry:
    """Publish one challenger's insights tree and return its catalog entry."""
    insights_directory = Path(output_root) / year / region / challenger / "insights"
    insights_base_url = f"{base_url.rstrip('/')}/{year}/{region}/{challenger}/insights"
    write_insights_manifest(artifacts, str(insights_directory), base_url=insights_base_url)
    return CatalogEntry(
        release=release,
        year=year,
        region=region,
        challenger=challenger,
        insights_manifest_url=f"{insights_base_url}/{INSIGHTS_MANIFEST_FILENAME}",
        viewer_zarr_url=viewer_zarr_url,
    )


def publish_scores(
    runs_root: str,
    *,
    output_root: str,
    baseline_challenger: str | None = None,
) -> tuple[str, str]:
    """Compact every run parquet under ``runs_root`` into the catalog-root ``scores.parquet``.

    Also emits the precomputed aggregated ``scores-summary.json`` next to it (means, bootstrap
    CIs and optional skill vs ``baseline_challenger``) so the static score page can render the
    scorecard without recomputing the bootstrap in the browser. The parquet stays the canonical
    artifact; the summary is a derived convenience. Returns ``(scores_path, summary_path)``.
    """
    output_root_path = Path(output_root)
    output_root_path.mkdir(parents=True, exist_ok=True)
    scores_path = compact_runs_directory(runs_root, str(output_root_path / SCORES_FILENAME))

    summary = aggregate_scores(pandas.read_parquet(scores_path), baseline_challenger=baseline_challenger)
    summary_path = output_root_path / SCORES_SUMMARY_FILENAME
    summary_path.write_text(
        json.dumps(summary_to_json_records(summary), sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return scores_path, str(summary_path)


def publish_benchmark_catalog(
    entries: list[CatalogEntry],
    *,
    output_root: str,
    scores_url: str,
    challengers_url: str | None = None,
    generated_at: str | None = None,
) -> tuple[dict, str]:
    """Write the root ``catalog.json`` indexing the published challenger entries."""
    return write_catalog(
        entries,
        output_root,
        scores_url=scores_url,
        challengers_url=challengers_url,
        generated_at=generated_at,
    )
