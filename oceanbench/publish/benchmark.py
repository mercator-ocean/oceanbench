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

from pathlib import Path

from oceanbench.publish.catalog import CatalogEntry, write_catalog
from oceanbench.publish.insights_manifest import (
    INSIGHTS_MANIFEST_FILENAME,
    InsightArtifact,
    write_insights_manifest,
)


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
