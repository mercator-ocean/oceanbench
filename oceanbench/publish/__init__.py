# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Publish stage: catalog and insights-manifest writers (contracts.md §4, §5, §8)."""

from oceanbench.publish.benchmark import publish_benchmark_catalog, publish_challenger_insights
from oceanbench.publish.catalog import CatalogEntry, build_catalog, write_catalog
from oceanbench.publish.content_address import PublishedBlob, content_addressed_name, publish_blob, sha256_hex
from oceanbench.publish.insights_manifest import InsightArtifact, write_insights_manifest

__all__ = [
    "CatalogEntry",
    "InsightArtifact",
    "PublishedBlob",
    "build_catalog",
    "content_addressed_name",
    "publish_benchmark_catalog",
    "publish_blob",
    "publish_challenger_insights",
    "sha256_hex",
    "write_catalog",
    "write_insights_manifest",
]
