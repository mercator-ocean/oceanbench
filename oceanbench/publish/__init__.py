# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Publish stage: catalog and insights-manifest writers (contracts.md §4, §5, §8)."""

from oceanbench.publish.benchmark import (
    publish_benchmark_catalog,
    publish_challenger_insights,
    publish_challengers_registry,
)
from oceanbench.publish.catalog import CatalogEntry, build_catalog, write_catalog
from oceanbench.publish.column_store import ColumnStoreResult, build_column_store
from oceanbench.publish.content_address import PublishedBlob, content_addressed_name, publish_blob, sha256_hex
from oceanbench.publish.insights_manifest import InsightArtifact, write_insights_manifest
from oceanbench.publish.viewer_artifacts import (
    ViewerArtifactsResult,
    class4_bias_per_start_records,
    dataset_eddy_census,
    verify_matchup_parquet,
    write_eddy_census,
    write_matchup_parquet,
    write_viewer_artifacts,
)
from oceanbench.publish.s3 import (
    AwsCredentials,
    UploadPlanItem,
    UploadSummary,
    build_upload_plan,
    content_type_for_path,
    mint_sts_credentials,
    resolve_credentials,
    should_skip_upload,
    upload_tree,
)

__all__ = [
    "AwsCredentials",
    "CatalogEntry",
    "ColumnStoreResult",
    "InsightArtifact",
    "PublishedBlob",
    "UploadPlanItem",
    "UploadSummary",
    "ViewerArtifactsResult",
    "build_catalog",
    "build_column_store",
    "build_upload_plan",
    "class4_bias_per_start_records",
    "content_addressed_name",
    "content_type_for_path",
    "dataset_eddy_census",
    "mint_sts_credentials",
    "publish_benchmark_catalog",
    "publish_blob",
    "publish_challenger_insights",
    "publish_challengers_registry",
    "resolve_credentials",
    "sha256_hex",
    "should_skip_upload",
    "upload_tree",
    "verify_matchup_parquet",
    "write_catalog",
    "write_eddy_census",
    "write_insights_manifest",
    "write_matchup_parquet",
    "write_viewer_artifacts",
]
