# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Publish stage: catalog writer and upload (contracts.md §5, §8)."""

from oceanbench.publish.benchmark import (
    publish_benchmark_catalog,
    publish_challengers_registry,
)
from oceanbench.publish.catalog import CatalogEntry, build_catalog, write_catalog
from oceanbench.publish.column_store import ColumnStoreResult, build_column_store
from oceanbench.publish.viewer_artifacts import (
    ViewerArtifactsResult,
    class4_bias_per_start_records,
    dataset_eddy_census,
    dataset_fingerprint,
    stamp_dataset_fingerprint,
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
    resolve_credentials,
    should_skip_upload,
    unchanged_dataset_slugs,
    upload_tree,
)

__all__ = [
    "AwsCredentials",
    "CatalogEntry",
    "ColumnStoreResult",
    "UploadPlanItem",
    "UploadSummary",
    "ViewerArtifactsResult",
    "build_catalog",
    "build_column_store",
    "build_upload_plan",
    "class4_bias_per_start_records",
    "content_type_for_path",
    "dataset_eddy_census",
    "dataset_fingerprint",
    "publish_benchmark_catalog",
    "publish_challengers_registry",
    "resolve_credentials",
    "should_skip_upload",
    "stamp_dataset_fingerprint",
    "unchanged_dataset_slugs",
    "upload_tree",
    "verify_matchup_parquet",
    "write_catalog",
    "write_eddy_census",
    "write_matchup_parquet",
    "write_viewer_artifacts",
]
