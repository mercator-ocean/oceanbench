# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""End-to-end publication: scores + matchups + pyramid manifest -> a valid catalog tree.

Builds a small artifact tree with the publish writers, then re-reads the whole tree
from disk and validates the catalog and every insights manifest against their JSON
Schemas — proving the publish stage emits a self-consistent, schema-valid tree.
"""

import json
from pathlib import Path

import jsonschema

from oceanbench.core.schema_validation import load_schema
from oceanbench.publish.benchmark import publish_benchmark_catalog, publish_challenger_insights
from oceanbench.publish.insights_manifest import InsightArtifact

BASE_URL = "https://example.org/benchmark-dev"


def _blob(path: Path, data: bytes) -> str:
    path.write_bytes(data)
    return str(path)


def _validate_tree(output_root: Path) -> dict:
    catalog = json.loads((output_root / "catalog.json").read_text())
    jsonschema.validate(catalog, load_schema("catalog"))
    for release in catalog["releases"].values():
        for year in release["years"].values():
            for region in year["regions"].values():
                for challenger in region.values():
                    manifest_relative = challenger["insights_manifest_url"].removeprefix(BASE_URL + "/")
                    manifest = json.loads((output_root / manifest_relative).read_text())
                    jsonschema.validate(manifest, load_schema("insights-manifest"))
    return catalog


def test_publish_tree_is_schema_valid_end_to_end(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    matchups_blob = _blob(source / "class4-matchups.parquet", b"a tiny stand-in for the matchups parquet")
    aggregate_blob = _blob(source / "aggregate-map.json", b'{"stand_in": true}')

    output_root = tmp_path / "benchmark-dev"
    entries = []
    for challenger in ("glonet_1_degree", "climatology"):
        entries.append(
            publish_challenger_insights(
                [
                    InsightArtifact("class4-matchups", "class4-matchups", "1.0", matchups_blob),
                    InsightArtifact("aggregate-map", "aggregate-map", "1.0", aggregate_blob),
                ],
                output_root=str(output_root),
                base_url=BASE_URL,
                release="2.0.0",
                year="2024",
                region="global",
                challenger=challenger,
                viewer_zarr_url=f"{BASE_URL}/viewer/2024/{challenger}.zarr",
            )
        )

    catalog, catalog_path = publish_benchmark_catalog(
        entries,
        output_root=str(output_root),
        scores_url=f"{BASE_URL}/scores.parquet",
        challengers_url=f"{BASE_URL}/challengers.json",
        generated_at="2026-07-03T00:00:00+00:00",
    )

    validated = _validate_tree(output_root)
    assert validated == catalog
    challengers = validated["releases"]["2.0.0"]["years"]["2024"]["regions"]["global"]
    assert set(challengers) == {"glonet_1_degree", "climatology"}
    # Both challengers share the identical matchups blob -> one content-addressed copy each.
    glonet_manifest = json.loads((output_root / "2024/global/glonet_1_degree/insights/manifest.json").read_text())
    assert glonet_manifest["class4-matchups"]["kind"] == "class4-matchups"
    assert glonet_manifest["class4-matchups"]["url"].endswith(".parquet")
