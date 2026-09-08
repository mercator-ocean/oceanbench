# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""End-to-end publication: the publish writers emit a schema-valid catalog tree.

Builds a small artifact tree with the publish writers, then re-reads the catalog from
disk and validates it against its JSON Schema.
"""

import json
from pathlib import Path

import jsonschema

from oceanbench.core.schema_validation import load_schema
from oceanbench.publish.benchmark import publish_benchmark_catalog
from oceanbench.publish.catalog import CatalogEntry

BASE_URL = "https://example.org/benchmark-dev"


def _validate_tree(output_root: Path) -> dict:
    catalog = json.loads((output_root / "catalog.json").read_text())
    jsonschema.validate(catalog, load_schema("catalog"))
    return catalog


def test_publish_tree_is_schema_valid_end_to_end(tmp_path):
    output_root = tmp_path / "benchmark-dev"
    entries = [
        CatalogEntry(
            release="2.0.0",
            year="2024",
            region="global",
            challenger=challenger,
            viewer_zarr_url=f"{BASE_URL}/viewer/2024/{challenger}.zarr",
        )
        for challenger in ("glonet_1_degree", "climatology")
    ]

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
    assert challengers["glonet_1_degree"]["viewer_zarr_url"].endswith("glonet_1_degree.zarr")
