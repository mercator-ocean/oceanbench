# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

import jsonschema
import pytest

from oceanbench.core.schema_validation import load_schema
from oceanbench.publish.catalog import CatalogEntry, build_catalog, write_catalog


def _entry(challenger: str = "glonet_1_degree", region: str = "global") -> CatalogEntry:
    return CatalogEntry(
        release="2.0.0",
        year="2024",
        region=region,
        challenger=challenger,
        insights_manifest_url=f"https://example.org/2024/{region}/{challenger}/insights/manifest.json",
        viewer_zarr_url=f"https://example.org/viewer/2024/{challenger}.zarr",
    )


def test_build_catalog_nests_and_validates():
    catalog = build_catalog(
        [_entry("glonet_1_degree", "global"), _entry("climatology", "global"), _entry("glonet_1_degree", "ibi")],
        scores_url="https://example.org/scores.parquet",
        generated_at="2026-07-03T00:00:00+00:00",
    )
    jsonschema.validate(catalog, load_schema("catalog"))
    regions = catalog["releases"]["2.0.0"]["years"]["2024"]["regions"]
    assert set(regions["global"]) == {"glonet_1_degree", "climatology"}
    assert set(regions) == {"global", "ibi"}


def test_write_catalog_emits_valid_file(tmp_path):
    catalog, path = write_catalog(
        [_entry()],
        str(tmp_path),
        scores_url="https://example.org/scores.parquet",
        challengers_url="https://example.org/challengers.json",
    )
    written = json.loads((tmp_path / "catalog.json").read_text())
    jsonschema.validate(written, load_schema("catalog"))
    assert written["challengers_url"] == "https://example.org/challengers.json"
    assert written["schema_version"] == "2.0"


def test_duplicate_entry_rejected():
    with pytest.raises(ValueError):
        build_catalog([_entry(), _entry()], scores_url="https://example.org/scores.parquet")


def test_empty_entries_rejected():
    with pytest.raises(ValueError):
        build_catalog([], scores_url="https://example.org/scores.parquet")


def test_bad_release_version_is_rejected_by_schema():
    entry = CatalogEntry(
        release="not-a-semver",
        year="2024",
        region="global",
        challenger="glonet_1_degree",
        insights_manifest_url="https://example.org/m.json",
        viewer_zarr_url="https://example.org/v.zarr",
    )
    with pytest.raises(jsonschema.ValidationError):
        build_catalog([entry], scores_url="https://example.org/scores.parquet")
