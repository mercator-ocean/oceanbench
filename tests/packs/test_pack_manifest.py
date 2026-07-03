# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Pack manifest: schema shape, validation-in-writer, and attribution (contracts.md §7, §11)."""

import json

import jsonschema
import pytest

from oceanbench.core.attribution import COPERNICUS_MARINE_CREDIT, COPERNICUS_MARINE_DISCLAIMER
from oceanbench.core.schema_validation import load_schema
from oceanbench.packs.builder import PackBuildResult, _pack_readme
from oceanbench.packs.manifest import PACK_MANIFEST_FILENAME, write_pack_manifest


def _valid_manifest() -> dict:
    return {
        "schema_version": "1.0",
        "kind": "quick",
        "year": 2024,
        "region": "global",
        "resolution": "one_degree",
        "oceanbench_version": "0.2.1",
        "generated_at": "2026-07-04T00:00:00+00:00",
        "start_dates": ["2024-01-03", "2024-01-10"],
        "attribution": COPERNICUS_MARINE_CREDIT,
        "disclaimer": COPERNICUS_MARINE_DISCLAIMER,
        "source_products": {
            "GLORYS12": "GLOBAL_MULTIYEAR_PHY_001_030",
            "GLO12": "GLOBAL_ANALYSISFORECAST_PHY_001_024",
        },
        "upstream": [
            {"name": "glorys", "product_id": "GLOBAL_MULTIYEAR_PHY_001_030", "retrieved": "2026-07-04"},
            {"name": "glo12", "product_id": "GLOBAL_ANALYSISFORECAST_PHY_001_024", "retrieved": "2026-07-04"},
            {"name": "observations", "product_id": "OceanBench-observations-2024-v3", "retrieved": "2026-07-04"},
        ],
        "contents": {
            "references": {
                "glorys": {
                    "path": "references/glorys.zarr",
                    "variables": ["sea_surface_height_above_geoid"],
                    "depths": ["surface"],
                },
                "glo12": {
                    "path": "references/glo12.zarr",
                    "variables": ["sea_surface_height_above_geoid"],
                    "depths": ["surface"],
                },
            },
            "observations": {"path": "observations/observations.zarr"},
            "mean_dynamic_topography": {
                "path": "class4-mean-dynamic-topography-2024-glo12-one_degree.zarr",
                "resolution": "one_degree",
            },
            "baselines": {},
        },
        "baselines_available": False,
        "notes": ["Baselines (climatology / persistence) are not available yet."],
    }


def test_valid_manifest_writes_and_round_trips(tmp_path):
    result = write_pack_manifest(_valid_manifest(), str(tmp_path))
    written = json.loads((tmp_path / PACK_MANIFEST_FILENAME).read_text())
    jsonschema.validate(written, load_schema("pack-manifest"))
    assert written == result.manifest
    # The manifest locates every reference from the manifest alone (self-describing).
    assert set(written["contents"]["references"]) == {"glorys", "glo12"}
    assert written["contents"]["observations"]["path"].endswith(".zarr")


def test_writer_rejects_a_manifest_missing_a_required_field(tmp_path):
    invalid = _valid_manifest()
    del invalid["kind"]
    with pytest.raises(jsonschema.ValidationError):
        write_pack_manifest(invalid, str(tmp_path))
    assert not (tmp_path / PACK_MANIFEST_FILENAME).exists()


def test_writer_rejects_an_unknown_pack_kind(tmp_path):
    invalid = _valid_manifest()
    invalid["kind"] = "medium"
    with pytest.raises(jsonschema.ValidationError):
        write_pack_manifest(invalid, str(tmp_path))


def test_manifest_carries_the_copernicus_marine_attribution():
    manifest = _valid_manifest()
    assert manifest["attribution"] == COPERNICUS_MARINE_CREDIT
    assert manifest["disclaimer"] == COPERNICUS_MARINE_DISCLAIMER
    assert "GLOBAL_MULTIYEAR_PHY_001_030" in manifest["source_products"].values()


def test_absent_baselines_are_flagged():
    manifest = _valid_manifest()
    assert manifest["baselines_available"] is False
    assert manifest["contents"]["baselines"] == {}
    assert any("baseline" in note.lower() for note in manifest["notes"])


def test_readme_embeds_attribution_and_disclaimer_verbatim():
    readme = _pack_readme(_valid_manifest())
    assert COPERNICUS_MARINE_CREDIT in readme
    assert COPERNICUS_MARINE_DISCLAIMER in readme
    assert "No climatology / persistence baselines are bundled" in readme


def test_pack_build_result_shape():
    # The build result reports the pack directory, manifest and flags to the caller.
    result = PackBuildResult(pack_directory="p", manifest=_valid_manifest(), manifest_path="p/pack-manifest.json")
    assert result.flags == []
    assert result.manifest["kind"] == "quick"
