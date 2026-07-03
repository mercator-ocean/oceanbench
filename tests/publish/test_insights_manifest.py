# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

import jsonschema
import pytest

from oceanbench.core.schema_validation import load_schema
from oceanbench.publish.insights_manifest import InsightArtifact, write_insights_manifest


def _artifact(tmp_path, semantic_key: str, kind: str, data: bytes) -> InsightArtifact:
    source = tmp_path / f"{semantic_key}.blob"
    source.write_bytes(data)
    return InsightArtifact(semantic_key=semantic_key, kind=kind, schema_version="1.0", source_path=str(source))


def test_written_manifest_is_schema_valid_and_content_addressed(tmp_path):
    artifacts = [
        _artifact(tmp_path, "class4-matchups", "class4-matchups", b"matchup rows"),
        _artifact(tmp_path, "spectra", "spectra", b"spectra json"),
    ]
    output_directory = tmp_path / "insights"
    result = write_insights_manifest(artifacts, str(output_directory), base_url="https://example.org/insights")

    jsonschema.validate(json.loads((output_directory / "manifest.json").read_text()), load_schema("insights-manifest"))
    assert set(result.manifest) == {"class4-matchups", "spectra"}
    for entry in result.manifest.values():
        blob_name = entry["url"].rsplit("/", 1)[-1]
        assert (output_directory / blob_name).exists()
        assert entry["bytes"] == (output_directory / blob_name).stat().st_size


def test_relative_urls_when_no_base_url(tmp_path):
    result = write_insights_manifest([_artifact(tmp_path, "eddies", "eddies", b"eddies")], str(tmp_path / "insights"))
    assert "/" not in result.manifest["eddies"]["url"]


def test_unknown_kind_is_rejected(tmp_path):
    with pytest.raises(jsonschema.ValidationError):
        write_insights_manifest([_artifact(tmp_path, "mystery", "not-a-kind", b"x")], str(tmp_path / "insights"))


def test_empty_artifacts_rejected(tmp_path):
    with pytest.raises(ValueError):
        write_insights_manifest([], str(tmp_path / "insights"))
