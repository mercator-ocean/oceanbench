# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.scores_file_reader import get_all_model_scores_from_scores_file  # noqa: E402

ASSETS_DIRECTORY = Path(__file__).resolve().parents[1] / "assets"
SCHEMA_PATH = Path(__file__).resolve().parents[1] / "oceanbench" / "schemas" / "scores-v1.schema.json"
SAMPLE_SCORES_FILES = sorted(ASSETS_DIRECTORY.glob("*.scores.json"))
CHALLENGER_NAME = "your_challenger"
NODE_SCRIPT = """
const api = require(process.argv[1]);
const document = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify({
  errors: api.scoresFileErrors(document),
  scores: api.scoresFileErrors(document).length === 0 ? api.scoresFileToModelScores(document, process.argv[2]) : null,
}));
"""

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def _read_in_the_browser_code(document: dict) -> dict:
    completed = subprocess.run(
        ["node", "-e", NODE_SCRIPT, str(WEBSITE_DIRECTORY / "your-scores.js"), CHALLENGER_NAME],
        input=json.dumps(document),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def _first_sample() -> dict:
    return json.loads(SAMPLE_SCORES_FILES[0].read_text(encoding="utf-8"))


def _with_schema_version(document: dict) -> dict:
    return copy.deepcopy(document) | {"schema_version": 2}


def _with_text_value(document: dict) -> dict:
    changed = copy.deepcopy(document)
    changed["rows"][0]["value"] = "not a number"
    return changed


def _with_unknown_metric(document: dict) -> dict:
    changed = copy.deepcopy(document)
    changed["rows"][0]["metric"] = "unknown_metric"
    return changed


def _without_rows(document: dict) -> dict:
    return {key: value for key, value in document.items() if key != "rows"}


@pytest.mark.parametrize("path", SAMPLE_SCORES_FILES, ids=lambda path: path.name)
def test_browser_reads_sample_scores_files_like_the_website_build(path):
    published = {
        metric_key: score.model_dump()
        for metric_key, score in get_all_model_scores_from_scores_file(str(path), CHALLENGER_NAME).items()
        if score.depths
    }

    in_the_browser = _read_in_the_browser_code(json.loads(path.read_text(encoding="utf-8")))

    assert in_the_browser["errors"] == []
    assert in_the_browser["scores"] == published


@pytest.mark.parametrize(
    "break_document", [_with_schema_version, _with_text_value, _with_unknown_metric, _without_rows]
)
def test_browser_rejects_what_the_scores_schema_rejects(break_document):
    broken = break_document(_first_sample())
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(broken, schema)

    assert _read_in_the_browser_code(broken)["errors"] != []


COMPARED_VERSION_SCRIPT = """
const api = require(process.argv[1]);
const [fileVersion, published, compatible, forced] = JSON.parse(require("fs").readFileSync(0, "utf8"));
process.stdout.write(JSON.stringify(api.comparedVersion(fileVersion, published, compatible, forced)));
"""


def _compared_version(file_version: str, forced_version: str | None = None) -> str | None:
    completed = subprocess.run(
        ["node", "-e", COMPARED_VERSION_SCRIPT, str(WEBSITE_DIRECTORY / "your-scores.js")],
        input=json.dumps([file_version, ["0.5.0", "0.6.0"], {"0.6.1": "0.6.0"}, forced_version]),
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout)


def test_a_file_is_compared_with_the_published_scores_of_its_version():
    assert _compared_version("0.5.0") == "0.5.0"


def test_a_file_is_compared_with_the_published_scores_that_its_version_computes():
    assert _compared_version("0.6.1") == "0.6.0"


def test_a_file_of_a_version_without_comparable_scores_is_not_compared():
    assert _compared_version("0.7.0") is None


def test_a_forced_version_is_used_whatever_the_version_of_the_file():
    assert _compared_version("0.7.0", "0.6.0") == "0.6.0"
    assert _compared_version("0.6.0", "0.5.0") == "0.5.0"


def test_a_forced_version_that_is_not_published_is_not_compared():
    assert _compared_version("0.6.0", "0.9.0") is None
