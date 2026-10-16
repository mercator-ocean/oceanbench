# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[1]
WEBSITE_DIRECTORY = REPOSITORY_DIRECTORY / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.notebook_score_parser import get_all_model_scores_from_notebook  # noqa: E402
from helpers.scores_file_reader import get_all_model_scores_from_scores_file  # noqa: E402

# The metric cells of the 0.6.0 GLONET sample report notebook that assets/glonet_sample.global.scores.json
# was converted from.
NOTEBOOK_PATH = REPOSITORY_DIRECTORY / "tests" / "fixtures" / "glonet_sample.global.report.ipynb"
SCORES_FILE_PATH = REPOSITORY_DIRECTORY / "assets" / "glonet_sample.global.scores.json"


def _serialised_scores(scores: dict) -> str:
    return json.dumps({metric_key: score.model_dump() for metric_key, score in scores.items()})


def test_reader_gives_the_scores_the_parser_gives_for_the_same_report():
    notebook_scores = get_all_model_scores_from_notebook(str(NOTEBOOK_PATH), "glonet_sample")
    scores_file_scores = get_all_model_scores_from_scores_file(str(SCORES_FILE_PATH), "glonet_sample")

    assert len(notebook_scores) == 9
    assert _serialised_scores(scores_file_scores) == _serialised_scores(notebook_scores)


def test_reader_gives_an_empty_score_for_a_metric_without_rows(tmp_path):
    scores_file = json.loads(SCORES_FILE_PATH.read_text(encoding="utf-8"))
    scores_file["rows"] = [row for row in scores_file["rows"] if row["reference"] != "observations"]
    scores_file_path = tmp_path / "glonet_sample.global.scores.json"
    scores_file_path.write_text(json.dumps(scores_file), encoding="utf-8")

    scores = get_all_model_scores_from_scores_file(str(scores_file_path), "glonet_sample")

    assert scores["rmsd_variables_observations"].depths == {}
