# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[1]
WEBSITE_DIRECTORY = REPOSITORY_DIRECTORY / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.scores_file_reader import get_all_model_scores_from_scores_file  # noqa: E402

SCORES_FILE_PATH = REPOSITORY_DIRECTORY / "assets" / "glonet_sample.global.scores.json"


def test_reader_gives_a_score_for_each_metric_of_the_scores_file():
    scores = get_all_model_scores_from_scores_file(str(SCORES_FILE_PATH), "glonet_sample")

    assert len(scores) == 9
    assert scores["rmsd_variables_glorys"].depths["Surface"].variables["temperature"].unit


def test_reader_gives_an_empty_score_for_a_metric_without_rows(tmp_path):
    scores_file = json.loads(SCORES_FILE_PATH.read_text(encoding="utf-8"))
    scores_file["rows"] = [row for row in scores_file["rows"] if row["reference"] != "observations"]
    scores_file_path = tmp_path / "glonet_sample.global.scores.json"
    scores_file_path.write_text(json.dumps(scores_file), encoding="utf-8")

    scores = get_all_model_scores_from_scores_file(str(scores_file_path), "glonet_sample")

    assert scores["rmsd_variables_observations"].depths == {}
