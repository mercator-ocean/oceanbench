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


def test_reader_gives_the_smoothing_controlled_rmsd_and_its_diagnostics(tmp_path):
    scores_file = json.loads(SCORES_FILE_PATH.read_text(encoding="utf-8"))
    observation_rows = [row for row in scores_file["rows"] if row["reference"] == "observations"]
    scores_file["rows"] += [
        row | {"metric": "smoothing_controlled_rmsd_of_variables", "smoothing_width_km": 30.0, "smoothing_slope": 0.8}
        for row in observation_rows
    ]
    scores_file_path = tmp_path / "glonet_sample.global.scores.json"
    scores_file_path.write_text(json.dumps(scores_file), encoding="utf-8")

    scores = get_all_model_scores_from_scores_file(str(scores_file_path), "glonet_sample")

    first_row = observation_rows[0]
    depth, variable, lead_day = first_row["depth"].capitalize(), first_row["variable"], str(first_row["lead_day"])
    controlled = scores["smoothing_controlled_rmsd_variables_observations"].depths[depth].variables[variable]
    assert controlled.data[lead_day] == first_row["value"]
    width = scores["smoothing_width_observations"].depths[depth].variables[variable]
    assert (width.unit, width.data[lead_day]) == ("km", 30.0)
    assert scores["smoothing_slope_observations"].depths[depth].variables[variable].data[lead_day] == 0.8


def test_reader_gives_the_smoothing_controlled_rmsd_against_a_gridded_reference(tmp_path):
    scores_file = json.loads(SCORES_FILE_PATH.read_text(encoding="utf-8"))
    glorys_rows = [
        row
        for row in scores_file["rows"]
        if (row["metric"], row["reference"]) == ("rmsd_of_variables", "glorys_reanalysis")
    ]
    scores_file["rows"] += [
        row | {"metric": "smoothing_controlled_rmsd_of_variables", "smoothing_width_km": 45.0, "smoothing_slope": 0.9}
        for row in glorys_rows
    ]
    scores_file_path = tmp_path / "glonet_sample.global.scores.json"
    scores_file_path.write_text(json.dumps(scores_file), encoding="utf-8")

    scores = get_all_model_scores_from_scores_file(str(scores_file_path), "glonet_sample")

    first_row = glorys_rows[0]
    depth, variable, lead_day = first_row["depth"].capitalize(), first_row["variable"], str(first_row["lead_day"])
    controlled = scores["smoothing_controlled_rmsd_variables_glorys"].depths[depth].variables[variable]
    assert controlled.data[lead_day] == first_row["value"]
    assert scores["smoothing_width_glorys"].depths[depth].variables[variable].data[lead_day] == 45.0
    assert "smoothing_width_observations" not in scores


def test_each_rmsd_section_names_its_smoothing_controlled_metrics():
    from helpers.metric_definitions import SECTIONS

    assert set(SECTIONS) == {"observations", "reanalysis", "analysis"}
    assert SECTIONS["reanalysis"]["controlled_metrics"] == {
        "value": "smoothing_controlled_rmsd_variables_glorys",
        "width": "smoothing_width_glorys",
        "slope": "smoothing_slope_glorys",
    }
