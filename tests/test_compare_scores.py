# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import copy
import importlib.util
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_compare_scores_module():
    module_path = Path(__file__).with_name("compare_scores.py")
    spec = importlib.util.spec_from_file_location("compare_scores", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


compare_scores = _load_compare_scores_module()

REFERENCE_SCORES = json.loads((PROJECT_ROOT / "assets" / "glonet_sample.global.scores.json").read_text("utf-8"))


def _scores_with_first_row(**fields) -> dict:
    scores = copy.deepcopy(REFERENCE_SCORES)
    scores["rows"][0] |= fields
    return scores


def test_identical_scores_have_no_difference() -> None:
    assert compare_scores.scores_differences(REFERENCE_SCORES, copy.deepcopy(REFERENCE_SCORES)) == []


def test_tolerates_drift_in_the_last_printed_decimal() -> None:
    value = REFERENCE_SCORES["rows"][0]["value"]

    assert compare_scores.scores_differences(REFERENCE_SCORES, _scores_with_first_row(value=value + 1e-6)) == []


def test_rejects_material_value_changes() -> None:
    value = REFERENCE_SCORES["rows"][0]["value"]

    differences = compare_scores.scores_differences(REFERENCE_SCORES, _scores_with_first_row(value=value + 1e-4))

    assert len(differences) == 1
    assert "value" in differences[0]


def test_rejects_a_missing_value_and_a_changed_observation_count() -> None:
    observation_row_index = next(
        index for index, row in enumerate(REFERENCE_SCORES["rows"]) if row["reference"] == "observations"
    )
    scores = _scores_with_first_row(value=None)
    scores["rows"][observation_row_index]["observations"] += 1

    assert len(compare_scores.scores_differences(REFERENCE_SCORES, scores)) == 2


def test_reports_missing_rows_and_header_changes() -> None:
    scores = copy.deepcopy(REFERENCE_SCORES)
    removed_row = scores["rows"].pop()
    scores["start_dates"] = ["2024-01-10"]

    differences = compare_scores.scores_differences(REFERENCE_SCORES, scores)

    assert differences[0].startswith("start_dates:")
    assert differences[1] == f"missing row {tuple(removed_row[field] for field in compare_scores.ROW_KEY_FIELDS)}"
