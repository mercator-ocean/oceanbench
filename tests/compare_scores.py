# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import sys

SCORES_RELATIVE_TOLERANCE = 1e-6
# Reference scores converted from report notebooks keep the 6 printed decimal places, so one unit in the last
# decimal is 1e-6 whatever the magnitude.
SCORES_ABSOLUTE_TOLERANCE = 1e-6

COMPARED_HEADER_FIELDS = ["challenger", "region", "start_dates", "lead_days_count"]
ROW_KEY_FIELDS = ["metric", "reference", "variable", "depth", "lead_day"]
TOLERATED_ROW_FIELDS = {"value", "missing_fraction"}


def _numbers_are_within_tolerance(first_number: float | None, second_number: float | None) -> bool:
    if first_number is None or second_number is None:
        return first_number is second_number
    tolerance = SCORES_ABSOLUTE_TOLERANCE + SCORES_RELATIVE_TOLERANCE * max(abs(first_number), abs(second_number))
    return abs(first_number - second_number) <= tolerance


def _rows_by_key(scores: dict) -> dict[tuple, dict]:
    return {tuple(row[field] for field in ROW_KEY_FIELDS): row for row in scores["rows"]}


def _row_field_matches(field: str, first_row: dict, second_row: dict) -> bool:
    if field in TOLERATED_ROW_FIELDS:
        return _numbers_are_within_tolerance(first_row.get(field), second_row.get(field))
    return first_row.get(field) == second_row.get(field)


def scores_differences(reference_scores: dict, scores: dict) -> list[str]:
    differences = [
        f"{field}: {reference_scores[field]!r} != {scores[field]!r}"
        for field in COMPARED_HEADER_FIELDS
        if reference_scores[field] != scores[field]
    ]
    reference_rows = _rows_by_key(reference_scores)
    rows = _rows_by_key(scores)
    differences += [f"missing row {key}" for key in sorted(reference_rows.keys() - rows.keys())]
    differences += [f"unexpected row {key}" for key in sorted(rows.keys() - reference_rows.keys())]
    differences += [
        f"{key} {field}: {reference_rows[key].get(field)!r} != {rows[key].get(field)!r}"
        for key in sorted(reference_rows.keys() & rows.keys())
        for field in sorted(reference_rows[key].keys() | rows[key].keys())
        if not _row_field_matches(field, reference_rows[key], rows[key])
    ]
    return differences


def compare_scores_files(reference_file_path: str, file_path: str) -> int:
    with open(reference_file_path, encoding="utf-8") as reference_file, open(file_path, encoding="utf-8") as file:
        differences = scores_differences(json.load(reference_file), json.load(file))
    print("\n".join(differences) if differences else "No difference")
    return 1 if differences else 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_scores.py <reference.scores.json> <file.scores.json>")
        sys.exit(1)
    sys.exit(compare_scores_files(sys.argv[1], sys.argv[2]))
