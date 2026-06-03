# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import math
import re
from deepdiff import DeepDiff
import sys

NOTEBOOK_TEXT_FLOAT_ABSOLUTE_TOLERANCE = 1e-5

NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?(?![A-Za-z0-9_])"
)


def normalize_value(value):
    if isinstance(value, str):
        value = re.sub(r"datetime64\[(ns|us|ms|s)\]", "datetime64", value)
        value = re.sub(r"^'\d+\.\d+\.\d+'$", "'VERSION'", value)
    return value


def ignore_ids_and_execution(json_data):
    if isinstance(json_data, dict):
        new_dict = {}
        for key, value in json_data.items():
            if key in ["id", "metadata", "text/html", "execution_count"]:
                continue
            else:
                new_dict[key] = ignore_ids_and_execution(value)
        return new_dict
    elif isinstance(json_data, list):
        return [ignore_ids_and_execution(item) for item in json_data]
    else:
        return normalize_value(json_data)


def _split_numeric_text(value: str) -> tuple[list[str], list[float]]:
    text_parts = []
    numbers = []
    previous_match_end = 0

    for match in NUMBER_PATTERN.finditer(value):
        text_parts.append(value[previous_match_end : match.start()])
        numbers.append(float(match.group(0)))
        previous_match_end = match.end()

    text_parts.append(value[previous_match_end:])

    return text_parts, numbers


def _strings_differ_only_by_tolerated_float_rounding(first_value: str, second_value: str) -> bool:
    first_text_parts, first_numbers = _split_numeric_text(first_value)
    second_text_parts, second_numbers = _split_numeric_text(second_value)

    return (
        first_text_parts == second_text_parts
        and len(first_numbers) == len(second_numbers)
        and len(first_numbers) > 0
        and all(
            math.isclose(
                first_number,
                second_number,
                rel_tol=0.0,
                abs_tol=NOTEBOOK_TEXT_FLOAT_ABSOLUTE_TOLERANCE,
            )
            for first_number, second_number in zip(first_numbers, second_numbers)
        )
    )


def _remove_tolerated_text_float_changes(diff: DeepDiff) -> DeepDiff:
    values_changed = diff.get("values_changed", {})

    for path, change in list(values_changed.items()):
        old_value = change.get("old_value")
        new_value = change.get("new_value")
        if (
            isinstance(old_value, str)
            and isinstance(new_value, str)
            and _strings_differ_only_by_tolerated_float_rounding(old_value, new_value)
        ):
            del values_changed[path]

    if "values_changed" in diff and not diff["values_changed"]:
        del diff["values_changed"]

    return diff


def compare_notebook_files(file1_path, file2_path):
    try:
        with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
            json_data1 = json.load(file1)
            json_data2 = json.load(file2)

            # Preprocess data to ignore 'id' and 'execution_count'
            filtered_json_data1 = ignore_ids_and_execution(json_data1)
            filtered_json_data2 = ignore_ids_and_execution(json_data2)

            diff = DeepDiff(filtered_json_data1, filtered_json_data2)
            diff = _remove_tolerated_text_float_changes(diff)

            if diff:
                print(diff.to_json(indent=2))
                sys.exit(1)
            else:
                print("{}")
                sys.exit(0)

    except FileNotFoundError as e:
        print(f"Error: One of the files was not found. {e}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_notebook.py <file1.json> <file2.json>")
    else:
        file1_path, file2_path = sys.argv[1], sys.argv[2]
        compare_notebook_files(file1_path, file2_path)
