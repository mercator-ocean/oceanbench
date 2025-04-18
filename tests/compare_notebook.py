# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from deepdiff import DeepDiff
import sys


def ignore_ids_and_execution(json_data):
    if isinstance(json_data, dict):
        new_dict = {}
        for key, value in json_data.items():
            if key == "cells":
                # Special handling for cells
                new_dict[key] = [ignore_ids_and_execution(cell) for cell in value]
            elif key in ["id", "metadata", "text/html"]:
                continue
            else:
                new_dict[key] = ignore_ids_and_execution(value)
        return new_dict
    elif isinstance(json_data, list):
        return [ignore_ids_and_execution(item) for item in json_data]
    else:
        return json_data


def compare_notebook_files(file1_path, file2_path):
    try:
        with open(file1_path, "r") as file1, open(file2_path, "r") as file2:
            json_data1 = json.load(file1)
            json_data2 = json.load(file2)

            # Preprocess data to ignore 'id' and 'execution_count'
            filtered_json_data1 = ignore_ids_and_execution(json_data1)
            filtered_json_data2 = ignore_ids_and_execution(json_data2)

            diff = DeepDiff(filtered_json_data1, filtered_json_data2)

            if diff:
                # Convertir en JSON string avec gestion des types non sérialisables
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
