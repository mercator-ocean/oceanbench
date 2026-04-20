# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os

from helpers.s3_discovery import discover_challengers, download_notebook

SCRIPT_DIRECTORY = os.path.dirname(__file__)
REPORTS_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "reports")
QUARTO_METADATA_FILE_PATH = os.path.join(REPORTS_DIRECTORY, "_metadata.yml")


def _clear_report_notebooks() -> None:
    if not os.path.isdir(REPORTS_DIRECTORY):
        return
    for file_name in os.listdir(REPORTS_DIRECTORY):
        if file_name.endswith(".report.ipynb"):
            os.remove(os.path.join(REPORTS_DIRECTORY, file_name))


def main() -> None:
    challengers = discover_challengers()
    print(f"Discovered challengers: {challengers}")

    os.makedirs(REPORTS_DIRECTORY, exist_ok=True)
    _clear_report_notebooks()

    for challenger_name in challengers:
        destination = os.path.join(REPORTS_DIRECTORY, f"{challenger_name}.report.ipynb")
        print(f"Downloading {challenger_name}...", end=" ")
        result = download_notebook(challenger_name, REPORTS_DIRECTORY)
        if result:
            print(f"OK -> {result}")
            continue
        if os.path.exists(destination):
            os.remove(destination)
        print("FAILED")
        raise RuntimeError(f"Failed to download notebook for {challenger_name}.")

    with open(QUARTO_METADATA_FILE_PATH, "w") as file:
        file.write("execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n")
    print(f"Created {QUARTO_METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
