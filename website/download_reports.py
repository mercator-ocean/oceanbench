# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os

from helpers.s3_discovery import LOCAL_REPORTS_ENVIRONMENT_VARIABLE
from helpers.s3_discovery import discover_official_reports, download_notebook

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
    os.makedirs(REPORTS_DIRECTORY, exist_ok=True)

    if os.environ.get(LOCAL_REPORTS_ENVIRONMENT_VARIABLE) == "1":
        with open(QUARTO_METADATA_FILE_PATH, "w") as file:
            file.write("execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n")
        print(f"Using local report notebooks from {REPORTS_DIRECTORY}")
        print(f"Created {QUARTO_METADATA_FILE_PATH}")
        return

    published_reports = discover_official_reports()
    print(f"Discovered reports: {published_reports}")

    _clear_report_notebooks()

    for region_id, challengers in published_reports.items():
        for challenger_name in challengers:
            destination = os.path.join(REPORTS_DIRECTORY, f"{challenger_name}.{region_id}.report.ipynb")
            print(f"Downloading {challenger_name}.{region_id}...", end=" ")
            result = download_notebook(challenger_name, region_id, REPORTS_DIRECTORY)
            if result:
                print(f"OK -> {result}")
                continue
            if os.path.exists(destination):
                os.remove(destination)
            print("FAILED")
            raise RuntimeError(f"Failed to download notebook for {challenger_name}.{region_id}.")

    with open(QUARTO_METADATA_FILE_PATH, "w") as file:
        file.write("execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n")
    print(f"Created {QUARTO_METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
