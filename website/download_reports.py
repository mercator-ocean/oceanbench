# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os

from helpers.s3_discovery import discover_official_reports_by_year, download_notebook

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

    published_reports_by_year = discover_official_reports_by_year()
    print(f"Discovered reports: {published_reports_by_year}")
    if not any(
        challengers
        for published_reports in published_reports_by_year.values()
        for challengers in published_reports.values()
    ):
        raise RuntimeError("No evaluation reports were discovered in the official bucket.")

    _clear_report_notebooks()

    for year, published_reports in published_reports_by_year.items():
        for region_id, challengers in published_reports.items():
            for challenger_name in challengers:
                print(f"Downloading {year}.{challenger_name}.{region_id}...", end=" ")
                result = download_notebook(challenger_name, region_id, REPORTS_DIRECTORY, year)
                if result:
                    print(f"OK -> {result}")
                    continue
                print("FAILED")
                raise RuntimeError(f"Failed to download notebook for {year}.{challenger_name}.{region_id}.")

    with open(QUARTO_METADATA_FILE_PATH, "w") as file:
        file.write("execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n")
    print(f"Created {QUARTO_METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
