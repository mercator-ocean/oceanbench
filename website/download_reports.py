# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os

from helpers.s3_discovery import available_versions, discover_official_reports, download_notebook

SCRIPT_DIRECTORY = os.path.dirname(__file__)
REPORTS_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "reports")
QUARTO_METADATA_FILE_PATH = os.path.join(REPORTS_DIRECTORY, "_metadata.yml")


def _clear_report_notebooks() -> None:
    if not os.path.isdir(REPORTS_DIRECTORY):
        return
    for directory, _subdirectories, file_names in os.walk(REPORTS_DIRECTORY):
        for file_name in file_names:
            if file_name.endswith(".report.ipynb"):
                os.remove(os.path.join(directory, file_name))


def main() -> None:
    os.makedirs(REPORTS_DIRECTORY, exist_ok=True)
    _clear_report_notebooks()

    downloaded_any_report = False
    for version in available_versions():
        published_reports = discover_official_reports(version)
        print(f"[{version}] discovered reports: {published_reports}")
        version_directory = os.path.join(REPORTS_DIRECTORY, version)
        for region_id, challengers in published_reports.items():
            for challenger_name in challengers:
                print(f"Downloading {version}/{challenger_name}.{region_id}...", end=" ")
                result = download_notebook(version, challenger_name, region_id, version_directory)
                if result:
                    downloaded_any_report = True
                    print(f"OK -> {result}")
                    continue
                print("FAILED")
                raise RuntimeError(f"Failed to download notebook for {version}/{challenger_name}.{region_id}.")

    if not downloaded_any_report:
        raise RuntimeError("No evaluation reports were discovered in the official bucket.")

    with open(QUARTO_METADATA_FILE_PATH, "w") as file:
        file.write("execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n")
    print(f"Created {QUARTO_METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
