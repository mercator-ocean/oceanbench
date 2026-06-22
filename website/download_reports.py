# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os

from helpers.s3_discovery import (
    available_versions,
    default_version,
    discover_downloaded_reports,
    discover_official_reports,
    download_notebook,
)

SCRIPT_DIRECTORY = os.path.dirname(__file__)
REPORTS_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "reports")
QUARTO_METADATA_FILE_PATH = os.path.join(REPORTS_DIRECTORY, "_metadata.yml")


def _version_already_downloaded(version: str) -> bool:
    return any(discover_downloaded_reports(REPORTS_DIRECTORY, version).values())


def _clear_version_report_notebooks(version_directory: str) -> None:
    if not os.path.isdir(version_directory):
        return
    for file_name in os.listdir(version_directory):
        if file_name.endswith(".report.ipynb"):
            os.remove(os.path.join(version_directory, file_name))


def _download_version_reports(version: str) -> bool:
    version_directory = os.path.join(REPORTS_DIRECTORY, version)
    published_reports = discover_official_reports(version)
    print(f"[{version}] discovered reports: {published_reports}")
    downloaded_any_report = False
    for region_id, challengers in published_reports.items():
        for challenger_name in challengers:
            print(f"Downloading {version}/{challenger_name}.{region_id}...", end=" ")
            result = download_notebook(version, challenger_name, region_id, version_directory)
            if not result:
                print("FAILED")
                raise RuntimeError(f"Failed to download notebook for {version}/{challenger_name}.{region_id}.")
            print(f"OK -> {result}")
            downloaded_any_report = True
    return downloaded_any_report


def main() -> None:
    os.makedirs(REPORTS_DIRECTORY, exist_ok=True)
    active_version = default_version()

    has_reports = False
    for version in available_versions():
        if version != active_version and _version_already_downloaded(version):
            print(f"[{version}] already downloaded, keeping")
            has_reports = True
            continue
        if version == active_version:
            _clear_version_report_notebooks(os.path.join(REPORTS_DIRECTORY, version))
        if _download_version_reports(version):
            has_reports = True

    if not has_reports:
        raise RuntimeError("No evaluation reports were discovered in the official bucket.")

    with open(QUARTO_METADATA_FILE_PATH, "w") as file:
        file.write("execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n")
    print(f"Created {QUARTO_METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
