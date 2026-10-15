# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
from concurrent.futures import ThreadPoolExecutor

from helpers.published_versions import DISPLAYED_YEAR_VERSIONS
from helpers.s3_discovery import (
    MAXIMUM_PARALLEL_REQUESTS,
    available_versions,
    default_version,
    discover_downloaded_reports,
    discover_official_reports,
    download_notebook,
)

SCRIPT_DIRECTORY = os.path.dirname(__file__)
REPORTS_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "reports")
QUARTO_METADATA_FILE_PATH = os.path.join(REPORTS_DIRECTORY, "_metadata.yml")


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
    downloaded_reports = discover_downloaded_reports(REPORTS_DIRECTORY, version)
    pending_reports = [
        (challenger_name, region_id)
        for region_id, challenger_names in published_reports.items()
        for challenger_name in challenger_names
        if challenger_name not in downloaded_reports.get(region_id, [])
    ]

    def download(report: tuple[str, str]) -> str | None:
        challenger_name, region_id = report
        return download_notebook(version, challenger_name, region_id, version_directory)

    with ThreadPoolExecutor(max_workers=MAXIMUM_PARALLEL_REQUESTS) as pool:
        downloaded_paths = list(pool.map(download, pending_reports))

    for (challenger_name, region_id), downloaded_path in zip(pending_reports, downloaded_paths):
        if not downloaded_path:
            raise RuntimeError(f"Failed to download notebook for {version}/{challenger_name}.{region_id}.")
        print(f"Downloaded {version}/{challenger_name}.{region_id} -> {downloaded_path}")
    return bool(pending_reports)


def main() -> None:
    os.makedirs(REPORTS_DIRECTORY, exist_ok=True)
    available_version_set = set(available_versions())
    report_versions = [version for version in DISPLAYED_YEAR_VERSIONS if version in available_version_set]
    declared_default = default_version()
    active_version = declared_default if declared_default in report_versions else next(iter(report_versions), "")

    has_reports = False
    for version in report_versions:
        if version == active_version:
            _clear_version_report_notebooks(os.path.join(REPORTS_DIRECTORY, version))
        if _download_version_reports(version) or any(discover_downloaded_reports(REPORTS_DIRECTORY, version).values()):
            has_reports = True

    if not has_reports:
        raise RuntimeError("No evaluation reports were discovered in the official bucket.")

    with open(QUARTO_METADATA_FILE_PATH, "w") as file:
        file.write("execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n")
    print(f"Created {QUARTO_METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
