# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
from concurrent.futures import ThreadPoolExecutor

from helpers.s3_discovery import (
    MAXIMUM_PARALLEL_REQUESTS,
    REPORT_FILE_SUFFIXES,
    available_versions,
    default_version,
    discover_downloaded_reports,
    discover_official_reports,
    download_report,
)

SCRIPT_DIRECTORY = os.path.dirname(__file__)
REPORTS_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "reports")
QUARTO_METADATA_FILE_PATH = os.path.join(REPORTS_DIRECTORY, "_metadata.yml")


def _version_already_downloaded(version: str) -> bool:
    return any(discover_downloaded_reports(REPORTS_DIRECTORY, version).values())


def _clear_version_report_files(version_directory: str) -> None:
    if not os.path.isdir(version_directory):
        return
    for file_name in os.listdir(version_directory):
        if file_name.endswith(REPORT_FILE_SUFFIXES):
            os.remove(os.path.join(version_directory, file_name))


def _download_version_reports(version: str) -> bool:
    version_directory = os.path.join(REPORTS_DIRECTORY, version)
    published_reports = discover_official_reports(version)
    print(f"[{version}] discovered reports: {published_reports}")
    pending_file_names = [
        file_name for region_file_names in published_reports.values() for file_name in region_file_names.values()
    ]

    def download(file_name: str) -> str | None:
        return download_report(version, file_name, version_directory)

    with ThreadPoolExecutor(max_workers=MAXIMUM_PARALLEL_REQUESTS) as pool:
        downloaded_paths = list(pool.map(download, pending_file_names))

    for file_name, downloaded_path in zip(pending_file_names, downloaded_paths):
        if not downloaded_path:
            raise RuntimeError(f"Failed to download {version}/{file_name}.")
        print(f"Downloaded {version}/{file_name} -> {downloaded_path}")
    return bool(pending_file_names)


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
            _clear_version_report_files(os.path.join(REPORTS_DIRECTORY, version))
        if _download_version_reports(version):
            has_reports = True

    if not has_reports:
        raise RuntimeError("No evaluation reports were discovered in the official bucket.")

    with open(QUARTO_METADATA_FILE_PATH, "w") as file:
        file.write("execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n")
    print(f"Created {QUARTO_METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
