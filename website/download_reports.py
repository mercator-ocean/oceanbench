# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from helpers.local_challengers import local_reports
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


def _version_already_downloaded(version: str) -> bool:
    return any(discover_downloaded_reports(REPORTS_DIRECTORY, version).values())


def _clear_version_report_notebooks(version_directory: str) -> None:
    if not os.path.isdir(version_directory):
        return
    for file_name in os.listdir(version_directory):
        if file_name.endswith(".report.ipynb"):
            os.remove(os.path.join(version_directory, file_name))


def _copy_local_reports(version: str) -> int:
    version_directory = os.path.join(REPORTS_DIRECTORY, version)
    copied = 0
    for challenger_name, region_id, source_path in local_reports(version):
        os.makedirs(version_directory, exist_ok=True)
        destination_path = os.path.join(version_directory, f"{challenger_name}.{region_id}.report.ipynb")
        shutil.copyfile(source_path, destination_path)
        print(f"Copied local report {version}/{challenger_name}.{region_id} -> {destination_path}")
        copied += 1
    return copied


def _download_version_reports(version: str) -> bool:
    version_directory = os.path.join(REPORTS_DIRECTORY, version)
    published_reports = discover_official_reports(version)
    print(f"[{version}] discovered reports: {published_reports}")
    pending_reports = [
        (challenger_name, region_id)
        for region_id, challenger_names in published_reports.items()
        for challenger_name in challenger_names
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
    copied_locally = _copy_local_reports(version)
    return bool(pending_reports) or bool(copied_locally)


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
