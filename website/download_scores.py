# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import shutil
from concurrent.futures import ThreadPoolExecutor

from helpers.s3_discovery import (
    DOWNLOADED_SCORES_DIRECTORY,
    MAXIMUM_PARALLEL_REQUESTS,
    available_versions,
    challengers_for_version,
    default_version,
    discover_downloaded_scores_files,
    discover_published_scores_files,
    download_scores_file,
)

SCRIPT_DIRECTORY = os.path.dirname(__file__)
SCORES_FILES_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, DOWNLOADED_SCORES_DIRECTORY)


def _version_already_downloaded(version: str) -> bool:
    return any(discover_downloaded_scores_files(SCORES_FILES_DIRECTORY, version).values())


def _clear_version_scores_files(version_directory: str) -> None:
    shutil.rmtree(version_directory, ignore_errors=True)


def _challengers_without_scores_files(version: str, published_scores_files: dict[str, dict[str, str]]) -> list[str]:
    published_challengers = {
        challenger_name
        for region_scores_files in published_scores_files.values()
        for challenger_name in region_scores_files
    }
    return [
        challenger_name
        for challenger_name in challengers_for_version(version)
        if challenger_name not in published_challengers
    ]


def _download_version_scores_files(version: str) -> bool:
    version_directory = os.path.join(SCORES_FILES_DIRECTORY, version)
    published_scores_files = discover_published_scores_files(version)
    print(f"[{version}] discovered scores files: {published_scores_files}")
    missing_challengers = _challengers_without_scores_files(version, published_scores_files)
    if missing_challengers:
        raise RuntimeError(f"No scores file in any region for {', '.join(missing_challengers)} in version {version}.")
    pending_file_names = [
        file_name for region_file_names in published_scores_files.values() for file_name in region_file_names.values()
    ]

    def download(file_name: str) -> str | None:
        return download_scores_file(version, file_name, version_directory)

    with ThreadPoolExecutor(max_workers=MAXIMUM_PARALLEL_REQUESTS) as pool:
        downloaded_paths = list(pool.map(download, pending_file_names))

    for file_name, downloaded_path in zip(pending_file_names, downloaded_paths):
        if not downloaded_path:
            raise RuntimeError(f"Failed to download {version}/{file_name}.")
        print(f"Downloaded {version}/{file_name} -> {downloaded_path}")
    return bool(pending_file_names)


def main() -> None:
    os.makedirs(SCORES_FILES_DIRECTORY, exist_ok=True)
    active_version = default_version()

    has_scores_files = False
    for version in available_versions():
        if version != active_version and _version_already_downloaded(version):
            print(f"[{version}] already downloaded, keeping")
            has_scores_files = True
            continue
        if version == active_version:
            _clear_version_scores_files(os.path.join(SCORES_FILES_DIRECTORY, version))
        if _download_version_scores_files(version):
            has_scores_files = True

    if not has_scores_files:
        raise RuntimeError("No scores files were discovered in the official bucket.")


if __name__ == "__main__":
    main()
