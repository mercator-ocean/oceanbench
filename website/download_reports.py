# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import os
import shutil
import tempfile

from helpers.notebook_score_parser import get_all_model_scores_from_notebook
from helpers.s3_discovery import (
    REPORT_VERSION_METADATA_FILE_NAME,
    discover_official_report_versions,
    download_report_notebook,
    report_local_path,
)

SCRIPT_DIRECTORY = os.path.dirname(__file__)
REPORTS_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "reports")
QUARTO_METADATA_FILE_NAME = "_metadata.yml"
QUARTO_METADATA_CONTENT = "execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n"
QUARTO_METADATA_FILE_PATH = os.path.join(REPORTS_DIRECTORY, QUARTO_METADATA_FILE_NAME)


def _clear_reports_directory() -> None:
    if os.path.isdir(REPORTS_DIRECTORY):
        shutil.rmtree(REPORTS_DIRECTORY)
    os.makedirs(REPORTS_DIRECTORY, exist_ok=True)


def _write_quarto_metadata(directory: str) -> None:
    with open(os.path.join(directory, QUARTO_METADATA_FILE_NAME), "w", encoding="utf-8") as file:
        file.write(QUARTO_METADATA_CONTENT)


def _write_report_version_metadata(version_directory: str, version_metadata: dict) -> None:
    with open(
        os.path.join(version_directory, REPORT_VERSION_METADATA_FILE_NAME),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(version_metadata, file, indent=2, sort_keys=True)


def _validate_report_notebooks(reports_directory: str, version_metadata: dict) -> None:
    for report in version_metadata["reports"]:
        report_path = report_local_path(reports_directory, version_metadata, report)
        scores = get_all_model_scores_from_notebook(report_path, report["challenger"])
        if not scores:
            raise RuntimeError(f"No scores were parsed from {report_path}.")


def _download_report_version(version_metadata: dict, destination_directory: str) -> dict | None:
    version = version_metadata["version"]
    version_directory = os.path.join(destination_directory, version)
    os.makedirs(version_directory, exist_ok=True)
    print(f"Checking evaluation report version {version}")

    for report in version_metadata["reports"]:
        print(f"Downloading {version}/{report['file']}...", end=" ")
        result = download_report_notebook(version_metadata, report, version_directory)
        if result:
            print(f"OK -> {result}")
            continue
        print("FAILED")
        return None

    _validate_report_notebooks(destination_directory, version_metadata)
    _write_report_version_metadata(version_directory, version_metadata)
    _write_quarto_metadata(version_directory)
    return version_metadata


def _install_report_versions(source_directory: str, version_metadata_list: list[dict]) -> None:
    _clear_reports_directory()
    _write_quarto_metadata(REPORTS_DIRECTORY)
    for version_metadata in version_metadata_list:
        version = version_metadata["version"]
        shutil.copytree(
            os.path.join(source_directory, version),
            os.path.join(REPORTS_DIRECTORY, version),
        )


def _download_valid_report_versions() -> list[dict]:
    valid_report_versions = []
    with tempfile.TemporaryDirectory() as candidate_directory:
        for version_metadata in discover_official_report_versions():
            try:
                valid_version_metadata = _download_report_version(version_metadata, candidate_directory)
            except Exception as error:
                print(f"Skipping evaluation report version {version_metadata.get('version')}: {error}")
                continue
            if valid_version_metadata is not None:
                valid_report_versions.append(valid_version_metadata)

        if not valid_report_versions:
            raise RuntimeError("No complete and parseable evaluation report version was discovered in the bucket.")

        _install_report_versions(candidate_directory, valid_report_versions)
    return valid_report_versions


def main() -> None:
    valid_report_versions = _download_valid_report_versions()
    print(f"Installed report versions: {[metadata['version'] for metadata in valid_report_versions]}")
    print(f"Created {QUARTO_METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
