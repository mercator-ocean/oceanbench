# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import os
import re
from pathlib import PurePosixPath

import requests

S3_BASE_URL = "https://minio.dive.edito.eu/project-oceanbench"
REPORTS_ROOT_PREFIX = "public/evaluation-reports"
REPORT_INDEX_FILE_NAME = "_index.json"
REPORT_VERSION_METADATA_FILE_NAME = "_metadata.json"
REPORT_FILE_PATTERN = re.compile(r"^(?P<challenger>.+)\.(?P<region>[a-z0-9_-]+)\.report\.ipynb$")
VERSION_PATTERN = re.compile(r"^\d+\.\d+\.\d+$")


def _reports_url(*path_parts: str) -> str:
    path = "/".join([REPORTS_ROOT_PREFIX, *path_parts])
    return f"{S3_BASE_URL}/{path}"


def _request_json(url: str) -> dict:
    response = requests.get(url, timeout=30)
    if response.status_code != 200:
        raise RuntimeError(f"Failed to download {url}: HTTP {response.status_code}")
    return json.loads(response.text)


def _version_sort_key(version: str) -> tuple[int, ...]:
    if VERSION_PATTERN.match(version) is None:
        raise ValueError(f"Report version must use X.Y.Z semantic versioning: {version!r}")
    return tuple(map(int, version.split(".")))


def _versions_newest_first(versions: list[str]) -> list[str]:
    return sorted(versions, key=_version_sort_key, reverse=True)


def _normalise_report_file_path(file_path: str) -> str:
    path = PurePosixPath(file_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Report file path must be relative to the version directory: {file_path!r}")
    if not file_path.endswith(".report.ipynb"):
        raise ValueError(f"Report file path must point to an evaluation notebook: {file_path!r}")
    return path.as_posix()


def _normalise_report_entry(report: dict) -> dict:
    challenger = str(report.get("challenger", "")).strip()
    region = str(report.get("region", "")).strip()
    if not challenger:
        raise ValueError(f"Report entry must define a challenger: {report!r}")
    if not region:
        raise ValueError(f"Report entry must define a region: {report!r}")

    file_path = report.get("file") or f"{challenger}.{region}.report.ipynb"
    normalised_report = dict(report)
    normalised_report["challenger"] = challenger
    normalised_report["region"] = region
    normalised_report["file"] = _normalise_report_file_path(str(file_path))
    return normalised_report


def _normalise_version_metadata(metadata: dict, expected_version: str | None = None) -> dict:
    version = str(metadata.get("version", "")).strip()
    if not version:
        raise ValueError("Report version metadata must define a version.")
    if expected_version is not None and version != expected_version:
        raise ValueError(f"Report version metadata mismatch: expected {expected_version!r}, got {version!r}.")

    reports = [_normalise_report_entry(report) for report in metadata.get("reports", [])]
    if not reports:
        raise ValueError(f"Report version {version!r} does not declare any reports.")
    report_keys = [(report["challenger"], report["region"]) for report in reports]
    if len(report_keys) != len(set(report_keys)):
        raise ValueError(f"Report version {version!r} declares duplicate challenger/region reports.")

    normalised_metadata = dict(metadata)
    normalised_metadata["version"] = version
    normalised_metadata["reports"] = reports
    normalised_metadata.setdefault("regions", {})
    normalised_metadata.setdefault("challengers", {})
    return normalised_metadata


def _index_versions(report_index: dict) -> list[str]:
    raw_versions = report_index.get("versions", [])
    versions = [
        str(version_entry.get("version") if isinstance(version_entry, dict) else version_entry).strip()
        for version_entry in raw_versions
    ]
    versions = [version for version in versions if version]
    if not versions:
        raise RuntimeError("Report index does not declare any versions.")
    return _versions_newest_first(versions)


def discover_official_report_versions() -> list[dict]:
    report_index = _request_json(_reports_url(REPORT_INDEX_FILE_NAME))
    report_versions = []
    for version in _index_versions(report_index):
        try:
            metadata = _request_json(_reports_url(version, REPORT_VERSION_METADATA_FILE_NAME))
            report_versions.append(_normalise_version_metadata(metadata, expected_version=version))
        except Exception as error:
            print(f"Skipping evaluation report version {version}: {error}")
    if not report_versions:
        raise RuntimeError("No evaluation report version metadata was discovered in the bucket.")
    return report_versions


def report_file_name(report: dict) -> str:
    return os.path.basename(report["file"])


def report_local_path(reports_directory: str, version_metadata: dict, report: dict) -> str:
    return os.path.join(reports_directory, version_metadata["version"], report["file"])


def download_report_notebook(
    version_metadata: dict,
    report: dict,
    destination_directory: str,
) -> str | None:
    version = version_metadata["version"]
    destination_path = os.path.join(destination_directory, report["file"])
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    url = _reports_url(version, report["file"])

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        with open(destination_path, "wb") as file:
            file.write(response.content)
        return destination_path
    except Exception as error:
        print(f"Failed to download {report['file']} from {url}: {error}")
    return None


def load_downloaded_report_versions(reports_directory: str) -> list[dict]:
    if not os.path.isdir(reports_directory):
        return []

    report_versions = []
    for directory_name in os.listdir(reports_directory):
        version_directory = os.path.join(reports_directory, directory_name)
        metadata_path = os.path.join(version_directory, REPORT_VERSION_METADATA_FILE_NAME)
        if not os.path.isfile(metadata_path):
            continue
        with open(metadata_path, encoding="utf-8") as file:
            metadata = json.load(file)
        report_versions.append(_normalise_version_metadata(metadata, expected_version=directory_name))

    return sorted(report_versions, key=lambda metadata: _version_sort_key(metadata["version"]), reverse=True)


def reports_by_region(version_metadata: dict) -> dict[str, list[str]]:
    discovered_reports: dict[str, list[str]] = {}
    for report in version_metadata["reports"]:
        region = report["region"]
        discovered_reports.setdefault(region, [])
        discovered_reports[region].append(report["challenger"])
    return discovered_reports


def report_by_challenger_region(version_metadata: dict) -> dict[tuple[str, str], dict]:
    return {(report["challenger"], report["region"]): report for report in version_metadata["reports"]}


def discover_downloaded_reports(reports_directory: str) -> dict[str, list[str]]:
    report_versions = load_downloaded_report_versions(reports_directory)
    if not report_versions:
        return {}
    return reports_by_region(report_versions[0])
