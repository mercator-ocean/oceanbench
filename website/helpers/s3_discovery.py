# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from helpers.published_regions import published_region_ids

S3_BASE_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket"
REPORTS_ROOT_PREFIX = "public/evaluation-reports"
REPORT_INDEX_URL = f"{S3_BASE_URL}/{REPORTS_ROOT_PREFIX}/index.json"
SCORES_FILE_SUFFIX = ".scores.json"
NOTEBOOK_SUFFIX = ".report.ipynb"
# Versions evaluated without notebooks publish one scores file per challenger and region, earlier versions
# a report notebook. A scores file is preferred when both exist.
REPORT_FILE_SUFFIXES = (SCORES_FILE_SUFFIX, NOTEBOOK_SUFFIX)
REPORT_FILE_PATTERN = re.compile(r"^(?P<challenger>.+)\.(?P<region>[a-z0-9_-]+)(?:\.scores\.json|\.report\.ipynb)$")

# The report discovery and download steps each issue one HTTP request per challenger and region.
# Running them concurrently keeps the website rebuild well under the gateway timeout as the number
# of published versions grows.
MAXIMUM_PARALLEL_REQUESTS = 16

# index.json stored next to the reports is the single source of truth: editing it
# (or uploading a new report) publishes a version or challenger without a library release.
_report_index_cache = None


def _fetch_report_index() -> dict:
    response = requests.get(REPORT_INDEX_URL, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(f"Could not read the report index at {REPORT_INDEX_URL} (HTTP {response.status_code}).")
    return response.json()


def _report_index() -> dict:
    global _report_index_cache
    if _report_index_cache is None:
        _report_index_cache = _fetch_report_index()
    return _report_index_cache


def _version_sort_key(version: str) -> list[int]:
    return [int(component) if component.isdigit() else -1 for component in version.split(".")]


def available_versions() -> list[str]:
    versions = list(_report_index().get("versions", {}).keys())
    return sorted(versions, key=_version_sort_key, reverse=True)


def default_version() -> str:
    versions = available_versions()
    declared_default = _report_index().get("default")
    if declared_default in versions:
        return declared_default
    return versions[0] if versions else ""


def challengers_for_version(version: str) -> list[str]:
    return _report_index().get("versions", {}).get(version, {}).get("challengers", [])


def report_url(version: str, file_name: str) -> str:
    return f"{S3_BASE_URL}/{REPORTS_ROOT_PREFIX}/{version}/{file_name}"


def _report_exists(version: str, file_name: str) -> bool:
    try:
        response = requests.head(report_url(version, file_name), timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def _published_report_file_name(version: str, challenger_name: str, region_id: str) -> str | None:
    for suffix in REPORT_FILE_SUFFIXES:
        file_name = f"{challenger_name}.{region_id}{suffix}"
        if _report_exists(version, file_name):
            return file_name
    return None


def discover_official_reports(version: str) -> dict[str, dict[str, str]]:
    """Published report file name of each challenger, by region."""
    challenger_names = challengers_for_version(version)
    probes = [
        (challenger_name, region_id) for region_id in published_region_ids() for challenger_name in challenger_names
    ]

    def published_report_file_name(probe: tuple[str, str]) -> str | None:
        challenger_name, region_id = probe
        return _published_report_file_name(version, challenger_name, region_id)

    with ThreadPoolExecutor(max_workers=MAXIMUM_PARALLEL_REQUESTS) as pool:
        file_names = {
            probe: file_name
            for probe, file_name in zip(probes, pool.map(published_report_file_name, probes))
            if file_name
        }

    return {
        region_id: {
            challenger_name: file_names[(challenger_name, region_id)]
            for challenger_name in challenger_names
            if (challenger_name, region_id) in file_names
        }
        for region_id in published_region_ids()
    }


def discover_downloaded_reports(reports_directory: str, version: str) -> dict[str, list[str]]:
    version_directory = os.path.join(reports_directory, version)
    known_challengers = challengers_for_version(version)
    discovered_reports = {region_id: set() for region_id in published_region_ids()}
    if not os.path.isdir(version_directory):
        return {region_id: [] for region_id in published_region_ids()}

    for file_name in os.listdir(version_directory):
        report_match = REPORT_FILE_PATTERN.match(file_name)
        if report_match is None:
            continue
        challenger_name = report_match.group("challenger")
        region_id = report_match.group("region")
        if region_id in discovered_reports and challenger_name in known_challengers:
            discovered_reports[region_id].add(challenger_name)

    return {
        region_id: [
            challenger_name for challenger_name in known_challengers if challenger_name in discovered_reports[region_id]
        ]
        for region_id in published_region_ids()
    }


def downloaded_report_path(reports_directory: str, version: str, challenger_name: str, region_id: str) -> str | None:
    for suffix in REPORT_FILE_SUFFIXES:
        path = os.path.join(reports_directory, version, f"{challenger_name}.{region_id}{suffix}")
        if os.path.exists(path):
            return path
    return None


def report_href(reports_directory: str, version: str, challenger_name: str, region_id: str) -> str:
    """Link of a challenger report: its scores file in the bucket, or the report page rendered from its notebook."""
    report_path = downloaded_report_path(reports_directory, version, challenger_name, region_id)
    if report_path and report_path.endswith(SCORES_FILE_SUFFIX):
        return report_url(version, os.path.basename(report_path))
    return f"reports/{version}/{challenger_name}.{region_id}.report.html"


def download_report(version: str, file_name: str, destination_directory: str) -> str | None:
    os.makedirs(destination_directory, exist_ok=True)
    destination_path = os.path.join(destination_directory, file_name)
    url = report_url(version, file_name)

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        with open(destination_path, "wb") as file:
            file.write(response.content)
        return destination_path
    except Exception as error:
        print(f"Failed to download {file_name} ({version}) from {url}: {error}")
    return None
