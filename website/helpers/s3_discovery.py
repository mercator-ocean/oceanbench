# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import re
from concurrent.futures import ThreadPoolExecutor

import requests

from helpers.published_regions import published_region_ids

S3_BASE_URL = "https://minio.dive.edito.eu/project-oceanbench"
REPORTS_ROOT_PREFIX = "dev/evaluation-reports/baselines-0.2.1"
REPORT_INDEX_URL = f"{S3_BASE_URL}/{REPORTS_ROOT_PREFIX}/index.json"
REPORT_FILE_PATTERN = re.compile(r"^(?P<challenger>.+)\.(?P<region>[a-z0-9_-]+)\.report\.ipynb$")

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


def _notebook_url(version: str, challenger_name: str, region_id: str) -> str:
    return f"{S3_BASE_URL}/{REPORTS_ROOT_PREFIX}/{version}/{challenger_name}.{region_id}.report.ipynb"


def _report_exists(version: str, challenger_name: str, region_id: str) -> bool:
    try:
        response = requests.head(_notebook_url(version, challenger_name, region_id), timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def discover_official_reports(version: str) -> dict[str, list[str]]:
    challenger_names = challengers_for_version(version)
    probes = [
        (challenger_name, region_id) for region_id in published_region_ids() for challenger_name in challenger_names
    ]

    def report_exists(probe: tuple[str, str]) -> bool:
        challenger_name, region_id = probe
        return _report_exists(version, challenger_name, region_id)

    with ThreadPoolExecutor(max_workers=MAXIMUM_PARALLEL_REQUESTS) as pool:
        available_probes = {probe for probe, exists in zip(probes, pool.map(report_exists, probes)) if exists}

    return {
        region_id: [
            challenger_name for challenger_name in challenger_names if (challenger_name, region_id) in available_probes
        ]
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


def download_notebook(version: str, challenger_name: str, region_id: str, destination_directory: str) -> str | None:
    os.makedirs(destination_directory, exist_ok=True)
    destination_path = os.path.join(destination_directory, f"{challenger_name}.{region_id}.report.ipynb")
    url = _notebook_url(version, challenger_name, region_id)

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        with open(destination_path, "wb") as file:
            file.write(response.content)
        return destination_path
    except Exception as error:
        print(f"Failed to download {challenger_name}.{region_id} ({version}) from {url}: {error}")
    return None
