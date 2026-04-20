# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import re
from pathlib import Path

import requests

from helpers.challenger_metadata import KNOWN_CHALLENGERS
from helpers.published_regions import GLOBAL_REGION_NAME
from helpers.published_regions import published_region_ids

S3_BASE_URL = "https://minio.dive.edito.eu/project-oceanbench"
REPORTS_PREFIX = "public/evaluation-reports/1.0.0/"
LOCAL_REPORTS_ENVIRONMENT_VARIABLE = "OCEANBENCH_WEBSITE_USE_LOCAL_REPORTS"
LOCAL_REPORTS_DIRECTORY = Path(__file__).resolve().parents[1] / "reports"

EXPLICIT_REPORT_PATTERN = re.compile(r"^(?P<challenger>.+)\.(?P<region>[a-z0-9_-]+)\.report\.ipynb$")


def _parse_report_key(file_name: str) -> tuple[str, str] | None:
    explicit_match = EXPLICIT_REPORT_PATTERN.match(file_name)
    if explicit_match is None:
        return None
    return explicit_match.group("challenger"), explicit_match.group("region")


def _empty_published_reports() -> dict[str, set[str]]:
    return {region_id: set() for region_id in published_region_ids()}


def _sorted_published_reports(discovered_reports: dict[str, set[str]]) -> dict[str, list[str]]:
    return {region_id: sorted(challenger_names) for region_id, challenger_names in discovered_reports.items()}


def _discover_local_reports() -> dict[str, list[str]]:
    discovered_reports = _empty_published_reports()
    if not LOCAL_REPORTS_DIRECTORY.is_dir():
        return _sorted_published_reports(discovered_reports)

    for report_path in LOCAL_REPORTS_DIRECTORY.glob("*.report.ipynb"):
        parsed = _parse_report_key(report_path.name)
        if parsed is None:
            continue
        challenger_name, region_id = parsed
        if region_id in discovered_reports:
            discovered_reports[region_id].add(challenger_name)
    return _sorted_published_reports(discovered_reports)


def discover_official_reports() -> dict[str, list[str]]:
    if os.environ.get(LOCAL_REPORTS_ENVIRONMENT_VARIABLE) == "1":
        return _discover_local_reports()

    discovered_reports = _empty_published_reports()
    try:
        url = f"{S3_BASE_URL}?list-type=2&prefix={REPORTS_PREFIX}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            keys = re.findall(r"<Key>(.*?)</Key>", response.text)
            for key in keys:
                parsed = _parse_report_key(key.split("/")[-1])
                if parsed is None:
                    continue
                challenger_name, region_id = parsed
                if region_id in discovered_reports:
                    discovered_reports[region_id].add(challenger_name)
    except Exception:
        pass

    if any(challengers for challengers in discovered_reports.values()):
        return _sorted_published_reports(discovered_reports)

    return {
        GLOBAL_REGION_NAME: list(KNOWN_CHALLENGERS),
        **{region_id: [] for region_id in published_region_ids() if region_id != GLOBAL_REGION_NAME},
    }


def _notebook_url(challenger_name: str, region_id: str) -> str:
    return f"{S3_BASE_URL}/{REPORTS_PREFIX}{challenger_name}.{region_id}.report.ipynb"


def download_notebook(challenger_name: str, region_id: str, destination_directory: str) -> str | None:
    os.makedirs(destination_directory, exist_ok=True)
    destination_path = os.path.join(destination_directory, f"{challenger_name}.{region_id}.report.ipynb")
    url = _notebook_url(challenger_name, region_id)

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        with open(destination_path, "wb") as file:
            file.write(response.content)
        return destination_path
    except Exception as error:
        print(f"Failed to download {challenger_name}.{region_id} from {url}: {error}")
    return None
