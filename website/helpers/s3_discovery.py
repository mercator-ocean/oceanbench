# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import re

import requests

from helpers.challenger_metadata import KNOWN_CHALLENGERS
from helpers.published_regions import published_region_ids

S3_BASE_URL = "https://minio.dive.edito.eu/project-oceanbench"
REPORTS_PREFIX = "public/evaluation-reports/1.2.0/"
REPORT_FILE_PATTERN = re.compile(r"^(?P<challenger>.+)\.(?P<region>[a-z0-9_-]+)\.report\.ipynb$")


def _notebook_url(challenger_name: str, region_id: str) -> str:
    return f"{S3_BASE_URL}/{REPORTS_PREFIX}{challenger_name}.{region_id}.report.ipynb"


def _report_exists(challenger_name: str, region_id: str) -> bool:
    try:
        response = requests.head(_notebook_url(challenger_name, region_id), timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def discover_official_reports() -> dict[str, list[str]]:
    return {
        region_id: [
            challenger_name for challenger_name in KNOWN_CHALLENGERS if _report_exists(challenger_name, region_id)
        ]
        for region_id in published_region_ids()
    }


def discover_downloaded_reports(reports_directory: str) -> dict[str, list[str]]:
    discovered_reports = {region_id: set() for region_id in published_region_ids()}
    if not os.path.isdir(reports_directory):
        return {region_id: [] for region_id in published_region_ids()}

    for file_name in os.listdir(reports_directory):
        report_match = REPORT_FILE_PATTERN.match(file_name)
        if report_match is None:
            continue
        challenger_name = report_match.group("challenger")
        region_id = report_match.group("region")
        if region_id in discovered_reports and challenger_name in KNOWN_CHALLENGERS:
            discovered_reports[region_id].add(challenger_name)

    return {
        region_id: [
            challenger_name for challenger_name in KNOWN_CHALLENGERS if challenger_name in discovered_reports[region_id]
        ]
        for region_id in published_region_ids()
    }


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
