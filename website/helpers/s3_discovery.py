# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os

import requests

from helpers.challenger_metadata import KNOWN_CHALLENGERS
from helpers.published_regions import published_region_ids

S3_BASE_URL = "https://minio.dive.edito.eu/project-oceanbench"
REPORTS_PREFIX = "public/evaluation-reports/1.1.0/"


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
