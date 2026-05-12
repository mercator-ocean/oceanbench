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
DEFAULT_REPORT_YEAR = 2024
SUPPORTED_REPORT_YEARS = (2023, 2024, 2025)
REPORT_FILE_PATTERN = re.compile(r"^(?:(?P<year>\d{4})\.)?(?P<challenger>.+)\.(?P<region>[a-z0-9_-]+)\.report\.ipynb$")


def _notebook_key(challenger_name: str, region_id: str, year: int = DEFAULT_REPORT_YEAR) -> str:
    if year == DEFAULT_REPORT_YEAR:
        return f"{REPORTS_PREFIX}{challenger_name}.{region_id}.report.ipynb"
    return f"{REPORTS_PREFIX}{year}/{challenger_name}.{region_id}.report.ipynb"


def _notebook_url(challenger_name: str, region_id: str, year: int = DEFAULT_REPORT_YEAR) -> str:
    return f"{S3_BASE_URL}/{_notebook_key(challenger_name, region_id, year)}"


def downloaded_report_file_name(challenger_name: str, region_id: str, year: int = DEFAULT_REPORT_YEAR) -> str:
    if year == DEFAULT_REPORT_YEAR:
        return f"{challenger_name}.{region_id}.report.ipynb"
    return f"{year}.{challenger_name}.{region_id}.report.ipynb"


def _report_exists(challenger_name: str, region_id: str, year: int = DEFAULT_REPORT_YEAR) -> bool:
    try:
        response = requests.head(_notebook_url(challenger_name, region_id, year), timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def discover_official_reports(year: int = DEFAULT_REPORT_YEAR) -> dict[str, list[str]]:
    return {
        region_id: [
            challenger_name for challenger_name in KNOWN_CHALLENGERS if _report_exists(challenger_name, region_id, year)
        ]
        for region_id in published_region_ids()
    }


def discover_official_reports_by_year() -> dict[int, dict[str, list[str]]]:
    return {year: discover_official_reports(year) for year in SUPPORTED_REPORT_YEARS}


def _report_file_year(report_match: re.Match) -> int:
    year = report_match.group("year")
    return DEFAULT_REPORT_YEAR if year is None else int(year)


def discover_downloaded_reports_by_year(reports_directory: str) -> dict[int, dict[str, list[str]]]:
    discovered_reports = {
        year: {region_id: set() for region_id in published_region_ids()} for year in SUPPORTED_REPORT_YEARS
    }
    if not os.path.isdir(reports_directory):
        return {year: {region_id: [] for region_id in published_region_ids()} for year in SUPPORTED_REPORT_YEARS}

    for file_name in os.listdir(reports_directory):
        report_match = REPORT_FILE_PATTERN.match(file_name)
        if report_match is None:
            continue
        year = _report_file_year(report_match)
        challenger_name = report_match.group("challenger")
        region_id = report_match.group("region")
        if (
            year in discovered_reports
            and region_id in discovered_reports[year]
            and challenger_name in KNOWN_CHALLENGERS
        ):
            discovered_reports[year][region_id].add(challenger_name)

    return {
        year: {
            region_id: [
                challenger_name
                for challenger_name in KNOWN_CHALLENGERS
                if challenger_name in discovered_reports[year][region_id]
            ]
            for region_id in published_region_ids()
        }
        for year in SUPPORTED_REPORT_YEARS
    }


def discover_downloaded_reports(reports_directory: str, year: int = DEFAULT_REPORT_YEAR) -> dict[str, list[str]]:
    return discover_downloaded_reports_by_year(reports_directory)[year]


def download_notebook(
    challenger_name: str,
    region_id: str,
    destination_directory: str,
    year: int = DEFAULT_REPORT_YEAR,
) -> str | None:
    os.makedirs(destination_directory, exist_ok=True)
    destination_path = os.path.join(
        destination_directory,
        downloaded_report_file_name(challenger_name, region_id, year),
    )
    url = _notebook_url(challenger_name, region_id, year)

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
