# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

import requests

from helpers.challenger_metadata import KNOWN_CHALLENGERS
from helpers.published_regions import published_region_ids


def _project_version() -> str:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with open(pyproject_path, "rb") as file:
        return tomllib.load(file)["project"]["version"]


DEFAULT_REPORTS_VERSION = _project_version()
S3_BASE_URL = os.environ.get("OCEANBENCH_REPORTS_BASE_URL", "https://minio.dive.edito.eu/project-oceanbench")
REPORTS_VERSION = os.environ.get("OCEANBENCH_REPORTS_VERSION", DEFAULT_REPORTS_VERSION)
REPORTS_PREFIX = os.environ.get("OCEANBENCH_REPORTS_PREFIX", f"public/evaluation-reports/{REPORTS_VERSION}/")
REPORT_CATALOG_FILE_NAME = "_report_catalog.json"
REPORT_MANIFEST_FILE_NAME = "manifest.json"
REPORT_URL_KEYS = ("report_url", "notebook_url", "scores_url")


def _ensure_trailing_slash(value: str) -> str:
    return value if value.endswith("/") else f"{value}/"


REPORTS_PREFIX = _ensure_trailing_slash(REPORTS_PREFIX)


def _artifact_url(challenger_name: str, region_id: str, suffix: str) -> str:
    return f"{S3_BASE_URL}/{REPORTS_PREFIX}{challenger_name}.{region_id}.{suffix}"


def report_html_url(challenger_name: str, region_id: str) -> str:
    return _artifact_url(challenger_name, region_id, "report.html")


def notebook_url(challenger_name: str, region_id: str) -> str:
    return _artifact_url(challenger_name, region_id, "report.ipynb")


def scores_url(challenger_name: str, region_id: str) -> str:
    return _artifact_url(challenger_name, region_id, "scores.json")


def manifest_url() -> str:
    return f"{S3_BASE_URL}/{REPORTS_PREFIX}{REPORT_MANIFEST_FILE_NAME}"


def _load_remote_report_manifest() -> dict:
    try:
        response = requests.get(manifest_url(), timeout=30)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {"reports": []}


def _manifest_published_reports(manifest: dict) -> dict[str, set[str]]:
    catalog = report_catalog_from_manifest(manifest)
    return {region_id: set(region_reports) for region_id, region_reports in catalog["regions"].items()}


def discover_official_reports() -> dict[str, list[str]]:
    discovered_reports = _manifest_published_reports(_load_remote_report_manifest())
    return {
        region_id: [
            challenger_name for challenger_name in KNOWN_CHALLENGERS if challenger_name in discovered_reports[region_id]
        ]
        for region_id in published_region_ids()
    }


def _report_catalog_metadata(report: dict) -> dict | None:
    challenger_name = report.get("challenger")
    region_id = report.get("region")
    if challenger_name not in KNOWN_CHALLENGERS or region_id not in published_region_ids():
        return None
    if not all(report.get(key) for key in REPORT_URL_KEYS):
        return None
    metadata = {key: report[key] for key in REPORT_URL_KEYS}
    if report.get("assets_url"):
        metadata["assets_url"] = report["assets_url"]
    metadata["scores_path"] = f"reports/{challenger_name}.{region_id}.scores.json"
    return metadata


def report_catalog_from_manifest(manifest: dict) -> dict:
    catalog = _empty_report_catalog()
    for report in manifest.get("reports", []):
        metadata = _report_catalog_metadata(report)
        if metadata is None:
            continue
        catalog["regions"][report["region"]][report["challenger"]] = metadata
    return catalog


def official_report_catalog() -> dict:
    return report_catalog_from_manifest(_load_remote_report_manifest())


def report_catalog_published_reports(catalog: dict) -> dict[str, list[str]]:
    regions = catalog.get("regions", {})
    return {
        region_id: [
            challenger_name for challenger_name in KNOWN_CHALLENGERS if challenger_name in regions.get(region_id, {})
        ]
        for region_id in published_region_ids()
    }


def report_catalog_path(reports_directory: str) -> str:
    return os.path.join(reports_directory, REPORT_CATALOG_FILE_NAME)


def _empty_report_catalog() -> dict:
    return {"regions": {region_id: {} for region_id in published_region_ids()}}


def write_report_catalog(reports_directory: str, catalog: dict) -> str:
    path = report_catalog_path(reports_directory)
    os.makedirs(reports_directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(catalog, file, indent=2, sort_keys=True)
    return path


def load_report_catalog(reports_directory: str) -> dict:
    path = report_catalog_path(reports_directory)
    if not os.path.exists(path):
        return _empty_report_catalog()
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def discover_downloaded_reports(reports_directory: str) -> dict[str, list[str]]:
    return report_catalog_published_reports(load_report_catalog(reports_directory))


def report_metadata(reports_directory: str, challenger_name: str, region_id: str) -> dict:
    return load_report_catalog(reports_directory)["regions"][region_id][challenger_name]


def download_scores(
    challenger_name: str,
    region_id: str,
    destination_directory: str,
    report: dict | None = None,
) -> str | None:
    destination_path = os.path.join(destination_directory, f"{challenger_name}.{region_id}.scores.json")
    url = report["scores_url"] if report else scores_url(challenger_name, region_id)

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            return None
        os.makedirs(destination_directory, exist_ok=True)
        with open(destination_path, "wb") as file:
            file.write(response.content)
        return destination_path
    except Exception as error:
        print(f"Failed to download {challenger_name}.{region_id} scores from {url}: {error}")
    return None
