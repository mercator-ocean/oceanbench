# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
from concurrent.futures import ThreadPoolExecutor

import requests

from helpers.published_regions import published_region_ids

# A branch deployment can read its scores from another prefix (or another server, for a local build) without
# changing main, which keeps the production defaults below.
S3_BASE_URL = os.environ.get("OCEANBENCH_REPORTS_BASE_URL", "https://s3.waw3-1.cloudferro.com/oceanbench-bucket")
REPORTS_ROOT_PREFIX = os.environ.get("OCEANBENCH_REPORTS_PREFIX", "public/evaluation-reports").strip("/")
EVALUATION_INDEX_URL = f"{S3_BASE_URL}/{REPORTS_ROOT_PREFIX}/index.json"
SCORES_FILE_SUFFIX = ".scores.json"
DOWNLOADED_SCORES_DIRECTORY = "downloaded-scores"

# The scores file discovery and download steps each issue one HTTP request per challenger and region.
# Running them concurrently keeps the website rebuild well under the gateway timeout as the number
# of published versions grows.
MAXIMUM_PARALLEL_REQUESTS = 16

# index.json stored next to the reports is the single source of truth: editing it
# (or uploading a new report) publishes a version or challenger without a library release.
_evaluation_index_cache = None


def _fetch_evaluation_index() -> dict:
    response = requests.get(EVALUATION_INDEX_URL, timeout=10)
    if response.status_code != 200:
        raise RuntimeError(
            f"Could not read the evaluation index at {EVALUATION_INDEX_URL} (HTTP {response.status_code})."
        )
    return response.json()


def _evaluation_index() -> dict:
    global _evaluation_index_cache
    if _evaluation_index_cache is None:
        _evaluation_index_cache = _fetch_evaluation_index()
    return _evaluation_index_cache


def version_sort_key(version: str) -> list[int]:
    return [int(component) if component.isdigit() else -1 for component in version.split(".")]


def available_versions() -> list[str]:
    versions = list(_evaluation_index().get("versions", {}).keys())
    return sorted(versions, key=version_sort_key, reverse=True)


def default_version() -> str:
    versions = available_versions()
    declared_default = _evaluation_index().get("default")
    if declared_default in versions:
        return declared_default
    return versions[0] if versions else ""


def challengers_for_version(version: str) -> list[str]:
    return _evaluation_index().get("versions", {}).get(version, {}).get("challengers", [])


def published_file_url(version: str, file_name: str) -> str:
    return f"{S3_BASE_URL}/{REPORTS_ROOT_PREFIX}/{version}/{file_name}"


def _published_file_exists(version: str, file_name: str) -> bool:
    try:
        response = requests.head(published_file_url(version, file_name), timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def _scores_file_name(challenger_name: str, region_id: str) -> str:
    return f"{challenger_name}.{region_id}{SCORES_FILE_SUFFIX}"


def _published_scores_file_name(version: str, challenger_name: str, region_id: str) -> str | None:
    file_name = _scores_file_name(challenger_name, region_id)
    return file_name if _published_file_exists(version, file_name) else None


def discover_published_scores_files(version: str) -> dict[str, dict[str, str]]:
    """Published scores file name of each challenger, by region."""
    challenger_names = challengers_for_version(version)
    probes = [
        (challenger_name, region_id) for region_id in published_region_ids() for challenger_name in challenger_names
    ]

    def published_scores_file_name(probe: tuple[str, str]) -> str | None:
        challenger_name, region_id = probe
        return _published_scores_file_name(version, challenger_name, region_id)

    with ThreadPoolExecutor(max_workers=MAXIMUM_PARALLEL_REQUESTS) as pool:
        file_names = {
            probe: file_name
            for probe, file_name in zip(probes, pool.map(published_scores_file_name, probes))
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


def discover_downloaded_scores_files(scores_directory: str, version: str) -> dict[str, list[str]]:
    return {
        region_id: [
            challenger_name
            for challenger_name in challengers_for_version(version)
            if downloaded_scores_file_path(scores_directory, version, challenger_name, region_id)
        ]
        for region_id in published_region_ids()
    }


def downloaded_scores_file_path(
    scores_directory: str, version: str, challenger_name: str, region_id: str
) -> str | None:
    scores_file_path = os.path.join(scores_directory, version, f"{challenger_name}.{region_id}{SCORES_FILE_SUFFIX}")
    return scores_file_path if os.path.isfile(scores_file_path) else None


def download_scores_file(version: str, file_name: str, destination_directory: str) -> str | None:
    destination_path = os.path.join(destination_directory, file_name)
    os.makedirs(destination_directory, exist_ok=True)
    url = published_file_url(version, file_name)

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
