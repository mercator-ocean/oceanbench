# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.published_regions import published_region_ids  # noqa: E402
from helpers.published_regions import published_region_ids_with_reports  # noqa: E402
from helpers.published_regions import published_region_metadata  # noqa: E402
from helpers.s3_discovery import REPORT_VERSION_METADATA_FILE_NAME  # noqa: E402
from helpers.s3_discovery import discover_downloaded_reports  # noqa: E402
from helpers.s3_discovery import discover_official_report_versions  # noqa: E402
from helpers.s3_discovery import download_report_notebook  # noqa: E402
from helpers.s3_discovery import load_downloaded_report_versions  # noqa: E402
from helpers.s3_discovery import report_by_challenger_region  # noqa: E402
from helpers.s3_discovery import reports_by_region  # noqa: E402


class MockResponse:
    def __init__(self, status_code: int, text: str = "", content: bytes = b""):
        self.status_code = status_code
        self.text = text
        self.content = content


def _metadata(version: str) -> dict:
    return {
        "version": version,
        "generated_at": "2026-05-12T00:00:00Z",
        "oceanbench_version": version,
        "regions": {
            "global": {"label": "Global", "description": "Global ocean domain.", "bounds": None},
            "ibi": {"label": "IBI", "description": "Iberia-Biscay-Ireland regional domain.", "bounds": {}},
        },
        "challengers": {
            "glonet": {"label": "GLONET"},
            "xihe": {"label": "XiHe"},
        },
        "reports": [
            {"challenger": "glonet", "region": "global", "file": "glonet.global.report.ipynb"},
            {"challenger": "xihe", "region": "global", "file": "xihe.global.report.ipynb"},
            {"challenger": "glonet", "region": "ibi", "file": "glonet.ibi.report.ipynb"},
        ],
    }


def test_discover_official_report_versions_reads_index_and_metadata(monkeypatch) -> None:
    requested_urls = []
    payloads = {
        "https://minio.dive.edito.eu/project-oceanbench/public/evaluation-reports/_index.json": {
            "versions": ["0.1.0", "0.1.1"]
        },
        "https://minio.dive.edito.eu/project-oceanbench/public/evaluation-reports/0.1.1/_metadata.json": _metadata(
            "0.1.1"
        ),
        "https://minio.dive.edito.eu/project-oceanbench/public/evaluation-reports/0.1.0/_metadata.json": _metadata(
            "0.1.0"
        ),
    }

    def fake_get(url: str, timeout: int):
        assert timeout == 30
        requested_urls.append(url)
        return MockResponse(status_code=200, text=json.dumps(payloads[url]))

    monkeypatch.setattr("helpers.s3_discovery.requests.get", fake_get)

    report_versions = discover_official_report_versions()

    assert [metadata["version"] for metadata in report_versions] == ["0.1.1", "0.1.0"]
    assert requested_urls == [
        "https://minio.dive.edito.eu/project-oceanbench/public/evaluation-reports/_index.json",
        "https://minio.dive.edito.eu/project-oceanbench/public/evaluation-reports/0.1.1/_metadata.json",
        "https://minio.dive.edito.eu/project-oceanbench/public/evaluation-reports/0.1.0/_metadata.json",
    ]


def test_discover_official_report_versions_skips_missing_version_metadata(monkeypatch) -> None:
    payloads = {
        "https://minio.dive.edito.eu/project-oceanbench/public/evaluation-reports/_index.json": {
            "versions": ["0.1.0", "0.1.1"]
        },
        "https://minio.dive.edito.eu/project-oceanbench/public/evaluation-reports/0.1.0/_metadata.json": _metadata(
            "0.1.0"
        ),
    }

    def fake_get(url: str, timeout: int):
        assert timeout == 30
        if url not in payloads:
            return MockResponse(status_code=404)
        return MockResponse(status_code=200, text=json.dumps(payloads[url]))

    monkeypatch.setattr("helpers.s3_discovery.requests.get", fake_get)

    report_versions = discover_official_report_versions()

    assert [metadata["version"] for metadata in report_versions] == ["0.1.0"]


def test_published_regions_have_stable_order_and_metadata() -> None:
    assert published_region_ids() == ["global", "ibi"]

    global_metadata = published_region_metadata("global")
    ibi_metadata = published_region_metadata("ibi")

    assert global_metadata["label"] == "Global"
    assert global_metadata["description"]
    assert global_metadata["bounds"] is None
    assert ibi_metadata["label"] == "IBI"
    assert ibi_metadata["description"]
    assert ibi_metadata["bounds"] == {
        "minimum_latitude": 26.17,
        "maximum_latitude": 56.08,
        "minimum_longitude": -19.08,
        "maximum_longitude": 5.08,
    }


def test_published_region_ids_with_reports_filters_empty_regions() -> None:
    assert published_region_ids_with_reports({"global": ["glo12"], "ibi": []}) == ["global"]
    assert published_region_ids_with_reports({"global": [], "ibi": ["glo12"]}) == ["ibi"]


def test_load_downloaded_report_versions_reads_local_metadata(tmp_path) -> None:
    version_directory = tmp_path / "0.1.1"
    version_directory.mkdir()
    (version_directory / REPORT_VERSION_METADATA_FILE_NAME).write_text(json.dumps(_metadata("0.1.1")))

    report_versions = load_downloaded_report_versions(str(tmp_path))

    assert [metadata["version"] for metadata in report_versions] == ["0.1.1"]
    assert reports_by_region(report_versions[0]) == {
        "global": ["glonet", "xihe"],
        "ibi": ["glonet"],
    }
    assert report_by_challenger_region(report_versions[0])[("glonet", "ibi")]["file"] == "glonet.ibi.report.ipynb"


def test_discover_downloaded_reports_returns_latest_local_version(tmp_path) -> None:
    for version in ["0.1.0", "0.1.1"]:
        version_directory = tmp_path / version
        version_directory.mkdir()
        (version_directory / REPORT_VERSION_METADATA_FILE_NAME).write_text(json.dumps(_metadata(version)))

    reports = discover_downloaded_reports(str(tmp_path))

    assert reports == {
        "global": ["glonet", "xihe"],
        "ibi": ["glonet"],
    }


def test_download_report_notebook_uses_metadata_file_path(monkeypatch, tmp_path) -> None:
    requests_seen = []
    version_metadata = _metadata("0.1.1")
    report = version_metadata["reports"][0]

    def fake_get(url: str, timeout: int):
        requests_seen.append((url, timeout))
        if url.endswith("/0.1.1/glonet.global.report.ipynb"):
            return MockResponse(status_code=200, content=b"{}")
        return MockResponse(status_code=404)

    monkeypatch.setattr("helpers.s3_discovery.requests.get", fake_get)

    destination = download_report_notebook(version_metadata, report, str(tmp_path))

    assert destination == str(tmp_path / "glonet.global.report.ipynb")
    assert (tmp_path / "glonet.global.report.ipynb").read_bytes() == b"{}"
    assert requests_seen == [
        (
            "https://minio.dive.edito.eu/project-oceanbench/public/evaluation-reports/0.1.1/glonet.global.report.ipynb",
            30,
        )
    ]
