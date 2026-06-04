# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import json
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.s3_discovery import download_scores  # noqa: E402
from helpers.s3_discovery import discover_downloaded_reports  # noqa: E402
from helpers.s3_discovery import discover_official_reports  # noqa: E402
from helpers.s3_discovery import manifest_url  # noqa: E402
from helpers.s3_discovery import notebook_url  # noqa: E402
from helpers.s3_discovery import report_catalog_from_manifest  # noqa: E402
from helpers.s3_discovery import report_catalog_published_reports  # noqa: E402
from helpers.s3_discovery import report_html_url  # noqa: E402
from helpers.s3_discovery import report_metadata  # noqa: E402
from helpers.s3_discovery import scores_url  # noqa: E402
from helpers.s3_discovery import write_report_catalog  # noqa: E402
from helpers.published_regions import published_region_ids  # noqa: E402
from helpers.published_regions import published_region_ids_with_reports  # noqa: E402
from helpers.published_regions import published_region_metadata  # noqa: E402


class MockResponse:
    def __init__(self, status_code: int, text: str = "", content: bytes = b""):
        self.status_code = status_code
        self.text = text
        self.content = content

    def json(self) -> dict:
        return json.loads(self.content.decode("utf-8"))


def test_discover_official_reports_reads_prebuilt_report_manifest(monkeypatch) -> None:
    manifest = {
        "reports": [
            {
                "challenger": "glo12",
                "region": "global",
                "report_url": report_html_url("glo12", "global"),
                "notebook_url": notebook_url("glo12", "global"),
                "scores_url": scores_url("glo12", "global"),
            },
            {
                "challenger": "glo12",
                "region": "ibi",
                "report_url": report_html_url("glo12", "ibi"),
                "notebook_url": notebook_url("glo12", "ibi"),
                "scores_url": scores_url("glo12", "ibi"),
            },
            {
                "challenger": "wenhai",
                "region": "global",
                "report_url": report_html_url("wenhai", "global"),
                "notebook_url": notebook_url("wenhai", "global"),
                "scores_url": scores_url("wenhai", "global"),
            },
            {
                "challenger": "xihe",
                "region": "global",
                "report_url": report_html_url("xihe", "global"),
            },
            {
                "challenger": "unknown",
                "region": "global",
                "report_url": "https://example.test/unknown.global.report.html",
                "notebook_url": "https://example.test/unknown.global.report.ipynb",
                "scores_url": "https://example.test/unknown.global.scores.json",
            },
        ]
    }
    requested_urls = []

    def fake_get(url: str, timeout: int):
        requested_urls.append((url, timeout))
        return MockResponse(status_code=200, content=json.dumps(manifest).encode("utf-8"))

    monkeypatch.setattr("helpers.s3_discovery.requests.get", fake_get)

    reports = discover_official_reports()

    assert list(reports) == published_region_ids()
    assert reports["global"] == ["glo12", "wenhai"]
    assert reports["ibi"] == ["glo12"]
    assert requested_urls == [(manifest_url(), 30)]


def test_report_catalog_preserves_manifest_urls() -> None:
    manifest = {
        "reports": [
            {
                "challenger": "glonet",
                "region": "global",
                "report_url": "https://reports.example/glonet-clean.html",
                "notebook_url": "https://reports.example/glonet.report.ipynb",
                "scores_url": "https://reports.example/glonet.scores.json",
                "assets_url": "https://reports.example/glonet.assets/",
            },
        ]
    }

    catalog = report_catalog_from_manifest(manifest)

    assert report_catalog_published_reports(catalog) == {"global": ["glonet"], "ibi": []}
    assert catalog["regions"]["global"]["glonet"] == {
        "assets_url": "https://reports.example/glonet.assets/",
        "notebook_url": "https://reports.example/glonet.report.ipynb",
        "report_url": "https://reports.example/glonet-clean.html",
        "scores_url": "https://reports.example/glonet.scores.json",
        "scores_path": "reports/glonet.global.scores.json",
    }


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


def test_discover_downloaded_reports_reads_local_report_catalog(tmp_path) -> None:
    write_report_catalog(
        str(tmp_path),
        {
            "regions": {
                "global": {
                    "glonet": {
                        "notebook_url": "https://reports.example/glonet.global.report.ipynb",
                        "report_url": "https://reports.example/glonet.global.report.html",
                        "scores_url": "https://reports.example/glonet.global.scores.json",
                        "scores_path": "reports/glonet.global.scores.json",
                    },
                },
                "ibi": {
                    "glonet": {
                        "notebook_url": "https://reports.example/glonet.ibi.report.ipynb",
                        "report_url": "https://reports.example/glonet.ibi.report.html",
                        "scores_url": "https://reports.example/glonet.ibi.scores.json",
                        "scores_path": "reports/glonet.ibi.scores.json",
                    },
                    "unknown": {
                        "notebook_url": "https://reports.example/unknown.ibi.report.ipynb",
                        "report_url": "https://reports.example/unknown.ibi.report.html",
                        "scores_url": "https://reports.example/unknown.ibi.scores.json",
                        "scores_path": "reports/unknown.ibi.scores.json",
                    },
                },
            },
        },
    )

    reports = discover_downloaded_reports(str(tmp_path))

    assert reports["global"] == ["glonet"]
    assert reports["ibi"] == ["glonet"]
    assert report_metadata(str(tmp_path), "glonet", "global") == {
        "notebook_url": "https://reports.example/glonet.global.report.ipynb",
        "report_url": "https://reports.example/glonet.global.report.html",
        "scores_url": "https://reports.example/glonet.global.scores.json",
        "scores_path": "reports/glonet.global.scores.json",
    }


def test_download_scores_uses_only_explicit_region_name(monkeypatch, tmp_path) -> None:
    requests_seen = []

    def fake_get(url: str, timeout: int):
        requests_seen.append((url, timeout))
        if url.endswith("/glonet.global.scores.json"):
            return MockResponse(status_code=200, content=b"{}")
        return MockResponse(status_code=404)

    monkeypatch.setattr("helpers.s3_discovery.requests.get", fake_get)

    destination = download_scores(
        "glonet",
        "global",
        str(tmp_path),
        report={"scores_url": "https://reports.example/custom/glonet.global.scores.json"},
    )

    assert destination == str(tmp_path / "glonet.global.scores.json")
    assert (tmp_path / "glonet.global.scores.json").read_bytes() == b"{}"
    assert requests_seen == [
        (
            "https://reports.example/custom/glonet.global.scores.json",
            30,
        )
    ]
