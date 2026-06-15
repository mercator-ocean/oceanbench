# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

import helpers.s3_discovery as s3_discovery  # noqa: E402
from helpers.s3_discovery import download_notebook  # noqa: E402
from helpers.s3_discovery import discover_downloaded_reports  # noqa: E402
from helpers.s3_discovery import discover_official_reports  # noqa: E402
from helpers.s3_discovery import REPORTS_ROOT_PREFIX  # noqa: E402
from helpers.s3_discovery import S3_BASE_URL  # noqa: E402
from helpers.published_regions import published_region_ids  # noqa: E402
from helpers.published_regions import published_region_ids_with_reports  # noqa: E402
from helpers.published_regions import published_region_metadata  # noqa: E402

TEST_VERSION = "0.1.4"


class MockResponse:
    def __init__(self, status_code: int, text: str = "", content: bytes = b""):
        self.status_code = status_code
        self.text = text
        self.content = content


def _set_report_index(monkeypatch, challengers: list[str]) -> None:
    monkeypatch.setattr(
        s3_discovery,
        "_report_index_cache",
        {"default": TEST_VERSION, "versions": {TEST_VERSION: {"challengers": list(challengers)}}},
    )


def _official_report_url(version: str, challenger_name: str, region_id: str) -> str:
    return f"{S3_BASE_URL}/{REPORTS_ROOT_PREFIX}/{version}/{challenger_name}.{region_id}.report.ipynb"


def test_discover_official_reports_probes_only_official_region_report_names(monkeypatch) -> None:
    _set_report_index(monkeypatch, ["glo12", "wenhai", "xihe"])
    existing_report_urls = {
        _official_report_url(TEST_VERSION, "glo12", "global"),
        _official_report_url(TEST_VERSION, "glo12", "ibi"),
        _official_report_url(TEST_VERSION, "wenhai", "global"),
    }
    requested_urls = []

    def fake_head(url: str, timeout: int):
        assert timeout == 10
        requested_urls.append(url)
        return MockResponse(status_code=200 if url in existing_report_urls else 404)

    monkeypatch.setattr("helpers.s3_discovery.requests.head", fake_head)

    reports = discover_official_reports(TEST_VERSION)

    assert list(reports) == published_region_ids()
    assert reports["global"] == ["glo12", "wenhai"]
    assert reports["ibi"] == ["glo12"]
    assert all(f"/{TEST_VERSION}/" in url for url in requested_urls)
    assert all(".global.report.ipynb" in url or ".ibi.report.ipynb" in url for url in requested_urls)


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


def test_discover_downloaded_reports_reads_local_report_files(tmp_path, monkeypatch) -> None:
    _set_report_index(monkeypatch, ["glonet"])
    version_directory = tmp_path / TEST_VERSION
    version_directory.mkdir()
    (version_directory / "glonet.global.report.ipynb").write_text("{}", encoding="utf-8")
    (version_directory / "glonet.ibi.report.ipynb").write_text("{}", encoding="utf-8")
    (version_directory / "glonet.custom_box.report.ipynb").write_text("{}", encoding="utf-8")
    (version_directory / "glonet.report.ipynb").write_text("{}", encoding="utf-8")
    (version_directory / "unknown.ibi.report.ipynb").write_text("{}", encoding="utf-8")

    reports = discover_downloaded_reports(str(tmp_path), TEST_VERSION)

    assert reports["global"] == ["glonet"]
    assert reports["ibi"] == ["glonet"]


def test_download_notebook_uses_only_explicit_region_name(monkeypatch, tmp_path) -> None:
    requests_seen = []

    def fake_get(url: str, timeout: int):
        requests_seen.append((url, timeout))
        if url.endswith("/glonet.global.report.ipynb"):
            return MockResponse(status_code=200, content=b"{}")
        return MockResponse(status_code=404)

    monkeypatch.setattr("helpers.s3_discovery.requests.get", fake_get)

    destination = download_notebook(TEST_VERSION, "glonet", "global", str(tmp_path))

    assert destination == str(tmp_path / "glonet.global.report.ipynb")
    assert (tmp_path / "glonet.global.report.ipynb").read_bytes() == b"{}"
    assert requests_seen == [
        (
            _official_report_url(TEST_VERSION, "glonet", "global"),
            30,
        )
    ]
