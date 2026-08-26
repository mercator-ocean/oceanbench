# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

import helpers.s3_discovery as s3_discovery  # noqa: E402
from helpers.s3_discovery import download_notebook  # noqa: E402
from helpers.s3_discovery import downloaded_report_path  # noqa: E402
from helpers.s3_discovery import discover_downloaded_reports  # noqa: E402
from helpers.s3_discovery import discover_official_reports  # noqa: E402
from helpers.s3_discovery import report_html_file_name  # noqa: E402
from helpers.s3_discovery import report_notebook_file_name  # noqa: E402
from helpers.s3_discovery import REPORTS_ROOT_PREFIX  # noqa: E402
from helpers.s3_discovery import S3_BASE_URL  # noqa: E402
from helpers.published_regions import published_region_ids  # noqa: E402
from helpers.published_regions import published_region_ids_with_reports  # noqa: E402
from helpers.published_regions import published_region_metadata  # noqa: E402

TEST_RELEASE_VERSION = "0.4.0"
TEST_YEAR_VERSION = "2023"


class MockResponse:
    def __init__(self, status_code: int, text: str = "", content: bytes = b""):
        self.status_code = status_code
        self.text = text
        self.content = content


def _set_report_index(monkeypatch, versions: dict[str, list[str]], default_version: str) -> None:
    monkeypatch.setattr(
        s3_discovery,
        "_report_index_cache",
        {
            "default": default_version,
            "versions": {version: {"challengers": list(challengers)} for version, challengers in versions.items()},
        },
    )


def _official_report_url(version: str, challenger_name: str, region_id: str) -> str:
    file_name = report_notebook_file_name(version, challenger_name, region_id)
    return f"{S3_BASE_URL}/{REPORTS_ROOT_PREFIX}/{version}/{file_name}"


def _legacy_official_report_url(version: str, challenger_name: str, region_id: str) -> str:
    return f"{S3_BASE_URL}/{REPORTS_ROOT_PREFIX}/{version}/{challenger_name}.{region_id}.report.ipynb"


def test_report_file_names_include_year_only_for_year_versions() -> None:
    assert report_notebook_file_name("2023", "glonet", "global") == "glonet.2023.global.report.ipynb"
    assert report_html_file_name("2023", "glonet", "global") == "glonet.2023.global.report.html"
    assert report_notebook_file_name("0.4.0", "glonet", "global") == "glonet.global.report.ipynb"
    assert report_html_file_name("0.4.0", "glonet", "global") == "glonet.global.report.html"


def test_discover_official_reports_probes_year_named_report_files(monkeypatch) -> None:
    _set_report_index(monkeypatch, {TEST_YEAR_VERSION: ["glo12", "wenhai", "xihe"]}, TEST_YEAR_VERSION)
    existing_report_urls = {
        _official_report_url(TEST_YEAR_VERSION, "glo12", "global"),
        _official_report_url(TEST_YEAR_VERSION, "glo12", "ibi"),
        _official_report_url(TEST_YEAR_VERSION, "wenhai", "global"),
    }
    requested_urls = []

    def fake_head(url: str, timeout: int):
        assert timeout == 10
        requested_urls.append(url)
        return MockResponse(status_code=200 if url in existing_report_urls else 404)

    monkeypatch.setattr("helpers.s3_discovery.requests.head", fake_head)

    reports = discover_official_reports(TEST_YEAR_VERSION)

    assert list(reports) == published_region_ids()
    assert reports["global"] == ["glo12", "wenhai"]
    assert reports["ibi"] == ["glo12"]
    assert all(f"/{TEST_YEAR_VERSION}/" in url for url in requested_urls)
    assert any("glo12.2023.global.report.ipynb" in url for url in requested_urls)


def test_discover_official_reports_keeps_legacy_report_names_for_release_versions(
    monkeypatch,
) -> None:
    _set_report_index(monkeypatch, {TEST_RELEASE_VERSION: ["glo12", "wenhai"]}, TEST_RELEASE_VERSION)
    existing_report_urls = {
        _official_report_url(TEST_RELEASE_VERSION, "glo12", "global"),
        _official_report_url(TEST_RELEASE_VERSION, "wenhai", "ibi"),
    }

    def fake_head(url: str, timeout: int):
        return MockResponse(status_code=200 if url in existing_report_urls else 404)

    monkeypatch.setattr("helpers.s3_discovery.requests.head", fake_head)

    reports = discover_official_reports(TEST_RELEASE_VERSION)

    assert reports["global"] == ["glo12"]
    assert reports["ibi"] == ["wenhai"]


def test_discover_official_reports_falls_back_to_legacy_names_for_year_versions(
    monkeypatch,
) -> None:
    _set_report_index(monkeypatch, {TEST_YEAR_VERSION: ["glo12"]}, TEST_YEAR_VERSION)
    existing_report_urls = {_legacy_official_report_url(TEST_YEAR_VERSION, "glo12", "global")}
    requested_urls = []

    def fake_head(url: str, timeout: int):
        requested_urls.append(url)
        return MockResponse(status_code=200 if url in existing_report_urls else 404)

    monkeypatch.setattr("helpers.s3_discovery.requests.head", fake_head)

    reports = discover_official_reports(TEST_YEAR_VERSION)

    assert reports["global"] == ["glo12"]
    assert requested_urls[:2] == [
        _official_report_url(TEST_YEAR_VERSION, "glo12", "global"),
        _legacy_official_report_url(TEST_YEAR_VERSION, "glo12", "global"),
    ]


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


def test_discover_downloaded_reports_reads_year_named_local_report_files(tmp_path, monkeypatch) -> None:
    _set_report_index(monkeypatch, {TEST_YEAR_VERSION: ["glonet"]}, TEST_YEAR_VERSION)
    version_directory = tmp_path / TEST_YEAR_VERSION
    version_directory.mkdir()
    (version_directory / "glonet.2023.global.report.ipynb").write_text("{}", encoding="utf-8")
    (version_directory / "glonet.2023.ibi.report.ipynb").write_text("{}", encoding="utf-8")
    (version_directory / "glonet.2024.ibi.report.ipynb").write_text("{}", encoding="utf-8")
    (version_directory / "glonet.custom_box.report.ipynb").write_text("{}", encoding="utf-8")
    (version_directory / "glonet.report.ipynb").write_text("{}", encoding="utf-8")
    (version_directory / "unknown.2023.ibi.report.ipynb").write_text("{}", encoding="utf-8")

    reports = discover_downloaded_reports(str(tmp_path), TEST_YEAR_VERSION)

    assert reports["global"] == ["glonet"]
    assert reports["ibi"] == ["glonet"]


def test_downloaded_report_path_prefers_canonical_year_named_file(tmp_path) -> None:
    version_directory = tmp_path / TEST_YEAR_VERSION
    version_directory.mkdir()
    legacy_path = version_directory / "glonet.global.report.ipynb"
    canonical_path = version_directory / "glonet.2023.global.report.ipynb"
    legacy_path.write_text("{}", encoding="utf-8")

    assert downloaded_report_path(str(tmp_path), TEST_YEAR_VERSION, "glonet", "global") == str(legacy_path)

    canonical_path.write_text("{}", encoding="utf-8")

    assert downloaded_report_path(str(tmp_path), TEST_YEAR_VERSION, "glonet", "global") == str(canonical_path)


def test_download_notebook_uses_year_specific_report_path(monkeypatch, tmp_path) -> None:
    requests_seen = []

    def fake_get(url: str, timeout: int):
        requests_seen.append((url, timeout))
        if url == _official_report_url(TEST_YEAR_VERSION, "glonet", "global"):
            return MockResponse(status_code=200, content=b"{}")
        return MockResponse(status_code=404)

    monkeypatch.setattr("helpers.s3_discovery.requests.get", fake_get)

    destination = download_notebook(TEST_YEAR_VERSION, "glonet", "global", str(tmp_path))

    assert destination == str(tmp_path / "glonet.2023.global.report.ipynb")
    assert (tmp_path / "glonet.2023.global.report.ipynb").read_bytes() == b"{}"
    assert requests_seen == [(_official_report_url(TEST_YEAR_VERSION, "glonet", "global"), 30)]


def test_download_notebook_falls_back_to_legacy_year_report_path(monkeypatch, tmp_path) -> None:
    requests_seen = []

    def fake_get(url: str, timeout: int):
        requests_seen.append((url, timeout))
        if url == _legacy_official_report_url(TEST_YEAR_VERSION, "glonet", "global"):
            return MockResponse(status_code=200, content=b"{}")
        return MockResponse(status_code=404)

    monkeypatch.setattr("helpers.s3_discovery.requests.get", fake_get)

    destination = download_notebook(TEST_YEAR_VERSION, "glonet", "global", str(tmp_path))

    assert destination == str(tmp_path / "glonet.2023.global.report.ipynb")
    assert requests_seen == [
        (_official_report_url(TEST_YEAR_VERSION, "glonet", "global"), 30),
        (_legacy_official_report_url(TEST_YEAR_VERSION, "glonet", "global"), 30),
    ]


def test_download_notebook_keeps_release_report_path(monkeypatch, tmp_path) -> None:
    requests_seen = []

    def fake_get(url: str, timeout: int):
        requests_seen.append((url, timeout))
        if url == _official_report_url(TEST_RELEASE_VERSION, "glonet", "global"):
            return MockResponse(status_code=200, content=b"{}")
        return MockResponse(status_code=404)

    monkeypatch.setattr("helpers.s3_discovery.requests.get", fake_get)

    destination = download_notebook(TEST_RELEASE_VERSION, "glonet", "global", str(tmp_path))

    assert destination == str(tmp_path / "glonet.global.report.ipynb")
    assert requests_seen == [(_official_report_url(TEST_RELEASE_VERSION, "glonet", "global"), 30)]
