# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.s3_discovery import download_notebook  # noqa: E402
from helpers.s3_discovery import downloaded_report_file_name  # noqa: E402
from helpers.s3_discovery import discover_downloaded_reports  # noqa: E402
from helpers.s3_discovery import discover_downloaded_reports_by_year  # noqa: E402
from helpers.s3_discovery import discover_official_reports  # noqa: E402
from helpers.s3_discovery import discover_official_reports_by_year  # noqa: E402
from helpers.s3_discovery import REPORTS_PREFIX  # noqa: E402
from helpers.published_regions import published_region_ids  # noqa: E402
from helpers.published_regions import published_region_ids_with_reports  # noqa: E402
from helpers.published_regions import published_region_metadata  # noqa: E402


class MockResponse:
    def __init__(self, status_code: int, text: str = "", content: bytes = b""):
        self.status_code = status_code
        self.text = text
        self.content = content


def test_discover_official_reports_probes_only_official_region_report_names(monkeypatch) -> None:
    existing_report_urls = {
        f"https://minio.dive.edito.eu/project-oceanbench/{REPORTS_PREFIX}glo12.global.report.ipynb",
        f"https://minio.dive.edito.eu/project-oceanbench/{REPORTS_PREFIX}glo12.ibi.report.ipynb",
        f"https://minio.dive.edito.eu/project-oceanbench/{REPORTS_PREFIX}wenhai.global.report.ipynb",
    }
    requested_urls = []

    def fake_head(url: str, timeout: int):
        assert timeout == 10
        requested_urls.append(url)
        return MockResponse(status_code=200 if url in existing_report_urls else 404)

    monkeypatch.setattr("helpers.s3_discovery.requests.head", fake_head)

    reports = discover_official_reports()

    assert list(reports) == published_region_ids()
    assert reports["global"] == ["glo12", "wenhai"]
    assert reports["ibi"] == ["glo12"]
    assert all(".global.report.ipynb" in url or ".ibi.report.ipynb" in url for url in requested_urls)
    assert not any(
        ".report.ipynb" in url and ".global.report.ipynb" not in url and ".ibi.report.ipynb" not in url
        for url in requested_urls
    )


def test_discover_official_reports_by_year_uses_year_subdirectories_for_non_default_years(monkeypatch) -> None:
    existing_report_urls = {
        f"https://minio.dive.edito.eu/project-oceanbench/{REPORTS_PREFIX}2023/glo12.global.report.ipynb",
        f"https://minio.dive.edito.eu/project-oceanbench/{REPORTS_PREFIX}glo12.global.report.ipynb",
        f"https://minio.dive.edito.eu/project-oceanbench/{REPORTS_PREFIX}2025/glo12.global.report.ipynb",
    }
    requested_urls = []

    def fake_head(url: str, timeout: int):
        requested_urls.append(url)
        return MockResponse(status_code=200 if url in existing_report_urls else 404)

    monkeypatch.setattr("helpers.s3_discovery.requests.head", fake_head)

    reports = discover_official_reports_by_year()

    assert reports[2023]["global"] == ["glo12"]
    assert reports[2024]["global"] == ["glo12"]
    assert reports[2025]["global"] == ["glo12"]
    assert (
        f"https://minio.dive.edito.eu/project-oceanbench/{REPORTS_PREFIX}" "2023/glo12.global.report.ipynb"
    ) in requested_urls
    assert (
        f"https://minio.dive.edito.eu/project-oceanbench/{REPORTS_PREFIX}" "glo12.global.report.ipynb"
    ) in requested_urls
    assert (
        f"https://minio.dive.edito.eu/project-oceanbench/{REPORTS_PREFIX}" "2025/glo12.global.report.ipynb"
    ) in requested_urls


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


def test_discover_downloaded_reports_reads_local_report_files(tmp_path) -> None:
    (tmp_path / "2023.glonet.global.report.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "glonet.global.report.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "glonet.ibi.report.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "2025.glonet.ibi.report.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "glonet.custom_box.report.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "glonet.report.ipynb").write_text("{}", encoding="utf-8")
    (tmp_path / "unknown.ibi.report.ipynb").write_text("{}", encoding="utf-8")

    reports = discover_downloaded_reports(str(tmp_path))
    reports_by_year = discover_downloaded_reports_by_year(str(tmp_path))

    assert reports["global"] == ["glonet"]
    assert reports["ibi"] == ["glonet"]
    assert reports_by_year[2023]["global"] == ["glonet"]
    assert reports_by_year[2024]["global"] == ["glonet"]
    assert reports_by_year[2025]["ibi"] == ["glonet"]


def test_downloaded_report_file_name_keeps_default_year_backward_compatible() -> None:
    assert downloaded_report_file_name("glonet", "global", 2024) == "glonet.global.report.ipynb"
    assert downloaded_report_file_name("glonet", "global", 2023) == "2023.glonet.global.report.ipynb"


def test_download_notebook_uses_only_explicit_region_name(monkeypatch, tmp_path) -> None:
    requests_seen = []

    def fake_get(url: str, timeout: int):
        requests_seen.append((url, timeout))
        if url.endswith("/glonet.global.report.ipynb"):
            return MockResponse(status_code=200, content=b"{}")
        return MockResponse(status_code=404)

    monkeypatch.setattr("helpers.s3_discovery.requests.get", fake_get)

    destination = download_notebook("glonet", "global", str(tmp_path))

    assert destination == str(tmp_path / "glonet.global.report.ipynb")
    assert (tmp_path / "glonet.global.report.ipynb").read_bytes() == b"{}"
    assert requests_seen == [
        (
            f"https://minio.dive.edito.eu/project-oceanbench/{REPORTS_PREFIX}glonet.global.report.ipynb",
            30,
        )
    ]


def test_download_notebook_uses_year_specific_report_path(monkeypatch, tmp_path) -> None:
    requests_seen = []

    def fake_get(url: str, timeout: int):
        requests_seen.append((url, timeout))
        if url.endswith("/2025/glonet.global.report.ipynb"):
            return MockResponse(status_code=200, content=b"{}")
        return MockResponse(status_code=404)

    monkeypatch.setattr("helpers.s3_discovery.requests.get", fake_get)

    destination = download_notebook("glonet", "global", str(tmp_path), 2025)

    assert destination == str(tmp_path / "2025.glonet.global.report.ipynb")
    assert (tmp_path / "2025.glonet.global.report.ipynb").read_bytes() == b"{}"
    assert requests_seen == [
        (
            f"https://minio.dive.edito.eu/project-oceanbench/{REPORTS_PREFIX}2025/glonet.global.report.ipynb",
            30,
        )
    ]
