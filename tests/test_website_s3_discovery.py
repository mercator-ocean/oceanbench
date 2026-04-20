# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import sys

import pytest

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.s3_discovery import download_notebook  # noqa: E402
from helpers.s3_discovery import discover_official_reports  # noqa: E402
from helpers.s3_discovery import LOCAL_REPORTS_ENVIRONMENT_VARIABLE  # noqa: E402
from helpers.published_regions import published_region_ids  # noqa: E402
from helpers.published_regions import published_region_ids_with_reports  # noqa: E402
from helpers.published_regions import published_region_metadata  # noqa: E402


class MockResponse:
    def __init__(self, status_code: int, text: str = "", content: bytes = b""):
        self.status_code = status_code
        self.text = text
        self.content = content


def test_discover_official_reports_ignores_non_official_regions_and_legacy_names(monkeypatch) -> None:
    xml_payload = """
    <ListBucketResult>
      <Contents><Key>public/evaluation-reports/1.0.0/glo12.global.report.ipynb</Key></Contents>
      <Contents><Key>public/evaluation-reports/1.0.0/glo12.ibi.report.ipynb</Key></Contents>
      <Contents><Key>public/evaluation-reports/1.0.0/glonet.report.ipynb</Key></Contents>
      <Contents><Key>public/evaluation-reports/1.0.0/glonet.custom_box.report.ipynb</Key></Contents>
      <Contents><Key>public/evaluation-reports/1.0.0/wenhai.global.report.ipynb</Key></Contents>
    </ListBucketResult>
    """

    def fake_get(_url: str, timeout: int):
        assert timeout == 10
        return MockResponse(status_code=200, text=xml_payload)

    monkeypatch.setattr("helpers.s3_discovery.requests.get", fake_get)

    reports = discover_official_reports()

    assert list(reports) == published_region_ids()
    assert reports["global"] == ["glo12", "wenhai"]
    assert reports["ibi"] == ["glo12"]


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


def test_discover_official_reports_can_use_local_reports(monkeypatch, tmp_path) -> None:
    reports_directory = tmp_path / "reports"
    reports_directory.mkdir()
    (reports_directory / "glonet.global.report.ipynb").write_text("{}", encoding="utf-8")
    (reports_directory / "glonet.ibi.report.ipynb").write_text("{}", encoding="utf-8")
    (reports_directory / "glonet.custom_box.report.ipynb").write_text("{}", encoding="utf-8")
    (reports_directory / "glonet.report.ipynb").write_text("{}", encoding="utf-8")

    monkeypatch.setenv(LOCAL_REPORTS_ENVIRONMENT_VARIABLE, "1")
    monkeypatch.setattr("helpers.s3_discovery.LOCAL_REPORTS_DIRECTORY", reports_directory)
    monkeypatch.setattr(
        "helpers.s3_discovery.requests.get",
        lambda _url, timeout: pytest.fail("Local report discovery should not call S3"),
    )

    reports = discover_official_reports()

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

    destination = download_notebook("glonet", "global", str(tmp_path))

    assert destination == str(tmp_path / "glonet.global.report.ipynb")
    assert (tmp_path / "glonet.global.report.ipynb").read_bytes() == b"{}"
    assert requests_seen == [
        (
            "https://minio.dive.edito.eu/project-oceanbench/public/evaluation-reports/1.0.0/glonet.global.report.ipynb",
            30,
        )
    ]
