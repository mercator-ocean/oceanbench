# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

import download_reports  # noqa: E402


def test_main_launches_observation_refresh_before_downloads(monkeypatch, tmp_path: Path) -> None:
    events = []
    manifest_path = tmp_path / "nrt-validation-manifest.json"

    def maybe_launch_daily_observation_refresh() -> None:
        events.append("observation-refresh")

    def discover_official_reports() -> dict[str, list[str]]:
        events.append("discover-reports")
        return {"global": ["glonet"]}

    def download_notebook(challenger_name: str, region_id: str, destination_directory: str) -> str:
        events.append(f"download-{challenger_name}-{region_id}")
        destination = Path(destination_directory) / f"{challenger_name}.{region_id}.report.ipynb"
        destination.write_text("{}", encoding="utf-8")
        return str(destination)

    def download_report_file(file_name: str, destination_directory: str) -> str:
        events.append(f"download-{file_name}")
        assert file_name == download_reports.NRT_MANIFEST_FILE_NAME
        manifest_path.write_text(json.dumps({"evaluations": []}), encoding="utf-8")
        return str(manifest_path)

    monkeypatch.setattr(sys, "argv", ["download_reports.py"])
    monkeypatch.setattr(download_reports, "REPORTS_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(download_reports, "QUARTO_METADATA_FILE_PATH", str(tmp_path / "_metadata.yml"))
    monkeypatch.setattr(
        download_reports, "maybe_launch_daily_observation_refresh", maybe_launch_daily_observation_refresh
    )
    monkeypatch.setattr(download_reports, "discover_official_reports", discover_official_reports)
    monkeypatch.setattr(download_reports, "download_notebook", download_notebook)
    monkeypatch.setattr(download_reports, "download_report_file", download_report_file)

    download_reports.main()

    assert events == [
        "observation-refresh",
        "discover-reports",
        "download-glonet-global",
        f"download-{download_reports.NRT_MANIFEST_FILE_NAME}",
    ]
    assert Path(download_reports.QUARTO_METADATA_FILE_PATH).exists()


def test_download_nrt_reports_skips_pending_entries(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "nrt-validation-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "evaluations": [
                    {
                        "system_label": "GLONET",
                        "status": "Forecast pending",
                        "report_notebook": "glonet.latest.global.report.ipynb",
                    },
                    {
                        "system_label": "XIHE",
                        "status": "Complete",
                        "report_notebook": "xihe.latest.global.report.ipynb",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    downloaded_files = []

    def download_report_file(file_name: str, destination_directory: str) -> str:
        downloaded_files.append(file_name)
        return str(Path(destination_directory) / file_name)

    monkeypatch.setattr(download_reports, "REPORTS_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(download_reports, "download_report_file", download_report_file)

    download_reports._download_nrt_report_notebooks_from_manifest(str(manifest_path))

    assert downloaded_files == ["xihe.latest.global.report.ipynb"]


def test_download_nrt_reports_prefers_manifest_report_url(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "nrt-validation-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "evaluations": [
                    {
                        "system_label": "GLONET",
                        "status": "Complete",
                        "report_notebook": "glonet.latest.global.report.ipynb",
                        "report_url": "https://example.test/custom/glonet.latest.global.report.ipynb",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    downloaded_urls = []

    def download_report_url(
        report_url: str,
        destination_directory: str,
        destination_file_name: str,
    ) -> str:
        downloaded_urls.append((report_url, destination_file_name))
        return str(Path(destination_directory) / destination_file_name)

    def download_report_file(*_) -> str:
        raise AssertionError("download_report_file should not be used when report_url is present")

    monkeypatch.setattr(download_reports, "REPORTS_DIRECTORY", str(tmp_path))
    monkeypatch.setattr(download_reports, "download_report_url", download_report_url)
    monkeypatch.setattr(download_reports, "download_report_file", download_report_file)

    download_reports._download_nrt_report_notebooks_from_manifest(str(manifest_path))

    assert downloaded_urls == [
        (
            "https://example.test/custom/glonet.latest.global.report.ipynb",
            "glonet.latest.global.report.ipynb",
        )
    ]


def test_write_sample_nrt_manifest_uses_placeholder_status(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(download_reports, "REPORTS_DIRECTORY", str(tmp_path))

    manifest_path = download_reports._write_sample_nrt_manifest()

    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    evaluation = manifest["evaluations"][0]

    assert manifest_path == str(tmp_path / "nrt-validation-manifest.json")
    assert evaluation["system_label"] == "GLONET"
    assert evaluation["forecast_init"] == "Unavailable"
    assert evaluation["observation_cutoff"] == "Unavailable"
    assert evaluation["status"] == "Manifest unavailable"
    assert "demo" not in evaluation
    assert "initial_condition_provenance_validated" not in evaluation
