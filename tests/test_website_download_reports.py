# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

import pytest

WEBSITE_DIRECTORY = Path(__file__).resolve().parents[1] / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

import download_reports  # noqa: E402


def _score_table() -> str:
    return (
        "<table>"
        "<thead><tr><th></th><th>Lead day 1</th></tr></thead>"
        "<tbody><tr><th>Temperature (C) [sea_water_potential_temperature]{surface}</th><td>0.1</td></tr></tbody>"
        "</table>"
    )


def _write_valid_report(report_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["oceanbench.metrics.rmsd_of_variables_compared_to_glorys_reanalysis(challenger_dataset)"],
                "outputs": [{"data": {"text/html": [_score_table()]}}],
            }
        ]
    }
    report_path.write_text(json.dumps(notebook), encoding="utf-8")


def _write_invalid_report(report_path: Path) -> None:
    report_path.write_text("{}", encoding="utf-8")


def _metadata(version: str) -> dict:
    return {
        "version": version,
        "reports": [
            {"challenger": "glonet", "region": "global", "file": "glonet.global.report.ipynb"},
            {"challenger": "xihe", "region": "global", "file": "xihe.global.report.ipynb"},
        ],
    }


def _configure_reports_directory(monkeypatch, reports_directory: Path) -> None:
    monkeypatch.setattr(download_reports, "REPORTS_DIRECTORY", str(reports_directory))
    monkeypatch.setattr(
        download_reports,
        "QUARTO_METADATA_FILE_PATH",
        str(reports_directory / "_metadata.yml"),
    )


def test_download_reports_installs_all_parseable_versions(monkeypatch, tmp_path) -> None:
    reports_directory = tmp_path / "reports"
    _configure_reports_directory(monkeypatch, reports_directory)
    monkeypatch.setattr(
        download_reports,
        "discover_official_report_versions",
        lambda: [_metadata("0.1.1"), _metadata("0.1.0")],
    )
    downloaded_versions = []

    def fake_download_report_notebook(version_metadata, report, destination_directory):
        downloaded_versions.append(version_metadata["version"])
        report_path = Path(destination_directory) / report["file"]
        _write_valid_report(report_path)
        return str(report_path)

    monkeypatch.setattr(download_reports, "download_report_notebook", fake_download_report_notebook)

    download_reports.main()

    assert set(downloaded_versions) == {"0.1.1", "0.1.0"}
    assert (reports_directory / "0.1.1" / "glonet.global.report.ipynb").exists()
    assert (reports_directory / "0.1.0" / "glonet.global.report.ipynb").exists()
    assert (reports_directory / "0.1.1" / "_metadata.json").exists()
    assert (reports_directory / "0.1.1" / "_metadata.yml").exists()
    assert (reports_directory / "_metadata.yml").exists()


def test_download_reports_skips_malformed_versions(monkeypatch, tmp_path) -> None:
    reports_directory = tmp_path / "reports"
    _configure_reports_directory(monkeypatch, reports_directory)
    monkeypatch.setattr(
        download_reports,
        "discover_official_report_versions",
        lambda: [_metadata("0.1.1"), _metadata("0.1.0")],
    )

    def fake_download_report_notebook(version_metadata, report, destination_directory):
        report_path = Path(destination_directory) / report["file"]
        if version_metadata["version"] == "0.1.1":
            _write_invalid_report(report_path)
        else:
            _write_valid_report(report_path)
        return str(report_path)

    monkeypatch.setattr(download_reports, "download_report_notebook", fake_download_report_notebook)

    download_reports.main()

    assert not (reports_directory / "0.1.1").exists()
    assert (reports_directory / "0.1.0" / "glonet.global.report.ipynb").exists()


def test_download_reports_keeps_existing_reports_when_no_version_is_valid(monkeypatch, tmp_path) -> None:
    reports_directory = tmp_path / "reports"
    reports_directory.mkdir()
    existing_report = reports_directory / "existing.global.report.ipynb"
    existing_report.write_text("existing", encoding="utf-8")
    _configure_reports_directory(monkeypatch, reports_directory)
    monkeypatch.setattr(download_reports, "discover_official_report_versions", lambda: [_metadata("0.1.1")])

    def fake_download_report_notebook(version_metadata, report, destination_directory):
        report_path = Path(destination_directory) / report["file"]
        _write_invalid_report(report_path)
        return str(report_path)

    monkeypatch.setattr(download_reports, "download_report_notebook", fake_download_report_notebook)

    with pytest.raises(RuntimeError, match="No complete and parseable evaluation report version"):
        download_reports.main()

    assert existing_report.read_text(encoding="utf-8") == "existing"
