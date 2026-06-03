# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from base64 import b64encode
import json
from pathlib import Path
import sys

from helper_scripts.package_evaluation_reports import package_report_notebooks
from helper_scripts.package_evaluation_reports import parse_report_identity
from helper_scripts.package_evaluation_reports import upload_package_to_s3


def _score_table() -> str:
    return (
        "<table>"
        "<thead><tr><th></th><th>Lead day 1</th></tr></thead>"
        "<tbody><tr><th>Temperature (C) [sea_water_potential_temperature]{surface}</th><td>0.1</td></tr></tbody>"
        "</table>"
    )


def _write_report_notebook(notebook_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["evaluation_report.class4_observation.rmsd"],
                "outputs": [{"data": {"text/html": [_score_table()]}}],
            }
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")


def test_package_report_notebooks_writes_report_package_manifest_and_external_assets(tmp_path) -> None:
    notebook_path = tmp_path / "glonet.global.report.ipynb"
    output_directory = tmp_path / "package"
    _write_report_notebook(notebook_path)
    encoded_image = b64encode(b"webp-payload").decode("ascii")

    def fake_render_html(rendered_notebook_path: Path, html_path: Path) -> None:
        assert rendered_notebook_path == output_directory / "glonet.global.report.ipynb"
        html_path.write_text(
            f'<html><body><img src="data:image/webp;base64,{encoded_image}"></body></html>',
            encoding="utf-8",
        )

    packaged_reports = package_report_notebooks(
        notebook_paths=[notebook_path],
        output_directory=output_directory,
        public_base_url="https://example.test/evaluation-reports/",
        render_html=fake_render_html,
    )

    assert parse_report_identity(notebook_path).challenger == "glonet"
    assert parse_report_identity(notebook_path).region == "global"
    assert packaged_reports[0].html_path == output_directory / "glonet.global.report.html"
    assert (output_directory / "glonet.global.report.ipynb").exists()

    scores = json.loads((output_directory / "glonet.global.scores.json").read_text(encoding="utf-8"))
    assert scores["rmsd_variables_observations"]["depths"]["Surface"]["variables"]["temperature"]["data"] == {"1": 0.1}

    html = (output_directory / "glonet.global.report.html").read_text(encoding="utf-8")
    assert "data:image/webp;base64" not in html
    assert "glonet.global.assets/image-" in html
    assert list((output_directory / "glonet.global.assets").glob("*.webp"))

    manifest = json.loads((output_directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {
        "reports": [
            {
                "assets_url": "https://example.test/evaluation-reports/glonet.global.assets/",
                "challenger": "glonet",
                "notebook_url": "https://example.test/evaluation-reports/glonet.global.report.ipynb",
                "region": "global",
                "report_url": "https://example.test/evaluation-reports/glonet.global.report.html",
                "scores_url": "https://example.test/evaluation-reports/glonet.global.scores.json",
            }
        ]
    }


def test_upload_package_to_s3_uses_relative_package_keys(monkeypatch, tmp_path) -> None:
    package_directory = tmp_path / "package"
    asset_directory = package_directory / "glonet.global.assets"
    asset_directory.mkdir(parents=True)
    (package_directory / "manifest.json").write_text("{}", encoding="utf-8")
    (asset_directory / "image.webp").write_bytes(b"image")
    uploads = []

    class FakeS3Client:
        def upload_file(self, source_path: str, bucket: str, key: str) -> None:
            uploads.append((Path(source_path).name, bucket, key))

    class FakeBoto3:
        def client(self, service_name: str, endpoint_url: str | None = None):
            assert service_name == "s3"
            assert endpoint_url == "https://minio.test"
            return FakeS3Client()

    monkeypatch.setitem(sys.modules, "boto3", FakeBoto3())
    monkeypatch.setenv("AWS_S3_ENDPOINT", "minio.test")

    upload_package_to_s3(package_directory, "project-oceanbench", "dev/evaluation-reports/249-webp-demo/")

    assert uploads == [
        ("image.webp", "project-oceanbench", "dev/evaluation-reports/249-webp-demo/glonet.global.assets/image.webp"),
        ("manifest.json", "project-oceanbench", "dev/evaluation-reports/249-webp-demo/manifest.json"),
    ]
