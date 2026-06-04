# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
import sys

from helper_scripts.package_evaluation_reports import package_report_notebooks
from helper_scripts.package_evaluation_reports import parse_report_identity
from helper_scripts.package_evaluation_reports import render_report_html
from helper_scripts.package_evaluation_reports import upload_package_to_s3


def _widget_iframe_output() -> str:
    return (
        "<iframe srcdoc='<!doctype html><html><body>"
        "<script>"
        "const request = new XMLHttpRequest();"
        "request.open(&quot;GET&quot;, &quot;glonet.global.assets/payload-direct.json&quot;, false);"
        "request.send(null);"
        "const payload = JSON.parse(request.responseText);"
        "document.body.dataset.title = payload.title;"
        "</script>"
        "</body></html>'></iframe>"
    )


def _write_report_notebook(notebook_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["evaluation_report.class4_observation_error_explorer"],
                "outputs": [{"data": {"text/html": [_widget_iframe_output()]}}],
            },
        ],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook), encoding="utf-8")


def _write_widget_assets(assets_directory: Path) -> None:
    assets_directory.mkdir()
    (assets_directory / "image-direct.webp").write_bytes(b"webp-payload")
    (assets_directory / "payload-direct.json").write_text(
        json.dumps(
            {
                "title": "Widget",
                "images": ["glonet.global.assets/image-direct.webp"],
            }
        ),
        encoding="utf-8",
    )


def _direct_scores() -> dict:
    return {
        "rmsd_variables_observations": {
            "name": "glonet",
            "depths": {
                "Surface": {
                    "variables": {
                        "temperature": {
                            "standard_name": "sea_water_potential_temperature",
                            "unit": "C",
                            "data": {"1": 0.1},
                        }
                    }
                }
            },
        }
    }


def _write_direct_scores(scores_path: Path) -> None:
    scores_path.write_text(json.dumps(_direct_scores()), encoding="utf-8")


def _write_minimal_report_notebook(notebook_path: Path) -> None:
    notebook = {
        "cells": [
            {
                "cell_type": "code",
                "source": ["evaluation_report.write_scores_json(score_file_path, challenger_name)"],
                "outputs": [],
            },
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
    _write_widget_assets(tmp_path / "glonet.global.assets")
    _write_direct_scores(tmp_path / "glonet.global.scores.json")

    def fake_render_html(rendered_notebook_path: Path, html_path: Path) -> None:
        assert rendered_notebook_path == output_directory / "glonet.global.report.ipynb"
        transformed_notebook = json.loads(rendered_notebook_path.read_text(encoding="utf-8"))
        transformed_widget = transformed_notebook["cells"][0]["outputs"][0]["data"]["text/html"]
        transformed_widget = "".join(transformed_widget) if isinstance(transformed_widget, list) else transformed_widget
        assert "data:image/webp;base64" not in transformed_widget
        assert "payload-direct.json" in transformed_widget
        assert "XMLHttpRequest" in transformed_widget
        html_path.write_text(f"<html><body>{transformed_widget}</body></html>", encoding="utf-8")

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
    assert (output_directory / "styles.css").exists()
    assert (output_directory / "theme-light.scss").exists()
    assert (output_directory / "theme-dark.scss").exists()
    report_quarto_project = (output_directory / "_quarto.yml").read_text(encoding="utf-8")
    assert "pre-render" not in report_quarto_project
    assert "css: styles.css" in report_quarto_project
    assert "page-layout: full" in report_quarto_project

    scores = json.loads((output_directory / "glonet.global.scores.json").read_text(encoding="utf-8"))
    assert scores["rmsd_variables_observations"]["depths"]["Surface"]["variables"]["temperature"]["data"] == {"1": 0.1}

    html = (output_directory / "glonet.global.report.html").read_text(encoding="utf-8")
    assert "data:image/webp;base64" not in html
    assert "glonet.global.assets/payload-direct.json" in html
    assert (output_directory / "glonet.global.assets" / "image-direct.webp").read_bytes() == b"webp-payload"
    payload = json.loads(
        (output_directory / "glonet.global.assets" / "payload-direct.json").read_text(encoding="utf-8")
    )
    assert payload["images"] == ["glonet.global.assets/image-direct.webp"]

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


def test_package_report_notebooks_requires_direct_scores_artifact(tmp_path) -> None:
    notebook_path = tmp_path / "glonet.global.report.ipynb"
    output_directory = tmp_path / "package"
    _write_minimal_report_notebook(notebook_path)

    def fake_render_html(_rendered_notebook_path: Path, html_path: Path) -> None:
        html_path.write_text("<html></html>", encoding="utf-8")

    try:
        package_report_notebooks(
            notebook_paths=[notebook_path],
            output_directory=output_directory,
            public_base_url="https://example.test/evaluation-reports/",
            render_html=fake_render_html,
        )
    except RuntimeError as error:
        assert "Missing direct OceanBench score artifact" in str(error)
    else:
        raise AssertionError("Expected missing score artifact to fail packaging.")


def test_package_report_notebooks_copies_direct_scores_artifact(tmp_path) -> None:
    notebook_path = tmp_path / "glonet.global.report.ipynb"
    scores_path = tmp_path / "glonet.global.scores.json"
    output_directory = tmp_path / "package"
    _write_minimal_report_notebook(notebook_path)
    _write_direct_scores(scores_path)

    def fake_render_html(_rendered_notebook_path: Path, html_path: Path) -> None:
        html_path.write_text("<html></html>", encoding="utf-8")

    package_report_notebooks(
        notebook_paths=[notebook_path],
        output_directory=output_directory,
        public_base_url="https://example.test/evaluation-reports/",
        render_html=fake_render_html,
    )

    assert json.loads((output_directory / "glonet.global.scores.json").read_text(encoding="utf-8")) == json.loads(
        scores_path.read_text(encoding="utf-8")
    )


def test_render_report_html_uses_quarto_without_execution(monkeypatch, tmp_path) -> None:
    notebook_path = tmp_path / "glonet.global.report.ipynb"
    html_path = tmp_path / "glonet.global.report.html"
    commands = []

    def fake_run(command: list[str], cwd: Path, check: bool) -> None:
        commands.append(command)
        assert cwd == notebook_path.parent
        assert check is True

    monkeypatch.setattr("helper_scripts.package_evaluation_reports.subprocess.run", fake_run)

    render_report_html(notebook_path, html_path)

    assert commands == [
        [
            "quarto",
            "render",
            notebook_path.name,
            "--to",
            "html",
            "--output",
            html_path.name,
            "--output-dir",
            ".",
            "--no-execute",
        ]
    ]


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

    upload_package_to_s3(package_directory, "project-oceanbench", "public/evaluation-reports/0.1.4/")

    assert uploads == [
        ("image.webp", "project-oceanbench", "public/evaluation-reports/0.1.4/glonet.global.assets/image.webp"),
        ("manifest.json", "project-oceanbench", "public/evaluation-reports/0.1.4/manifest.json"),
    ]
