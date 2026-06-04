# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Callable

REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[1]
WEBSITE_DIRECTORY = REPOSITORY_DIRECTORY / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.s3_discovery import S3_BASE_URL  # noqa: E402
from helpers.s3_discovery import REPORT_MANIFEST_FILE_NAME  # noqa: E402
from helpers.s3_discovery import REPORTS_PREFIX  # noqa: E402

DEFAULT_PUBLIC_BASE_URL = f"{S3_BASE_URL}/{REPORTS_PREFIX}"
REPORT_NOTEBOOK_SUFFIX = ".report.ipynb"
REPORT_QUARTO_FILES = ("theme-light.scss", "theme-dark.scss", "styles.css", "favicon-light.png")
REPORT_QUARTO_PROJECT = """project:
  type: website
website:
  title: "OceanBench"
  favicon: favicon-light.png
  open-graph: true
  navbar:
    logo: https://minio.dive.edito.eu/project-oceanbench/public/logo/favicon-light.png
    logo-alt: "OceanBench logo."
  repo-url: https://github.com/mercator-ocean/oceanbench
  repo-actions: [source, issue]
  page-footer: |
    Powered by:
    <a href="https://edito.eu">
      <img
        class="only-light"
        src="https://minio.dive.edito.eu/project-oceanbench/public/logo/EDITO_A1%20Version.png"
        alt="EDITO logo"
        height="65"
      />
      <img
        class="only-dark"
        src="https://minio.dive.edito.eu/project-oceanbench/public/logo/EDITO_A1%20Version%20Negative.png"
        alt="EDITO logo"
        height="65"
      />
    </a>
execute:
  enabled: false
format:
  html:
    theme:
      light: [flatly, theme-light.scss]
      dark: [darkly, theme-dark.scss]
    css: styles.css
    page-layout: full
"""


@dataclass(frozen=True)
class ReportIdentity:
    challenger: str
    region: str

    @property
    def stem(self) -> str:
        return f"{self.challenger}.{self.region}"


@dataclass(frozen=True)
class PackagedReport:
    identity: ReportIdentity
    notebook_path: Path
    html_path: Path
    scores_path: Path
    assets_directory: Path


def _ensure_trailing_slash(value: str) -> str:
    return value if value.endswith("/") else f"{value}/"


def parse_report_identity(notebook_path: Path) -> ReportIdentity:
    file_name = notebook_path.name
    if not file_name.endswith(REPORT_NOTEBOOK_SUFFIX):
        raise ValueError(f"Report notebook must end with {REPORT_NOTEBOOK_SUFFIX}: {file_name}")
    stem = file_name.removesuffix(REPORT_NOTEBOOK_SUFFIX)
    try:
        challenger_name, region_id = stem.rsplit(".", maxsplit=1)
    except ValueError as error:
        raise ValueError(f"Report notebook name must be <challenger>.<region>.report.ipynb: {file_name}") from error
    if not challenger_name or not region_id:
        raise ValueError(f"Report notebook name must be <challenger>.<region>.report.ipynb: {file_name}")
    return ReportIdentity(challenger=challenger_name, region=region_id)


def render_report_html(notebook_path: Path, html_path: Path) -> None:
    subprocess.run(
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
        ],
        cwd=notebook_path.parent,
        check=True,
    )


def _prepare_report_quarto_project(output_directory: Path) -> None:
    for file_name in REPORT_QUARTO_FILES:
        shutil.copy2(WEBSITE_DIRECTORY / file_name, output_directory / file_name)
    (output_directory / "_quarto.yml").write_text(REPORT_QUARTO_PROJECT, encoding="utf-8")


def _source_scores_path(notebook_path: Path, report_identity: ReportIdentity) -> Path:
    return notebook_path.with_name(f"{report_identity.stem}.scores.json")


def _copy_scores_json(
    source_notebook_path: Path,
    scores_path: Path,
    report_identity: ReportIdentity,
) -> None:
    source_scores_path = _source_scores_path(source_notebook_path, report_identity)
    if not source_scores_path.exists():
        raise RuntimeError(
            f"Missing direct OceanBench score artifact for {source_notebook_path}: {source_scores_path.name}"
        )
    if source_scores_path.resolve() != scores_path.resolve():
        shutil.copy2(source_scores_path, scores_path)


def _source_assets_directory(notebook_path: Path, report_identity: ReportIdentity) -> Path:
    return notebook_path.with_name(f"{report_identity.stem}.assets")


def _copy_report_assets(
    source_notebook_path: Path,
    assets_directory: Path,
    report_identity: ReportIdentity,
) -> None:
    assets_directory.mkdir(parents=True, exist_ok=True)
    source_assets_directory = _source_assets_directory(source_notebook_path, report_identity)
    if not source_assets_directory.exists():
        return
    if source_assets_directory.resolve() == assets_directory.resolve():
        return
    shutil.copytree(source_assets_directory, assets_directory, dirs_exist_ok=True)


def _copy_report_notebook(source_path: Path, destination_path: Path) -> None:
    if source_path.resolve() == destination_path.resolve():
        return
    shutil.copy2(source_path, destination_path)


def package_report_notebook(
    notebook_path: Path,
    output_directory: Path,
    render_html: Callable[[Path, Path], None] = render_report_html,
) -> PackagedReport:
    identity = parse_report_identity(notebook_path)
    output_directory.mkdir(parents=True, exist_ok=True)
    _prepare_report_quarto_project(output_directory)
    package_stem = identity.stem
    destination_notebook_path = output_directory / f"{package_stem}.report.ipynb"
    html_path = output_directory / f"{package_stem}.report.html"
    scores_path = output_directory / f"{package_stem}.scores.json"
    assets_directory = output_directory / f"{package_stem}.assets"

    _copy_report_notebook(notebook_path, destination_notebook_path)
    _copy_report_assets(notebook_path, assets_directory, identity)
    render_html(destination_notebook_path, html_path)
    _copy_scores_json(notebook_path, scores_path, identity)

    return PackagedReport(
        identity=identity,
        notebook_path=destination_notebook_path,
        html_path=html_path,
        scores_path=scores_path,
        assets_directory=assets_directory,
    )


def _manifest_entry(packaged_report: PackagedReport, public_base_url: str) -> dict[str, str]:
    base_url = _ensure_trailing_slash(public_base_url)
    stem = packaged_report.identity.stem
    return {
        "challenger": packaged_report.identity.challenger,
        "region": packaged_report.identity.region,
        "report_url": f"{base_url}{stem}.report.html",
        "notebook_url": f"{base_url}{stem}.report.ipynb",
        "scores_url": f"{base_url}{stem}.scores.json",
        "assets_url": f"{base_url}{stem}.assets/",
    }


def write_manifest(output_directory: Path, packaged_reports: list[PackagedReport], public_base_url: str) -> Path:
    manifest_path = output_directory / REPORT_MANIFEST_FILE_NAME
    manifest = {
        "reports": [
            _manifest_entry(packaged_report, public_base_url)
            for packaged_report in sorted(
                packaged_reports,
                key=lambda report: (report.identity.region, report.identity.challenger),
            )
        ]
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def package_report_notebooks(
    notebook_paths: list[Path],
    output_directory: Path,
    public_base_url: str,
    render_html: Callable[[Path, Path], None] = render_report_html,
) -> list[PackagedReport]:
    packaged_reports = [
        package_report_notebook(notebook_path, output_directory, render_html=render_html)
        for notebook_path in notebook_paths
    ]
    write_manifest(output_directory, packaged_reports, public_base_url)
    return packaged_reports


def _s3_endpoint_url() -> str | None:
    if endpoint_url := os.environ.get("BOTO3_ENDPOINT_URL"):
        return endpoint_url
    if endpoint := os.environ.get("AWS_S3_ENDPOINT"):
        return f"https://{endpoint}"
    return None


def upload_package_to_s3(package_directory: Path, bucket: str, prefix: str) -> None:
    try:
        import boto3
    except ImportError as error:
        raise RuntimeError("boto3 is required to upload report packages to S3.") from error

    s3_client = boto3.client("s3", endpoint_url=_s3_endpoint_url())
    resolved_prefix = prefix.strip("/")
    for path in sorted(package_directory.rglob("*")):
        if not path.is_file():
            continue
        relative_key = path.relative_to(package_directory).as_posix()
        key = f"{resolved_prefix}/{relative_key}" if resolved_prefix else relative_key
        s3_client.upload_file(str(path), bucket, key)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Package executed OceanBench report notebooks for the website.",
    )
    parser.add_argument(
        "notebooks",
        nargs="+",
        type=Path,
        help="Executed report notebooks named <challenger>.<region>.report.ipynb.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Directory where report HTML, notebooks, score JSON, assets, and manifest.json are written.",
    )
    parser.add_argument(
        "--public-base-url",
        default=DEFAULT_PUBLIC_BASE_URL,
        help="Public URL prefix where the package directory will be uploaded.",
    )
    parser.add_argument(
        "--s3-bucket",
        default=None,
        help="Optional S3 bucket where the completed report package is uploaded.",
    )
    parser.add_argument(
        "--s3-prefix",
        default=None,
        help="Optional S3 prefix where the completed report package is uploaded.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if (args.s3_bucket is None) != (args.s3_prefix is None):
        raise SystemExit("--s3-bucket and --s3-prefix must be provided together.")
    packaged_reports = package_report_notebooks(
        notebook_paths=args.notebooks,
        output_directory=args.output_directory,
        public_base_url=args.public_base_url,
    )
    if args.s3_bucket and args.s3_prefix:
        upload_package_to_s3(args.output_directory, args.s3_bucket, args.s3_prefix)
        print(f"Uploaded report package to s3://{args.s3_bucket}/{args.s3_prefix.strip('/')}")
    print(f"Packaged {len(packaged_reports)} report(s) in {args.output_directory}")


if __name__ == "__main__":
    main()
