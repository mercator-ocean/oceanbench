# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
from base64 import b64decode
from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Callable

REPOSITORY_DIRECTORY = Path(__file__).resolve().parents[1]
WEBSITE_DIRECTORY = REPOSITORY_DIRECTORY / "website"
sys.path.insert(0, str(WEBSITE_DIRECTORY))

from helpers.notebook_score_parser import get_all_model_scores_from_notebook  # noqa: E402
from helpers.s3_discovery import S3_BASE_URL  # noqa: E402
from helpers.s3_discovery import REPORT_MANIFEST_FILE_NAME  # noqa: E402
from helpers.s3_discovery import REPORTS_PREFIX  # noqa: E402

DEFAULT_PUBLIC_BASE_URL = f"{S3_BASE_URL}/{REPORTS_PREFIX}"
REPORT_NOTEBOOK_SUFFIX = ".report.ipynb"
DATA_IMAGE_URI_PATTERN = re.compile(r"data:image/(?P<format>png|jpeg|jpg|webp);base64,(?P<payload>[A-Za-z0-9+/=\s]+)")


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
            sys.executable,
            "-m",
            "jupyter",
            "nbconvert",
            "--to",
            "html",
            str(notebook_path),
            "--output",
            html_path.stem,
            "--output-dir",
            str(html_path.parent),
        ],
        check=True,
    )


def _write_scores_json(notebook_path: Path, scores_path: Path, challenger_name: str) -> None:
    scores = {
        metric_key: score.model_dump()
        for metric_key, score in get_all_model_scores_from_notebook(str(notebook_path), challenger_name).items()
    }
    if not scores:
        raise RuntimeError(f"No OceanBench scores were found in {notebook_path}.")
    scores_path.write_text(json.dumps(scores, indent=2, sort_keys=True), encoding="utf-8")


def _externalize_data_image_assets(html_path: Path, assets_directory: Path, asset_reference_prefix: str) -> None:
    html = html_path.read_text(encoding="utf-8")
    assets_directory.mkdir(parents=True, exist_ok=True)
    written_assets: dict[str, str] = {}

    def replacement(match: re.Match) -> str:
        image_format = match.group("format").lower()
        extension = "jpg" if image_format == "jpeg" else image_format
        payload = re.sub(r"\s+", "", match.group("payload"))
        content = b64decode(payload)
        digest = sha256(content).hexdigest()[:16]
        asset_file_name = f"image-{digest}.{extension}"
        if asset_file_name not in written_assets:
            (assets_directory / asset_file_name).write_bytes(content)
            written_assets[asset_file_name] = asset_file_name
        return f"{asset_reference_prefix}{asset_file_name}"

    html_path.write_text(DATA_IMAGE_URI_PATTERN.sub(replacement, html), encoding="utf-8")


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
    package_stem = identity.stem
    destination_notebook_path = output_directory / f"{package_stem}.report.ipynb"
    html_path = output_directory / f"{package_stem}.report.html"
    scores_path = output_directory / f"{package_stem}.scores.json"
    assets_directory = output_directory / f"{package_stem}.assets"

    _copy_report_notebook(notebook_path, destination_notebook_path)
    render_html(destination_notebook_path, html_path)
    _externalize_data_image_assets(html_path, assets_directory, f"{assets_directory.name}/")
    _write_scores_json(destination_notebook_path, scores_path, identity.challenger)

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
