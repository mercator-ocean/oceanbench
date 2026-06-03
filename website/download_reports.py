# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
import glob
import json
import os
import shutil

from helpers.s3_discovery import (
    discover_official_reports,
    download_notebook,
    download_report_file,
    download_report_url,
)

SCRIPT_DIRECTORY = os.path.dirname(__file__)
REPORTS_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "reports")
ASSETS_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "..", "assets")
QUARTO_METADATA_FILE_PATH = os.path.join(REPORTS_DIRECTORY, "_metadata.yml")
NRT_MANIFEST_FILE_NAME = "nrt-validation-manifest.json"


def _find_sample_notebook(challenger_name: str) -> str | None:
    pattern = os.path.join(ASSETS_DIRECTORY, f"{challenger_name}*.report.ipynb")
    matches = glob.glob(pattern)
    if matches:
        return matches[0]
    return None


def _clear_report_notebooks() -> None:
    if not os.path.isdir(REPORTS_DIRECTORY):
        return
    for file_name in os.listdir(REPORTS_DIRECTORY):
        if file_name.endswith(".report.ipynb"):
            os.remove(os.path.join(REPORTS_DIRECTORY, file_name))


def _download_nrt_report_notebooks_from_manifest(manifest_path: str) -> None:
    with open(manifest_path, encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    for evaluation in manifest.get("evaluations", []):
        if evaluation.get("status") != "Complete":
            print(f"Skipping pending NRT notebook for {evaluation.get('system_label', 'unknown system')}.")
            continue
        report_notebook = evaluation["report_notebook"]
        print(f"Downloading {report_notebook}...", end=" ")
        report_url = evaluation.get("report_url")
        result = (
            download_report_url(report_url, REPORTS_DIRECTORY, report_notebook)
            if report_url
            else download_report_file(report_notebook, REPORTS_DIRECTORY)
        )
        if result:
            print(f"OK -> {result}")
            continue
        print("FAILED")
        raise RuntimeError(f"Failed to download NRT notebook {report_notebook}.")


def _write_sample_nrt_manifest() -> str:
    manifest_path = os.path.join(REPORTS_DIRECTORY, NRT_MANIFEST_FILE_NAME)
    manifest = {
        "schema_version": 1,
        "evaluations": [
            {
                "system_id": "octo-glonet-p1d",
                "system_label": "GLONET",
                "region": "global",
                "forecast_init": "Unavailable",
                "forecast_lead_days": 10,
                "validated_lead_days": "1-10 days",
                "observation_cutoff": "Unavailable",
                "observation_zarr_template": "Unavailable",
                "forecast_url": "Unavailable",
                "report_notebook": "glonet.latest.global.report.ipynb",
                "report_url": "",
                "status": "Manifest unavailable",
                "demo": True,
                "initial_condition_provenance_validated": False,
                "oceanbench_version": "sample",
                "octo_process_package_name": None,
                "octo_process_package_version": None,
                "note": "Sample website build only: the NRT validation manifest was not downloaded.",
            }
        ],
    }
    with open(manifest_path, "w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download evaluation report notebooks.")
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Use local sample notebooks from assets/ instead of downloading from S3.",
    )
    args = parser.parse_args()

    os.makedirs(REPORTS_DIRECTORY, exist_ok=True)

    published_reports = discover_official_reports()
    print(f"Discovered reports: {published_reports}")
    if not any(published_reports.values()) and not args.use_samples:
        raise RuntimeError("No evaluation reports were discovered in the official bucket.")

    _clear_report_notebooks()

    for region_id, challengers in published_reports.items():
        for challenger_name in challengers:
            destination = os.path.join(REPORTS_DIRECTORY, f"{challenger_name}.{region_id}.report.ipynb")
            print(f"Downloading {challenger_name}.{region_id}...", end=" ")
            result = download_notebook(challenger_name, region_id, REPORTS_DIRECTORY)
            if result:
                print(f"OK -> {result}")
                continue
            sample = _find_sample_notebook(challenger_name)
            if sample:
                shutil.copy2(sample, destination)
                print(f"OK (sample fallback) -> {destination}")
                continue
            if os.path.exists(destination):
                os.remove(destination)
            print("FAILED")
            raise RuntimeError(f"Failed to download notebook for {challenger_name}.{region_id}.")

    print(f"Downloading {NRT_MANIFEST_FILE_NAME}...", end=" ")
    manifest_path = download_report_file(NRT_MANIFEST_FILE_NAME, REPORTS_DIRECTORY)
    if manifest_path:
        print(f"OK -> {manifest_path}")
        _download_nrt_report_notebooks_from_manifest(manifest_path)
    elif args.use_samples:
        print("FAILED")
        sample_manifest_path = _write_sample_nrt_manifest()
        print(f"Using sample NRT manifest -> {sample_manifest_path}")
    else:
        print("FAILED")
        raise RuntimeError(f"Failed to download {NRT_MANIFEST_FILE_NAME}.")

    if args.use_samples:
        sample_challengers = [
            sample.removesuffix(".report.ipynb")
            for sample in os.listdir(ASSETS_DIRECTORY)
            if sample.endswith(".report.ipynb")
        ]
        for challenger_name in sample_challengers:
            destination = os.path.join(REPORTS_DIRECTORY, f"{challenger_name}.global.report.ipynb")
            if os.path.exists(destination):
                continue
            sample = _find_sample_notebook(challenger_name)
            if sample:
                shutil.copy2(sample, destination)
                print(f"{challenger_name}.global: OK (sample) -> {destination}")

    with open(QUARTO_METADATA_FILE_PATH, "w") as file:
        file.write("execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n")
    print(f"Created {QUARTO_METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
