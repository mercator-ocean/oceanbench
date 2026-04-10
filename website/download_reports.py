# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
import glob
import os
import shutil

from helpers.s3_discovery import discover_official_reports, download_notebook

SCRIPT_DIRECTORY = os.path.dirname(__file__)
REPORTS_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "reports")
ASSETS_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "..", "assets")
QUARTO_METADATA_FILE_PATH = os.path.join(REPORTS_DIRECTORY, "_metadata.yml")


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
