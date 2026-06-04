# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
import glob
import json
import os

from helpers.challenger_metadata import KNOWN_CHALLENGERS
from helpers.notebook_score_parser import get_all_model_scores_from_notebook
from helpers.s3_discovery import REPORT_CATALOG_FILE_NAME
from helpers.s3_discovery import download_scores
from helpers.s3_discovery import official_report_catalog
from helpers.s3_discovery import report_catalog_published_reports
from helpers.s3_discovery import write_report_catalog

SCRIPT_DIRECTORY = os.path.dirname(__file__)
REPORTS_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "reports")
ASSETS_DIRECTORY = os.path.join(SCRIPT_DIRECTORY, "..", "assets")
QUARTO_METADATA_FILE_PATH = os.path.join(REPORTS_DIRECTORY, "_metadata.yml")
USE_SAMPLES_ENVIRONMENT_VARIABLE = "OCEANBENCH_WEBSITE_USE_SAMPLES"


def _find_sample_notebooks() -> list[str]:
    return glob.glob(os.path.join(ASSETS_DIRECTORY, "*.global.report.ipynb"))


def _clear_report_artifacts() -> None:
    if not os.path.isdir(REPORTS_DIRECTORY):
        return
    for file_name in os.listdir(REPORTS_DIRECTORY):
        if file_name.endswith((".report.ipynb", ".scores.json")) or file_name in (
            REPORT_CATALOG_FILE_NAME,
            "_metadata.yml",
        ):
            os.remove(os.path.join(REPORTS_DIRECTORY, file_name))


def _write_scores(challenger_name: str, region_id: str, scores: dict) -> str:
    destination_path = os.path.join(REPORTS_DIRECTORY, f"{challenger_name}.{region_id}.scores.json")
    with open(destination_path, "w", encoding="utf-8") as file:
        json.dump(scores, file, sort_keys=True)
    return destination_path


def _sample_challenger_name(sample_notebook_path: str) -> str:
    return os.path.basename(sample_notebook_path).removesuffix(".global.report.ipynb").removesuffix("_sample")


def _sample_report_catalog() -> dict:
    catalog = {"regions": {"global": {}, "ibi": {}}}
    for sample_notebook in _find_sample_notebooks():
        challenger_name = _sample_challenger_name(sample_notebook)
        if challenger_name not in KNOWN_CHALLENGERS:
            continue
        scores = {
            metric_key: score.model_dump()
            for metric_key, score in get_all_model_scores_from_notebook(sample_notebook, challenger_name).items()
        }
        if scores:
            scores_path = _write_scores(challenger_name, "global", scores)
            sample_notebook_name = os.path.basename(sample_notebook)
            catalog["regions"]["global"][challenger_name] = {
                "report_url": f"../assets/{sample_notebook_name}",
                "notebook_url": f"../assets/{sample_notebook_name}",
                "scores_url": "",
                "scores_path": os.path.relpath(scores_path, SCRIPT_DIRECTORY),
            }
            print(f"{challenger_name}.global: OK (sample scores)")
    return catalog


def _download_published_scores(catalog: dict) -> None:
    for region_id, challengers in catalog.get("regions", {}).items():
        for challenger_name, report in challengers.items():
            print(f"Downloading {challenger_name}.{region_id} scores...", end=" ")
            result = download_scores(challenger_name, region_id, REPORTS_DIRECTORY, report=report)
            if result:
                print(f"OK -> {result}")
                continue
            print("FAILED")
            raise RuntimeError(f"Failed to download scores for {challenger_name}.{region_id}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download lightweight website report metadata.")
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help=(
            "Use local sample notebooks from assets/ to generate lightweight scores. "
            f"Can also be enabled with {USE_SAMPLES_ENVIRONMENT_VARIABLE}=1."
        ),
    )
    args = parser.parse_args()
    use_samples = args.use_samples or os.environ.get(USE_SAMPLES_ENVIRONMENT_VARIABLE) == "1"

    os.makedirs(REPORTS_DIRECTORY, exist_ok=True)
    _clear_report_artifacts()

    catalog = _sample_report_catalog() if use_samples else official_report_catalog()
    published_reports = report_catalog_published_reports(catalog)
    print(f"Discovered reports: {published_reports}")
    if not any(published_reports.values()):
        raise RuntimeError("No complete prebuilt report packages were discovered.")

    if not use_samples:
        _download_published_scores(catalog)
    catalog_path = write_report_catalog(REPORTS_DIRECTORY, catalog)
    print(f"Created {catalog_path}")

    with open(QUARTO_METADATA_FILE_PATH, "w", encoding="utf-8") as file:
        file.write("execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n")
    print(f"Created {QUARTO_METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
