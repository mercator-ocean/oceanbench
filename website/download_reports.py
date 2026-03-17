# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
import glob
import os
import shutil

from helpers.s3_discovery import discover_challengers, download_notebook

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Download evaluation report notebooks.")
    parser.add_argument(
        "--use-samples",
        action="store_true",
        help="Use local sample notebooks from assets/ instead of downloading from S3.",
    )
    args = parser.parse_args()

    challengers = discover_challengers()
    print(f"Discovered challengers: {challengers}")

    os.makedirs(REPORTS_DIRECTORY, exist_ok=True)

    for challenger_name in challengers:
        destination = os.path.join(REPORTS_DIRECTORY, f"{challenger_name}.report.ipynb")

        if args.use_samples:
            sample = _find_sample_notebook(challenger_name)
            if sample:
                shutil.copy2(sample, destination)
                print(f"{challenger_name}: OK (sample) -> {destination}")
            else:
                print(f"{challenger_name}: no sample found in assets/")
        else:
            print(f"Downloading {challenger_name}...", end=" ")
            result = download_notebook(challenger_name, REPORTS_DIRECTORY)
            if result:
                print(f"OK -> {result}")
            else:
                print("FAILED")

    with open(QUARTO_METADATA_FILE_PATH, "w") as file:
        file.write("execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n")
    print(f"Created {QUARTO_METADATA_FILE_PATH}")


if __name__ == "__main__":
    main()
