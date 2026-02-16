# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import argparse
import glob
import os
import shutil

from helpers.s3_discovery import discover_challengers, download_notebook

SCRIPT_DIR = os.path.dirname(__file__)
REPORTS_DIR = os.path.join(SCRIPT_DIR, "reports")
ASSETS_DIR = os.path.join(SCRIPT_DIR, "..", "assets")
METADATA_YML = os.path.join(REPORTS_DIR, "_metadata.yml")


def _find_sample_notebook(name: str) -> str | None:
    pattern = os.path.join(ASSETS_DIR, f"{name}*.report.ipynb")
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

    os.makedirs(REPORTS_DIR, exist_ok=True)

    for name in challengers:
        destination = os.path.join(REPORTS_DIR, f"{name}.report.ipynb")

        if args.use_samples:
            sample = _find_sample_notebook(name)
            if sample:
                shutil.copy2(sample, destination)
                print(f"{name}: OK (sample) -> {destination}")
            else:
                print(f"{name}: no sample found in assets/")
        else:
            print(f"Downloading {name}...", end=" ")
            result = download_notebook(name, REPORTS_DIR)
            if result:
                print(f"OK -> {result}")
            else:
                print("FAILED")

    with open(METADATA_YML, "w") as file:
        file.write("execute:\n  enabled: false\nformat:\n  html:\n    page-layout: full\n")
    print(f"Created {METADATA_YML}")


if __name__ == "__main__":
    main()
