# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import shutil

from helpers.scores_payload import SCORES_DIRECTORY

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.abspath(os.environ["QUARTO_PROJECT_OUTPUT_DIR"])


def main() -> None:
    source_directory = os.path.join(SCRIPT_DIRECTORY, SCORES_DIRECTORY)
    destination_directory = os.path.join(OUTPUT_DIRECTORY, SCORES_DIRECTORY)
    shutil.copytree(source_directory, destination_directory, dirs_exist_ok=True)
    print(f"Published {len(os.listdir(source_directory))} version score files to {destination_directory}.")


if __name__ == "__main__":
    main()
