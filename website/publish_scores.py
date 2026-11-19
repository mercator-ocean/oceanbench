# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import shutil

from helpers.scores_export import SCORES_API_DIRECTORY
from helpers.scores_payload import SCORES_DIRECTORY

SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIRECTORY = os.path.abspath(os.environ["QUARTO_PROJECT_OUTPUT_DIR"])


def _publish_directory(directory: str) -> None:
    source_directory = os.path.join(SCRIPT_DIRECTORY, directory)
    destination_directory = os.path.join(OUTPUT_DIRECTORY, directory)
    shutil.copytree(source_directory, destination_directory, dirs_exist_ok=True)
    print(f"Published {len(os.listdir(source_directory))} entries of {directory} to {destination_directory}.")


def main() -> None:
    _publish_directory(SCORES_DIRECTORY)
    _publish_directory(SCORES_API_DIRECTORY)


if __name__ == "__main__":
    main()
