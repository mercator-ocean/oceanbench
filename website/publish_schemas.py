# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import shutil
from pathlib import Path

SCHEMAS_SOURCE_DIRECTORY = Path(__file__).resolve().parent.parent / "oceanbench" / "schemas"
PUBLISHED_SCHEMA_PATTERN = "scores-v*.schema.json"
OUTPUT_DIRECTORY = Path(os.environ["QUARTO_PROJECT_OUTPUT_DIR"]).resolve()
SCHEMAS_OUTPUT_DIRECTORY = OUTPUT_DIRECTORY / "schemas"


def main() -> None:
    SCHEMAS_OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    schema_paths = sorted(SCHEMAS_SOURCE_DIRECTORY.glob(PUBLISHED_SCHEMA_PATTERN))
    for schema_path in schema_paths:
        shutil.copy(schema_path, SCHEMAS_OUTPUT_DIRECTORY / schema_path.name)
    print(f"Published {len(schema_paths)} schemas to {SCHEMAS_OUTPUT_DIRECTORY}.")


if __name__ == "__main__":
    main()
