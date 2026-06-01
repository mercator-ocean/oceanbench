# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from create_synthetic_live_evaluation_buckets import (
    DEFAULT_FIRST_DAY,
    DEFAULT_OUTPUT_DIRECTORY,
    create_synthetic_live_evaluation_buckets,
)
from oceanbench.core.python2jupyter import generate_live_evaluation_notebook_file

OUTPUT_DIRECTORY = Path("dev/live-evaluations")


def _challenger_opening_code(synthetic_output_directory: Path) -> str:
    glonet_template = (synthetic_output_directory / "glonet" / "{date}" / "{date}.zarr").resolve()
    glo12_template = (synthetic_output_directory / "glo12" / "{date}" / "{date}.zarr").resolve()
    return f"""from datetime import datetime
from os import environ
import sys

sys.path.insert(0, "{REPOSITORY_ROOT.resolve()}")

import oceanbench
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable

environ[OceanbenchEnvironmentVariable.OCEANBENCH_LIVE_GLO12_ZARR_TEMPLATE.value] = "file://{glo12_template}"

challenger_dataset = oceanbench.datasets.challenger.glonet_latest(
    first_day_datetime=datetime.fromisoformat("{DEFAULT_FIRST_DAY}"),
    zarr_template="file://{glonet_template}",
)
"""


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    create_synthetic_live_evaluation_buckets()
    challenger_file = OUTPUT_DIRECTORY / "glonet_latest.py"
    output_notebook = OUTPUT_DIRECTORY / "glonet.latest.global.report.ipynb"
    challenger_file.write_text(_challenger_opening_code(DEFAULT_OUTPUT_DIRECTORY), encoding="utf-8")
    generate_live_evaluation_notebook_file(
        str(challenger_file),
        str(output_notebook),
        region="global",
    )
    print(output_notebook)


if __name__ == "__main__":
    main()
