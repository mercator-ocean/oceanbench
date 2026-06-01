# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from oceanbench.core.python2jupyter import generate_live_evaluation_notebook_file

OUTPUT_DIRECTORY = Path("dev/live-evaluations")


def _challenger_opening_code() -> str:
    return """from pathlib import Path
import sys

for repository_root in [Path.cwd(), *Path.cwd().parents]:
    if (repository_root / "oceanbench").is_dir():
        sys.path.insert(0, str(repository_root))
        break

import oceanbench

challenger_dataset = oceanbench.datasets.challenger.glonet_latest()
"""


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    challenger_file = OUTPUT_DIRECTORY / "glonet_latest.py"
    output_notebook = OUTPUT_DIRECTORY / "glonet.latest.global.report.ipynb"
    challenger_file.write_text(_challenger_opening_code(), encoding="utf-8")
    generate_live_evaluation_notebook_file(
        str(challenger_file),
        str(output_notebook),
        region="global",
    )
    print(output_notebook)


if __name__ == "__main__":
    main()
