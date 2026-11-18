# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import shutil
from pathlib import Path

from helpers.challengers_table import challengers_table
from helpers.compatible_versions import compatible_versions
from helpers.metric_definitions import METRIC_TITLES, SECTIONS
from helpers.scores_export import SCORES_API_DIRECTORY, write_scores_exports
from helpers.scores_payload import SCORES_DIRECTORY, scores_file_path, scores_payload
from helpers.version_bundles import version_bundles

SCORES_DATA_FILE = Path("_scores-data.html")
CHALLENGERS_TABLE_FILE = Path("challengers/_challengers-table.md")
STANDARD_NAME_DEFINITIONS_FILE = Path("cf_standard_name_definitions.json")
CHANGELOG_FILE = Path("..") / "CHANGELOG.md"
GET_STARTED_NOTEBOOK_SOURCE = Path("..") / "assets" / "evaluate_challenger_example.ipynb"
GET_STARTED_NOTEBOOK_PAGE = Path("documentation/get-started/index.ipynb")


def _scores_data_block(embedded_scores: str, standard_name_definitions: str) -> str:
    return (
        "```{=html}\n"
        f'<script type="application/json" id="scores-data">{embedded_scores}</script>\n'
        f'<script type="application/json" id="cf-standard-name-definitions">{standard_name_definitions}</script>\n'
        "```\n"
    )


def _write_deferred_scores(deferred_scores: dict[str, str]) -> None:
    os.makedirs(SCORES_DIRECTORY, exist_ok=True)
    for version, serialized_scores in deferred_scores.items():
        Path(scores_file_path(version)).write_text(serialized_scores, encoding="utf-8")


def _copy_get_started_notebook() -> None:
    GET_STARTED_NOTEBOOK_PAGE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(GET_STARTED_NOTEBOOK_SOURCE, GET_STARTED_NOTEBOOK_PAGE)


def main() -> None:
    bundles, default_version = version_bundles()
    versions_with_equal_scores = compatible_versions(CHANGELOG_FILE.read_text(encoding="utf-8"), list(bundles))
    payload = scores_payload(bundles, default_version, METRIC_TITLES, SECTIONS, versions_with_equal_scores)
    _write_deferred_scores(payload.deferred)
    write_scores_exports(bundles, METRIC_TITLES, SECTIONS, Path(SCORES_API_DIRECTORY))
    SCORES_DATA_FILE.write_text(
        _scores_data_block(payload.embedded, STANDARD_NAME_DEFINITIONS_FILE.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    CHALLENGERS_TABLE_FILE.write_text(challengers_table(), encoding="utf-8")
    _copy_get_started_notebook()
    print(f"Generated the scores pages data of {len(bundles)} versions.")


if __name__ == "__main__":
    main()
