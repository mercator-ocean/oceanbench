# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from helpers.challenger_metadata import CHALLENGERS, challenger_forecasts_run_by
from helpers.s3_discovery import DOWNLOADED_SCORES_DIRECTORY, default_version, discover_downloaded_scores_files

HEADER = (
    "| System | Organisation | Method | Forecast type | Initial conditions | Resolution | Run by |\n"
    "|--------|--------------|--------|---------------|--------------------|------------|--------|\n"
)
UNKNOWN_CHALLENGER_ROW = "| {challenger_name} |  |  |  |  |  |  |"


def _known_challenger_row(challenger_name: str, challenger_metadata: dict) -> str:
    system = f"[{challenger_metadata['label']}]({challenger_metadata['url']})"
    organisation = f"[{challenger_metadata['organisation']}]({challenger_metadata['organisation_url']})"
    return (
        f"| {system} | {organisation} | {challenger_metadata['method']} | "
        f"{challenger_metadata['forecast_type']} | {challenger_metadata['initial_conditions']} | "
        f"{challenger_metadata['resolution']} | {challenger_forecasts_run_by(challenger_name)} |"
    )


def _row(challenger_name: str) -> str:
    challenger_metadata = CHALLENGERS.get(challenger_name)
    if challenger_metadata:
        return _known_challenger_row(challenger_name, challenger_metadata)
    return UNKNOWN_CHALLENGER_ROW.format(challenger_name=challenger_name)


def challengers_table() -> str:
    """The Markdown table of the challengers that have downloaded scores in the default version."""
    published_scores_files = discover_downloaded_scores_files(DOWNLOADED_SCORES_DIRECTORY, default_version())
    challenger_names = sorted(
        {
            challenger_name
            for region_challengers in published_scores_files.values()
            for challenger_name in region_challengers
        }
    )
    rows = "\n".join(map(_row, challenger_names))
    return f'<div class="challengers-table-wrap">\n\n{HEADER}{rows}\n\n</div>\n'
