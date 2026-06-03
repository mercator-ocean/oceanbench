# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from html import escape
import json
from pathlib import Path

from helpers.live_validation_report import ForecastValidationMetadata

DEFAULT_MANIFEST_PATH = "reports/nrt-validation-manifest.json"


def read_live_validation_manifest(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> dict:
    with open(manifest_path, encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def live_validation_evaluations(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> list[dict]:
    manifest = read_live_validation_manifest(manifest_path)
    return list(manifest.get("evaluations", []))


def _report_page_name(evaluation: dict) -> str:
    system_label = str(evaluation["system_label"]).lower().replace(" ", "-")
    return f"{system_label}-forecast-validation.html"


def render_live_validation_table(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> str:
    rows = "".join(
        (
            "<tr>"
            f"<td>{escape(evaluation['system_label'])}</td>"
            f"<td>{escape(evaluation['forecast_init'])}</td>"
            f"<td>{escape(evaluation['validated_lead_days'])}</td>"
            f"<td>{escape(evaluation['observation_cutoff'])}</td>"
            f"<td>{escape(evaluation['status'])}</td>"
            + (
                f'<td><a href="{escape(_report_page_name(evaluation))}">Report</a></td>'
                if evaluation.get("status") == "Complete"
                else "<td>Pending</td>"
            )
            + "</tr>"
        )
        for evaluation in live_validation_evaluations(manifest_path)
    )
    return (
        '<div class="live-evaluations-table-wrap">'
        '<table class="live-evaluations-table">'
        "<thead><tr>"
        "<th>System</th><th>Forecast init</th><th>Validated lead days</th>"
        "<th>Observation cutoff</th><th>Status</th><th>Report</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
        "</div>"
    )


def forecast_validation_metadata(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    system_label: str = "GLONET",
) -> ForecastValidationMetadata:
    for evaluation in live_validation_evaluations(manifest_path):
        if evaluation.get("system_label") == system_label:
            return ForecastValidationMetadata(
                system_label=evaluation["system_label"],
                forecast_init=evaluation["forecast_init"],
                validated_lead_days=evaluation["validated_lead_days"],
                observation_cutoff=evaluation["observation_cutoff"],
                status=evaluation["status"],
                note=evaluation.get("note"),
            )
    raise ValueError(f"No NRT validation entry found for {system_label!r}.")


def report_notebook_path(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    system_label: str = "GLONET",
) -> str:
    for evaluation in live_validation_evaluations(manifest_path):
        if evaluation.get("system_label") == system_label:
            return str(Path("reports") / evaluation["report_notebook"])
    raise ValueError(f"No NRT validation entry found for {system_label!r}.")
