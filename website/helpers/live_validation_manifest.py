# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from html import escape
import json
from pathlib import Path

from helpers.live_validation_report import ForecastValidationMetadata

DEFAULT_MANIFEST_PATH = "reports/nrt-validation-manifest.json"
REPORT_PAGE_NAMES = {
    "octo-glonet-p1d": "glonet-forecast-validation.html",
    "octo-glonet2-p1d": "glonet2-forecast-validation.html",
    "octo-langya-p1d": "langya-forecast-validation.html",
    "octo-wenhai-p1d": "wenhai-forecast-validation.html",
    "octo-xihe-p1d": "xihe-forecast-validation.html",
}


def read_live_validation_manifest(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> dict:
    with open(manifest_path, encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def live_validation_evaluations(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> list[dict]:
    manifest = read_live_validation_manifest(manifest_path)
    return list(manifest.get("evaluations", []))


def _report_page_name(evaluation: dict) -> str:
    return REPORT_PAGE_NAMES[str(evaluation["system_id"])]


def _find_evaluation(evaluations: list[dict], system_id: str) -> dict:
    for evaluation in evaluations:
        if evaluation["system_id"] == system_id:
            return evaluation
    raise ValueError(f"No NRT validation entry found for {system_id!r}.")


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
    system_id: str = "octo-glonet-p1d",
) -> ForecastValidationMetadata:
    evaluation = _find_evaluation(live_validation_evaluations(manifest_path), system_id)
    return ForecastValidationMetadata(
        system_label=evaluation["system_label"],
        forecast_init=evaluation["forecast_init"],
        validated_lead_days=evaluation["validated_lead_days"],
        observation_cutoff=evaluation["observation_cutoff"],
        status=evaluation["status"],
        note=evaluation.get("note"),
        system_id=evaluation["system_id"],
    )


def report_notebook_path(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    system_id: str = "octo-glonet-p1d",
) -> str:
    evaluation = _find_evaluation(live_validation_evaluations(manifest_path), system_id)
    return str(Path("reports") / evaluation["report_notebook"])
