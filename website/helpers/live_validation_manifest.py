# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from html import escape
import json
from pathlib import Path

from helpers.live_validation_report import (
    ForecastValidationMetadata,
    render_forecast_validation_manifest_score_preview,
    render_forecast_validation_score_preview_script,
)

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


def _shared_evaluation_value(evaluations: list[dict], field_name: str) -> str:
    values = {str(evaluation[field_name]) for evaluation in evaluations}
    return next(iter(values)) if len(values) == 1 else "Mixed"


def _completed_system_count(evaluations: list[dict]) -> int:
    return sum(1 for evaluation in evaluations if evaluation["status"] == "Complete")


def _find_evaluation(evaluations: list[dict], system_id: str) -> dict:
    for evaluation in evaluations:
        if evaluation["system_id"] == system_id:
            return evaluation
    raise ValueError(f"No NRT validation entry found for {system_id!r}.")


def _forecast_horizon(evaluation: dict) -> str:
    lead_day_count = int(evaluation["forecast_lead_days"])
    unit = "day" if lead_day_count == 1 else "days"
    return f"{lead_day_count} {unit}"


def _metadata_from_evaluation(evaluation: dict) -> ForecastValidationMetadata:
    return ForecastValidationMetadata(
        system_label=evaluation["system_label"],
        forecast_init=evaluation["forecast_init"],
        validated_lead_days=evaluation["validated_lead_days"],
        observation_cutoff=evaluation["observation_cutoff"],
        status=evaluation["status"],
        note=evaluation.get("note"),
        system_id=evaluation["system_id"],
    )


def _preview_panel_id(evaluation: dict) -> str:
    return f"live-preview-{evaluation['system_id']}"


def render_live_validation_summary(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> str:
    manifest = read_live_validation_manifest(manifest_path)
    evaluations = list(manifest["evaluations"])
    completed_system_count = _completed_system_count(evaluations)
    cards = [
        ("Forecast init", _shared_evaluation_value(evaluations, "forecast_init")),
        ("Observation cutoff", _shared_evaluation_value(evaluations, "observation_cutoff")),
        ("Completed systems", f"{completed_system_count} / {len(evaluations)}"),
    ]
    if manifest.get("generated_at"):
        cards.append(("Last updated", str(manifest["generated_at"]).replace("T", " ").removesuffix("Z") + " UTC"))
    return (
        '<div class="live-evaluations-summary validation-summary-grid">'
        + "".join(
            '<div class="validation-summary-card">'
            f'<span class="validation-card-label">{escape(label)}</span>'
            f"<strong>{escape(value)}</strong>"
            "</div>"
            for label, value in cards
        )
        + "</div>"
    )


def render_live_validation_table(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> str:
    evaluations = live_validation_evaluations(manifest_path)
    selected_panel_id = next(
        (_preview_panel_id(evaluation) for evaluation in evaluations if evaluation.get("status") == "Complete"),
        None,
    )
    rows = "".join(
        (
            "<tr" + _live_validation_table_row_attributes(evaluation, selected_panel_id) + ">"
            f"<td>{_live_validation_system_cell(evaluation, selected_panel_id)}</td>"
            f"<td>{escape(_forecast_horizon(evaluation))}</td>"
            f"<td>{escape(evaluation['validated_lead_days'])}</td>"
            f"<td>{escape(evaluation['status'])}</td>"
            + (
                f'<td><a href="{escape(_report_page_name(evaluation))}">Report</a></td>'
                if evaluation.get("status") == "Complete"
                else "<td>Pending</td>"
            )
            + "</tr>"
        )
        for evaluation in evaluations
    )
    return (
        '<section class="live-evaluations-table-wrap">'
        '<table class="live-evaluations-table">'
        "<thead><tr>"
        "<th>System</th><th>Forecast horizon</th><th>Evaluated leads</th><th>Status</th><th>Report</th>"
        "</tr></thead>"
        f"<tbody>{rows}</tbody>"
        "</table>"
        "</section>"
    )


def _live_validation_table_row_attributes(evaluation: dict, selected_panel_id: str | None) -> str:
    if evaluation.get("status") != "Complete":
        return ' class="live-evaluation-row"'
    panel_id = _preview_panel_id(evaluation)
    selected_class = " is-selected" if panel_id == selected_panel_id else ""
    return (
        f' class="live-evaluation-row{selected_class}"'
        f' data-live-preview-target="{escape(panel_id)}"'
        f' aria-selected="{str(panel_id == selected_panel_id).lower()}"'
    )


def _live_validation_system_cell(evaluation: dict, selected_panel_id: str | None) -> str:
    if evaluation.get("status") != "Complete":
        return escape(evaluation["system_label"])
    panel_id = _preview_panel_id(evaluation)
    return (
        '<button type="button" class="live-preview-select'
        + (" active" if panel_id == selected_panel_id else "")
        + f'" data-live-preview-target="{escape(panel_id)}"'
        f' aria-controls="{escape(panel_id)}"'
        f' aria-selected="{str(panel_id == selected_panel_id).lower()}">'
        f"{escape(evaluation['system_label'])}"
        "</button>"
    )


def render_live_validation_preview_panel(manifest_path: str | Path = DEFAULT_MANIFEST_PATH) -> str:
    evaluations = [
        evaluation
        for evaluation in live_validation_evaluations(manifest_path)
        if evaluation.get("status") == "Complete"
    ]
    if not evaluations:
        return ""

    panels = "".join(
        (
            f'<section id="{escape(_preview_panel_id(evaluation))}" class="live-preview-panel-content" role="region"'
            + (" hidden" if index != 0 else "")
            + ">"
            + render_forecast_validation_manifest_score_preview(evaluation.get("score_preview"))
            + "</section>"
        )
        for index, evaluation in enumerate(evaluations)
    )
    return (
        '<section class="live-preview-panel">'
        "<h2>Representative lead-time scores</h2>"
        "<p>"
        "Select a challenger to preview selected RMSD metrics; reports contain the full evaluation tables."
        "</p>"
        f"{panels}"
        "</section>" + _live_preview_panel_script() + render_forecast_validation_score_preview_script()
    )


def _live_preview_panel_script() -> str:
    return """
<script>
(() => {
  const rows = Array.from(document.querySelectorAll(".live-evaluation-row[data-live-preview-target]"));
  const buttons = Array.from(document.querySelectorAll(".live-preview-select[data-live-preview-target]"));
  const panels = Array.from(document.querySelectorAll(".live-preview-panel-content"));

  function selectPanel(targetId) {
    rows.forEach((row) => {
      const isSelected = row.dataset.livePreviewTarget === targetId;
      row.classList.toggle("is-selected", isSelected);
      row.setAttribute("aria-selected", String(isSelected));
    });
    buttons.forEach((button) => {
      const isSelected = button.dataset.livePreviewTarget === targetId;
      button.classList.toggle("active", isSelected);
      button.setAttribute("aria-selected", String(isSelected));
    });
    panels.forEach((panel) => {
      panel.hidden = panel.id !== targetId;
    });
  }

  rows.forEach((row) => {
    row.addEventListener("click", (event) => {
      if (event.target.closest("a")) {
        return;
      }
      selectPanel(row.dataset.livePreviewTarget);
    });
  });

  buttons.forEach((button) => {
    button.addEventListener("click", () => {
      selectPanel(button.dataset.livePreviewTarget);
    });
  });
})();
</script>
"""


def forecast_validation_metadata(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    system_id: str = "octo-glonet-p1d",
) -> ForecastValidationMetadata:
    evaluation = _find_evaluation(live_validation_evaluations(manifest_path), system_id)
    return _metadata_from_evaluation(evaluation)


def report_notebook_path(
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    system_id: str = "octo-glonet-p1d",
) -> str:
    evaluation = _find_evaluation(live_validation_evaluations(manifest_path), system_id)
    return str(Path("reports") / evaluation["report_notebook"])
