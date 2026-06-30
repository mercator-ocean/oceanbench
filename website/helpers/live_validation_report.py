# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
from html import escape
import json
import math
from pathlib import Path
import re

from helpers.notebook_score_parser import get_all_model_scores_from_notebook
from helpers.type import ModelScore, ModelVariable

OBSERVATION_SCORE_KEY = "rmsd_variables_observations"
DRIFTER_SCORE_KEY = "drifter_trajectory_observations"
DRIFTER_EXPLORER_SOURCE = "evaluation_report.class4_drifter_trajectory_explorer"
CLASS4_EXPLORER_SOURCE = "evaluation_report.class4_observation_error_explorer"
DEPTH_ORDER = ["Surface", "0-5m", "5-100m", "100-300m", "300-600m", "15m"]
VARIABLE_ORDER = [
    "temperature",
    "salinity",
    "sea level anomaly",
    "zonal current",
    "meridional current",
]
SYSTEM_SCORE_NAMES = {
    "octo-glonet-p1d": "glonet",
    "octo-glonet-hr-p1d": "glonet-hr",
    "octo-glonet2-p1d": "glonet2",
    "octo-glonet2-ibi-p1d": "glonet2-ibi",
    "octo-langya-p1d": "langya",
    "octo-wenhai-p1d": "wenhai",
    "octo-xihe-p1d": "xihe",
}


@dataclass(frozen=True)
class ForecastValidationMetadata:
    system_id: str
    system_label: str
    forecast_init: str
    validated_lead_days: str
    observation_cutoff: str
    status: str
    note: str | None = None
    score_preview: dict | None = None


def _read_notebook(notebook_path: str | Path) -> dict:
    with open(notebook_path, encoding="utf-8") as notebook_file:
        return json.load(notebook_file)


def _cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _cell_html_output(cell: dict) -> str | None:
    for output in cell.get("outputs", []):
        data = output.get("data", {})
        if "text/html" not in data:
            continue
        html_output = data["text/html"]
        return "".join(html_output) if isinstance(html_output, list) else html_output
    return None


def _explorer_html(
    notebook_path: str | Path, source: str, unavailable_message: str
) -> str:
    notebook = _read_notebook(notebook_path)
    for cell in notebook.get("cells", []):
        if source in _cell_source(cell):
            html_output = _cell_html_output(cell)
            if html_output:
                return html_output
    return f'<p class="validation-empty">{escape(unavailable_message)}</p>'


def _drifter_explorer_html(notebook_path: str | Path) -> str:
    return _explorer_html(
        notebook_path,
        DRIFTER_EXPLORER_SOURCE,
        "Class IV drifter trajectory animation is not available in this report.",
    )


def _class4_explorer_html(notebook_path: str | Path) -> str:
    return _explorer_html(
        notebook_path,
        CLASS4_EXPLORER_SOURCE,
        "Class IV observation error maps are not available in this report.",
    )


def _observation_score(notebook_path: str | Path, score_name: str) -> ModelScore | None:
    scores = get_all_model_scores_from_notebook(str(notebook_path), score_name)
    return scores.get(OBSERVATION_SCORE_KEY)


def _drifter_score(notebook_path: str | Path, score_name: str) -> ModelScore | None:
    return get_all_model_scores_from_notebook(str(notebook_path), score_name).get(
        DRIFTER_SCORE_KEY
    )


def _notebook_score_name(metadata: ForecastValidationMetadata) -> str:
    score_name = SYSTEM_SCORE_NAMES.get(metadata.system_id)
    if score_name is None:
        raise ValueError(
            f"No notebook score name is configured for system {metadata.system_id!r}."
        )
    return score_name


def _lead_days(score: ModelScore) -> list[str]:
    for depth in score.depths.values():
        for variable in depth.variables.values():
            return sorted(variable.data, key=lambda lead_day: int(lead_day))
    return []


def _drifter_lead_label(lead_day: str) -> str:
    suffix = " (init)" if lead_day == "1" else ""
    return f"Lead {lead_day}{suffix}"


def _depth_sort_key(depth: str) -> int:
    return DEPTH_ORDER.index(depth) if depth in DEPTH_ORDER else len(DEPTH_ORDER)


def _variable_sort_key(variable: str) -> int:
    return (
        VARIABLE_ORDER.index(variable)
        if variable in VARIABLE_ORDER
        else len(VARIABLE_ORDER)
    )


def _score_rows(score: ModelScore) -> list[tuple[str, str, ModelVariable]]:
    rows = [
        (depth, variable_name, variable)
        for depth, depth_score in score.depths.items()
        for variable_name, variable in depth_score.variables.items()
    ]
    return sorted(
        rows, key=lambda row: (_variable_sort_key(row[1]), _depth_sort_key(row[0]))
    )


def _format_depth(depth: str) -> str:
    if depth == "Surface":
        return "surface"
    return re.sub(r"(?<=\d)m$", " m", depth)


def _format_variable(variable: str) -> str:
    return variable.capitalize()


def _format_number(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "NA"
    return f"{value:.3f}"


def _score_label(depth: str, variable: str) -> str:
    return f"{_format_variable(variable)}, {_format_depth(depth)}"


def _summary_cards(metadata: ForecastValidationMetadata) -> str:
    cards = [
        ("System", metadata.system_label),
        ("Forecast init", metadata.forecast_init),
        ("Evaluated lead days", metadata.validated_lead_days),
        ("Observation cutoff", metadata.observation_cutoff),
        ("Status", metadata.status),
    ]
    return (
        '<div class="validation-summary-grid">'
        + "".join(
            '<div class="validation-summary-card">'
            f'<span class="validation-card-label">{escape(label)}</span>'
            f"<strong>{escape(value)}</strong>"
            "</div>"
            for label, value in cards
        )
        + "</div>"
    )


def _score_table(score: ModelScore) -> str:
    lead_days = _lead_days(score)
    header = (
        "<thead><tr><th>Variable</th><th>Unit</th>"
        + "".join(f"<th>Lead {escape(lead_day)}</th>" for lead_day in lead_days)
        + "</tr></thead>"
    )
    body = "".join(
        "<tr>"
        f"<th>{escape(_score_label(depth, variable_name))}</th>"
        f"<td>{escape(variable.unit)}</td>"
        + "".join(
            f"<td>{_format_number(variable.data.get(lead_day))}</td>"
            for lead_day in lead_days
        )
        + "</tr>"
        for depth, variable_name, variable in _score_rows(score)
    )
    return (
        '<div class="validation-score-table-wrap">'
        f'<table class="validation-score-table">{header}<tbody>{body}</tbody></table>'
        "</div>"
    )


def _drifter_score_table(score: ModelScore | None) -> str:
    if score is None:
        return '<p class="validation-empty">Class IV drifter trajectory scores are not available in this report.</p>'
    lead_days = _lead_days(score)
    variables = score.depths.get("flat")
    if variables is None:
        return '<p class="validation-empty">Class IV drifter trajectory scores are not available in this report.</p>'
    header = (
        "<thead><tr><th>Metric</th><th>Unit</th>"
        + "".join(
            f"<th>{escape(_drifter_lead_label(lead_day))}</th>"
            for lead_day in lead_days
        )
        + "</tr></thead>"
    )
    body = "".join(
        "<tr>"
        f"<th>{escape(variable_name)}</th>"
        f"<td>{escape(variable.unit)}</td>"
        + "".join(
            f"<td>{_format_number(variable.data.get(lead_day))}</td>"
            for lead_day in lead_days
        )
        + "</tr>"
        for variable_name, variable in variables.variables.items()
    )
    return (
        '<div class="validation-score-table-wrap">'
        f'<table class="validation-score-table">{header}<tbody>{body}</tbody></table>'
        "</div>"
    )


def _sparkline(lead_days: list[str], values: list[float | None], unit: str) -> str:
    finite_values = [value for value in values if value is not None]
    if len(finite_values) < 2:
        return ""
    width = 210
    height = 64
    padding = 8
    minimum = min(finite_values)
    maximum = max(finite_values)
    value_range = maximum - minimum or 1
    step = (width - 2 * padding) / max(1, len(values) - 1)

    points = []
    for index, value in enumerate(values):
        if value is None:
            continue
        x = padding + index * step
        y = (
            height
            - padding
            - ((value - minimum) / value_range) * (height - 2 * padding)
        )
        points.append((index, value, x, y))

    polyline_points = " ".join(f"{x:.1f},{y:.1f}" for _, _, x, y in points)
    circles = "".join(
        '<circle class="validation-sparkline-point" '
        f'cx="{x:.1f}" cy="{y:.1f}" r="3.1" tabindex="0" '
        f'aria-label="Lead {escape(lead_days[index])}: {_format_number(value)} {escape(unit)}" '
        f'data-tooltip="Lead {escape(lead_days[index])}: {_format_number(value)} {escape(unit)}"></circle>'
        for index, value, x, y in points
    )
    return (
        f'<svg class="validation-sparkline" viewBox="0 0 {width} {height}" role="img" aria-hidden="true">'
        f'<polyline points="{polyline_points}"></polyline>{circles}</svg>'
    )


def _score_card(
    label: str, unit: str, lead_days: list[str], values: list[float | None]
) -> str:
    if not lead_days:
        return ""
    return (
        '<div class="validation-score-card">'
        f"<h3>{escape(label)}</h3>"
        f"<p><span>Lead {escape(lead_days[0])}</span><strong>{_format_number(values[0])} {escape(unit)}</strong></p>"
        "<p>"
        f"<span>Lead {escape(lead_days[-1])}</span>"
        f"<strong>{_format_number(values[-1])} {escape(unit)}</strong>"
        "</p>"
        f"{_sparkline(lead_days, values, unit)}"
        "</div>"
    )


def _manifest_score_preview_lead_days(lead_values: dict) -> list[str]:
    return sorted(
        (str(lead_day) for lead_day in lead_values),
        key=lambda lead_day: (
            (0, int(lead_day)) if lead_day.isdecimal() else (1, lead_day)
        ),
    )


def _manifest_score_preview_value(value) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def render_forecast_validation_manifest_score_preview(
    score_preview: dict | None,
) -> str:
    if not isinstance(score_preview, dict):
        return '<p class="validation-empty">Preview unavailable</p>'
    metrics = score_preview.get("metrics")
    if not isinstance(metrics, list):
        return '<p class="validation-empty">Preview unavailable</p>'

    cards = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        label = metric.get("label")
        unit = metric.get("unit")
        lead_values = metric.get("lead_values")
        if (
            not isinstance(label, str)
            or not isinstance(unit, str)
            or not isinstance(lead_values, dict)
        ):
            continue
        lead_days = _manifest_score_preview_lead_days(lead_values)
        cards.append(
            _score_card(
                label,
                unit,
                lead_days,
                [
                    _manifest_score_preview_value(lead_values.get(lead_day))
                    for lead_day in lead_days
                ],
            )
        )

    if not cards:
        return '<p class="validation-empty">Preview unavailable</p>'
    return '<div class="validation-score-grid">' + "".join(cards) + "</div>"


def _sparkline_tooltip_script() -> str:
    return """
<script>
(() => {
  const tooltip = document.createElement("div");
  tooltip.className = "validation-sparkline-tooltip";
  tooltip.setAttribute("role", "tooltip");
  document.body.appendChild(tooltip);

  function showTooltip(point, clientX, clientY) {
    tooltip.textContent = point.dataset.tooltip || "";
    tooltip.style.display = "block";
    moveTooltip(clientX, clientY);
  }

  function moveTooltip(clientX, clientY) {
    const offset = 12;
    const rect = tooltip.getBoundingClientRect();
    let left = clientX + offset;
    let top = clientY - rect.height - offset;
    if (left + rect.width > window.innerWidth - 8) {
      left = clientX - rect.width - offset;
    }
    if (top < 8) {
      top = clientY + offset;
    }
    tooltip.style.left = `${Math.max(8, left)}px`;
    tooltip.style.top = `${Math.max(8, top)}px`;
  }

  function hideTooltip() {
    tooltip.style.display = "none";
  }

  document.querySelectorAll(".validation-sparkline-point").forEach((point) => {
    point.addEventListener("pointerenter", (event) => {
      showTooltip(point, event.clientX, event.clientY);
    });
    point.addEventListener("pointermove", (event) => {
      moveTooltip(event.clientX, event.clientY);
    });
    point.addEventListener("pointerleave", hideTooltip);
    point.addEventListener("focus", () => {
      const rect = point.getBoundingClientRect();
      showTooltip(point, rect.left + rect.width / 2, rect.top);
    });
    point.addEventListener("blur", hideTooltip);
  });
})();
</script>
"""


def render_forecast_validation_score_preview_script() -> str:
    return _sparkline_tooltip_script()


def render_forecast_validation_page(
    notebook_path: str | Path,
    metadata: ForecastValidationMetadata,
) -> str:
    notebook_score_name = _notebook_score_name(metadata)
    score = _observation_score(notebook_path, notebook_score_name)
    drifter_score = _drifter_score(notebook_path, notebook_score_name)
    drifter_explorer_html = _drifter_explorer_html(notebook_path)
    note_html = (
        f'<p class="validation-note">{escape(metadata.note)}</p>'
        if metadata.note
        else ""
    )
    # The representative-scores widget reuses the same manifest preview the listing page renders,
    # so the report and the listing stay consistent. Gridded Class IV RMSD sections render only
    # when the system shares scorable variables with the observations; a surface-currents-only
    # system self-suppresses them and shows just its drifter diagnostics.
    rmsd_detail_section = (
        f"""
<section class="validation-section">
  <h2>Detailed Class IV RMSD</h2>
  {_score_table(score)}
</section>
"""
        if score is not None
        else ""
    )
    observation_maps_section = (
        f"""
<section class="validation-section validation-map-section">
  <h2>Observation error maps</h2>
  <p>Errors are shown at Class IV observation locations for each available variable and lead day.</p>
  {_class4_explorer_html(notebook_path)}
</section>
"""
        if score is not None
        else ""
    )
    return f"""
<section class="validation-intro">
  {_summary_cards(metadata)}
  <p class="validation-main-message">
    Class IV evaluation is complete for lead days {escape(metadata.validated_lead_days)}.
    Scores are RMSD against recent Class IV observations for the variables the system provides,
    plus Lagrangian drifter trajectory deviation.
    These diagnostics support scientific evaluation and operational monitoring; they are not model rankings.
  </p>
  {note_html}
</section>

<section class="validation-section">
  <h2>Representative lead-time scores</h2>
  <p>Representative scores at the first and last evaluated lead day.</p>
  {render_forecast_validation_manifest_score_preview(metadata.score_preview)}
  {_sparkline_tooltip_script()}
</section>
{rmsd_detail_section}
<section class="validation-section">
  <h2>Drifter trajectory scores</h2>
  <p>Observed Class IV 15 m drifter tracks are compared with challenger-advected trajectories.</p>
  {_drifter_score_table(drifter_score)}
</section>
{observation_maps_section}
<section class="validation-section validation-map-section">
  <h2>Drifter trajectory divergence</h2>
  <p>Observed and challenger-advected drifter tracks are animated with per-lead matched counts.</p>
  {drifter_explorer_html}
</section>

<section class="validation-method-note">
  <h2>Evaluation method</h2>
  <p>
    The forecast is matched to recent Class IV observations by lead day.
    The observation cutoff defines the latest observation date included in the evaluation.
    RMSD is aggregated by variable, depth range, and lead day; Lagrangian drifter deviation by lead day.
  </p>
</section>
"""
