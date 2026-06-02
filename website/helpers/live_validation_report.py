# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
import re

from helpers.notebook_score_parser import get_all_model_scores_from_notebook
from helpers.type import ModelScore, ModelVariable

OBSERVATION_SCORE_KEY = "rmsd_variables_observations"
CLASS4_EXPLORER_SOURCE = "evaluation_report.class4_observation_error_explorer"
REPRESENTATIVE_SERIES = [
    ("Surface", "temperature"),
    ("0-5m", "salinity"),
    ("15m", "zonal current"),
    ("15m", "meridional current"),
]
DEPTH_ORDER = ["Surface", "0-5m", "5-100m", "100-300m", "300-600m", "15m"]
VARIABLE_ORDER = [
    "temperature",
    "salinity",
    "sea level anomaly",
    "zonal current",
    "meridional current",
]


@dataclass(frozen=True)
class ForecastValidationMetadata:
    system_label: str
    forecast_init: str
    validated_lead_days: str
    observation_cutoff: str
    status: str


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


def _class4_explorer_html(notebook_path: str | Path) -> str:
    notebook = _read_notebook(notebook_path)
    for cell in notebook.get("cells", []):
        if CLASS4_EXPLORER_SOURCE in _cell_source(cell):
            html_output = _cell_html_output(cell)
            if html_output:
                return html_output
    return '<p class="validation-empty">Class IV observation error maps are not available in this report.</p>'


def _observation_score(notebook_path: str | Path, system_label: str) -> ModelScore:
    scores = get_all_model_scores_from_notebook(str(notebook_path), system_label.lower())
    if OBSERVATION_SCORE_KEY not in scores:
        raise ValueError(f"Missing {OBSERVATION_SCORE_KEY} in {notebook_path}.")
    return scores[OBSERVATION_SCORE_KEY]


def _lead_days(score: ModelScore) -> list[str]:
    for depth in score.depths.values():
        for variable in depth.variables.values():
            return sorted(variable.data, key=lambda lead_day: int(lead_day))
    return []


def _depth_sort_key(depth: str) -> int:
    return DEPTH_ORDER.index(depth) if depth in DEPTH_ORDER else len(DEPTH_ORDER)


def _variable_sort_key(variable: str) -> int:
    return VARIABLE_ORDER.index(variable) if variable in VARIABLE_ORDER else len(VARIABLE_ORDER)


def _score_rows(score: ModelScore) -> list[tuple[str, str, ModelVariable]]:
    rows = [
        (depth, variable_name, variable)
        for depth, depth_score in score.depths.items()
        for variable_name, variable in depth_score.variables.items()
    ]
    return sorted(rows, key=lambda row: (_variable_sort_key(row[1]), _depth_sort_key(row[0])))


def _format_depth(depth: str) -> str:
    if depth == "Surface":
        return "surface"
    return re.sub(r"(?<=\d)m$", " m", depth)


def _format_variable(variable: str) -> str:
    return variable.capitalize()


def _format_number(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def _score_label(depth: str, variable: str) -> str:
    return f"{_format_variable(variable)}, {_format_depth(depth)}"


def _summary_cards(metadata: ForecastValidationMetadata) -> str:
    cards = [
        ("System", metadata.system_label),
        ("Forecast init", metadata.forecast_init),
        ("Validated lead days", metadata.validated_lead_days),
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
        + "".join(f"<td>{_format_number(variable.data.get(lead_day))}</td>" for lead_day in lead_days)
        + "</tr>"
        for depth, variable_name, variable in _score_rows(score)
    )
    return (
        '<div class="validation-score-table-wrap">'
        f'<table class="validation-score-table">{header}<tbody>{body}</tbody></table>'
        "</div>"
    )


def _representative_variables(score: ModelScore) -> list[tuple[str, ModelVariable]]:
    selected_variables = []
    for depth, variable_name in REPRESENTATIVE_SERIES:
        depth_score = score.depths.get(depth)
        if depth_score is None or variable_name not in depth_score.variables:
            continue
        selected_variables.append((_score_label(depth, variable_name), depth_score.variables[variable_name]))
    return selected_variables


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
        y = height - padding - ((value - minimum) / value_range) * (height - 2 * padding)
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


def _score_cards(score: ModelScore) -> str:
    lead_days = _lead_days(score)
    cards = []
    for label, variable in _representative_variables(score):
        values = [variable.data.get(lead_day) for lead_day in lead_days]
        first_value = next((value for value in values if value is not None), None)
        last_value = next((value for value in reversed(values) if value is not None), None)
        cards.append(
            '<div class="validation-score-card">'
            f"<h3>{escape(label)}</h3>"
            f"<p><span>Lead 1</span><strong>{_format_number(first_value)} {escape(variable.unit)}</strong></p>"
            "<p>"
            f"<span>Lead {escape(lead_days[-1])}</span>"
            f"<strong>{_format_number(last_value)} {escape(variable.unit)}</strong>"
            "</p>"
            f"{_sparkline(lead_days, values, variable.unit)}"
            "</div>"
        )
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


def render_forecast_validation_page(
    notebook_path: str | Path,
    metadata: ForecastValidationMetadata,
) -> str:
    score = _observation_score(notebook_path, metadata.system_label)
    explorer_html = _class4_explorer_html(notebook_path)
    return f"""
<section class="validation-intro">
  {_summary_cards(metadata)}
  <p class="validation-main-message">
    Class IV validation is complete for lead days {escape(metadata.validated_lead_days)}.
    Scores are RMSD against recent observations for temperature, salinity, and currents.
    These diagnostics support scientific validation and operational monitoring; they are not model rankings.
  </p>
</section>

<section class="validation-section">
  <h2>Lead-time scores</h2>
  <p>Representative RMSD values are shown at the first and last validated lead day.</p>
  {_score_cards(score)}
  {_sparkline_tooltip_script()}
</section>

<section class="validation-section">
  <h2>Detailed Class IV RMSD</h2>
  {_score_table(score)}
</section>

<section class="validation-section validation-map-section">
  <h2>Observation error maps</h2>
  <p>Errors are shown at Class IV observation locations for each available variable and lead day.</p>
  {explorer_html}
</section>

<section class="validation-method-note">
  <h2>Validation method</h2>
  <p>
    The forecast is matched to recent Class IV observations by lead day.
    The observation cutoff defines the latest observation date included in the validation.
    RMSD is aggregated by variable, depth range, and lead day.
  </p>
</section>
"""
