// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Rendering for the scores page: the ranked table and the error-growth chart.
//
// The chart is the viewer's own `leadCurveSVG` with its in-SVG legend suppressed, because
// the rail legend lays out on one row sized for two or three series and this chart carries
// one line per challenger. The legend is rendered here in HTML instead, which also doubles
// as the direct labelling that the light-theme series colours require.

import { leadCurveSVG } from "./charts.js";
import {
  cellFor,
  challengerLabel,
  challengerSeriesSlot,
  formatMean,
  formatSkill,
  isBaselineChallenger,
  leadSeries,
  skillOf,
} from "./scores-data.js";

export function seriesColorFor(challenger) {
  const slot = challengerSeriesSlot(challenger);
  const value = getComputedStyle(document.documentElement).getPropertyValue(`--scores-series-${slot}`);
  return value.trim() || "#8b97a6";
}

function textCell(text, className) {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) cell.className = className;
  return cell;
}

function headerCell(text, unit, { sortable = false, sorted = 0, align = "right", onSort } = {}) {
  const cell = document.createElement("th");
  cell.className = align === "left" ? "align-left" : "align-right";
  const label = document.createElement("span");
  label.className = "column-label";
  label.textContent = text;
  cell.appendChild(label);
  if (unit) {
    const unitLabel = document.createElement("span");
    unitLabel.className = "column-unit";
    unitLabel.textContent = unit;
    cell.appendChild(unitLabel);
  }
  if (sortable) {
    cell.classList.add("sortable");
    cell.tabIndex = 0;
    cell.setAttribute("role", "button");
    cell.title = "Rank by this column (click to cycle best first, worst first, neutral)";
    if (sorted !== 0) {
      cell.classList.add("sorted");
      const marker = document.createElement("span");
      marker.className = "sort-marker";
      marker.textContent = sorted === 1 ? "▲" : "▼";
      cell.appendChild(marker);
    }
    cell.addEventListener("click", onSort);
    cell.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        onSort();
      }
    });
  }
  return cell;
}

function scoreCell(row, isRanked) {
  const cell = document.createElement("td");
  cell.className = isRanked ? "score-cell ranked" : "score-cell";
  if (!row) {
    cell.appendChild(textContent("span", "-", "cell-empty"));
    return cell;
  }
  cell.appendChild(textContent("span", formatMean(row.mean), "cell-mean"));
  if (Number.isFinite(row.ci_low) && Number.isFinite(row.ci_high)) {
    const halfWidth = (row.ci_high - row.ci_low) / 2;
    cell.appendChild(textContent("span", `± ${formatMean(halfWidth)}`, "cell-interval"));
  }
  const skill = formatSkill(skillOf(row));
  if (skill) {
    const element = textContent("span", `skill ${skill}`, "cell-skill");
    element.title = "Error reduction against the 1 degree persistence baseline; positive is better";
    cell.appendChild(element);
  }
  return cell;
}

function textContent(tagName, text, className) {
  const element = document.createElement(tagName);
  element.textContent = text;
  element.className = className;
  return element;
}

/**
 * Draw the challenger by variable table. `sort` is `{columnKey, direction}` where a
 * direction of 1 puts the lowest error first; with direction 0 the neutral alphabetical
 * order is kept and no rank numbers are shown.
 */
export function renderTable(table, { challengers, columns, index, leadDay, sort, labelOf, onSort }) {
  const head = document.createElement("tr");
  head.appendChild(headerCell("#", null, { align: "left" }));
  head.appendChild(headerCell("Model", null, { align: "left" }));
  for (const column of columns) {
    const key = `${column.variable}|${column.depth ?? "none"}`;
    const sorted = sort.columnKey === key ? sort.direction : 0;
    head.appendChild(
      headerCell(labelOf(column), column.unit, { sortable: true, sorted, onSort: () => onSort(key) }),
    );
  }
  table.querySelector("thead").replaceChildren(head);

  const body = table.querySelector("tbody");
  body.replaceChildren();
  const ranked = sort.direction !== 0;
  challengers.forEach((challenger, position) => {
    const row = document.createElement("tr");
    if (isBaselineChallenger(challenger)) row.classList.add("baseline-row");
    row.appendChild(textCell(ranked ? String(position + 1) : "", "rank-cell"));

    const nameCell = document.createElement("td");
    nameCell.className = "model-cell";
    const swatch = document.createElement("span");
    swatch.className = "series-swatch";
    swatch.style.background = seriesColorFor(challenger);
    nameCell.appendChild(swatch);
    nameCell.appendChild(textContent("span", challengerLabel(challenger), "model-name"));
    if (isBaselineChallenger(challenger)) nameCell.appendChild(textContent("span", "baseline", "model-tag"));
    row.appendChild(nameCell);

    for (const column of columns) {
      const key = `${column.variable}|${column.depth ?? "none"}`;
      row.appendChild(scoreCell(cellFor(index, challenger, column, leadDay), sort.columnKey === key));
    }
    body.appendChild(row);
  });

  if (!challengers.length || !columns.length) {
    const empty = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = Math.max(3, columns.length + 2);
    cell.className = "cell-empty";
    cell.textContent = "No published scores for this selection.";
    empty.appendChild(cell);
    body.appendChild(empty);
  }
}

/**
 * Draw the error-growth chart: the metric against lead day, one line per challenger, each
 * with its bootstrap confidence band. Series are added in colour-slot order so the legend
 * and the drawing order stay fixed no matter how the table is ranked.
 */
export function renderErrorGrowth(host, legendHost, { challengers, column, index, leadDays, metric, labelOf }) {
  if (!column) {
    host.innerHTML = "";
    legendHost.replaceChildren();
    return;
  }
  const ordered = [...challengers].sort((first, second) => challengerSeriesSlot(first) - challengerSeriesSlot(second));
  const series = new Map();
  const labels = new Map();
  const colors = new Map();
  for (const challenger of ordered) {
    const points = leadSeries(index, challenger, column, leadDays);
    if (!points.length) continue;
    series.set(challenger, points);
    labels.set(challenger, challengerLabel(challenger));
    colors.set(challenger, seriesColorFor(challenger));
  }

  host.innerHTML = leadCurveSVG(series, {
    title: `${metric} vs lead day, ${labelOf(column)}`,
    unit: column.unit ?? "",
    labels,
    colors,
    legend: false,
    emptyMessage: "no published scores for this selection",
  });

  legendHost.replaceChildren();
  for (const challenger of series.keys()) {
    const entry = document.createElement("span");
    entry.className = "chart-legend-entry";
    const swatch = document.createElement("span");
    swatch.className = "series-swatch";
    swatch.style.background = colors.get(challenger);
    entry.appendChild(swatch);
    entry.appendChild(textContent("span", labels.get(challenger), "chart-legend-label"));
    legendHost.appendChild(entry);
  }
}

/**
 * Crosshair and anchored readout for the error-growth chart: snap to the nearest lead day
 * and list every series at it. Mirrors the viewer rail's chart interaction.
 */
export function wireChartCursor(host) {
  const svg = host.querySelector("svg");
  if (!svg) return;
  const crosshair = svg.querySelector(".chart-crosshair");
  const tooltip = svg.querySelector(".chart-tooltip");
  const box = tooltip ? tooltip.querySelector("rect") : null;
  const points = [...svg.querySelectorAll(".chart-point")];
  if (!crosshair || !tooltip || !box || !points.length) return;

  const bySeries = new Map();
  for (const point of points) {
    const line = point.dataset.line || "";
    if (!bySeries.has(line)) bySeries.set(line, []);
    bySeries.get(line).push(point);
  }

  const setLines = (lines) => {
    for (const old of [...tooltip.querySelectorAll("text")]) old.remove();
    lines.forEach((text, position) => {
      const node = document.createElementNS("http://www.w3.org/2000/svg", "text");
      node.setAttribute("x", "6");
      node.setAttribute("y", String(13 + position * 12));
      node.textContent = text;
      tooltip.appendChild(node);
    });
    box.setAttribute("height", String(8 + lines.length * 12));
    box.setAttribute("width", String(Math.max(96, 6.4 * Math.max(...lines.map((line) => line.length)))));
  };

  const move = (event) => {
    const cursor = svg.createSVGPoint();
    cursor.x = event.clientX;
    cursor.y = event.clientY;
    const local = cursor.matrixTransform(svg.getScreenCTM().inverse());
    let nearestX = null;
    let nearestDistance = Infinity;
    const lines = [];
    for (const seriesPoints of bySeries.values()) {
      let best = null;
      let bestDistance = Infinity;
      for (const point of seriesPoints) {
        const distance = Math.abs(Number(point.getAttribute("cx")) - local.x);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = point;
        }
      }
      if (!best) continue;
      lines.push(`${best.dataset.line}: ${best.dataset.yLabel}`);
      if (bestDistance < nearestDistance) {
        nearestDistance = bestDistance;
        nearestX = Number(best.getAttribute("cx"));
      }
    }
    if (nearestX === null) return;
    crosshair.setAttribute("x1", String(nearestX));
    crosshair.setAttribute("x2", String(nearestX));
    crosshair.removeAttribute("hidden");
    setLines(lines);
    const width = Number(box.getAttribute("width"));
    const x = nearestX < 180 ? 360 - width - 4 : 44;
    tooltip.setAttribute("transform", `translate(${x.toFixed(1)} 16)`);
    tooltip.removeAttribute("hidden");
  };

  const leave = () => {
    crosshair.setAttribute("hidden", "");
    tooltip.setAttribute("hidden", "");
  };

  svg.addEventListener("mousemove", move);
  svg.addEventListener("mouseleave", leave);
}
