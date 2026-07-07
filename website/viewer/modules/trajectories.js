// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

import { FORECAST_COLORS } from "./forecast-colors.js";

export const TRAJECTORY_COLORS = FORECAST_COLORS;

export function trajectorySeparationSVG(rows, currentLead) {
  if (!rows || rows.length < 2) return "";
  const width = 360;
  const height = 190;
  const left = 42;
  const right = 12;
  const top = 14;
  const bottom = 30;
  const maximumLead = Math.max(1, ...rows.map((row) => row.lead_day));
  const maximumValue = Math.max(1, ...rows.map((row) => row.mean));
  const x = (lead) => left + lead / maximumLead * (width - left - right);
  const y = (value) => height - bottom - value / maximumValue * (height - top - bottom);
  let grid = "";
  for (let tick = 0; tick <= 4; tick += 1) {
    const value = maximumValue * tick / 4;
    const py = y(value);
    grid += `<line x1="${left}" y1="${py}" x2="${width - right}" y2="${py}" class="grid"/>`;
    grid += `<text x="${left - 4}" y="${py + 3}" class="tick" text-anchor="end">${value.toFixed(value < 10 ? 1 : 0)}</text>`;
  }
  const path = rows.map((row, index) => `${index ? "L" : "M"}${x(row.lead_day).toFixed(1)} ${y(row.mean).toFixed(1)}`).join(" ");
  const points = rows.map((row) => {
    const px = x(row.lead_day).toFixed(1);
    const py = y(row.mean).toFixed(1);
    return `<circle cx="${px}" cy="${py}" r="2" fill="${TRAJECTORY_COLORS[0]}"/>`
      + `<circle class="chart-point" data-line="mean separation" data-x-label="lead day ${row.lead_day}" `
      + `data-y-label="${row.mean.toFixed(1)} km" cx="${px}" cy="${py}" r="8"/>`;
  }).join("");
  const markerRow = Number.isFinite(currentLead)
    ? rows.reduce((closest, row) =>
        Math.abs(row.lead_day - currentLead) < Math.abs(closest.lead_day - currentLead) ? row : closest, rows[0])
    : null;
  let marker = "";
  if (markerRow) {
    const mx = x(markerRow.lead_day);
    const my = y(markerRow.mean);
    const label = `day ${markerRow.lead_day} · ${markerRow.mean.toFixed(markerRow.mean < 10 ? 1 : 0)} km`;
    const labelWidth = 8 + label.length * 4.6;
    const labelRight = mx + 8 + labelWidth <= width - right;
    const labelX = labelRight ? mx + 8 : mx - 8 - labelWidth;
    const labelY = Math.max(top + 2, my - 20);
    marker = `<line class="chart-lead-line" x1="${mx.toFixed(1)}" y1="${top}" x2="${mx.toFixed(1)}" y2="${(height - bottom).toFixed(1)}"/>`
      + `<circle class="chart-lead-marker" cx="${mx.toFixed(1)}" cy="${my.toFixed(1)}" r="3.5"/>`
      + `<g class="chart-lead-readout"><rect x="${labelX.toFixed(1)}" y="${labelY.toFixed(1)}" width="${labelWidth.toFixed(1)}" height="14" rx="3"/>`
      + `<text x="${(labelX + labelWidth / 2).toFixed(1)}" y="${(labelY + 9.5).toFixed(1)}" text-anchor="middle">${label}</text></g>`;
  }
  return `<svg viewBox="0 0 ${width} ${height}" class="rail-chart trajectory-separation-chart" role="img" aria-label="Trajectory separation">`
    + `<line x1="${left}" y1="${height - bottom}" x2="${width - right}" y2="${height - bottom}" class="axis"/>`
    + `<line x1="${left}" y1="${top}" x2="${left}" y2="${height - bottom}" class="axis"/>`
    + `<text x="${(left + width - right) / 2}" y="${height - 6}" class="axis-label" text-anchor="middle">lead day</text>`
    + `<text x="10" y="${(top + height - bottom) / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90 10 ${(top + height - bottom) / 2})">km</text>`
    + grid + `<path d="${path}" fill="none" stroke="${TRAJECTORY_COLORS[0]}" stroke-width="2"/>` + points + marker
    + `<line class="chart-crosshair" x1="0" y1="${top}" x2="0" y2="${height - bottom}" hidden/>`
    + `<g class="chart-tooltip" hidden><rect x="0" y="0" width="128" height="34" rx="4"/></g></svg>`;
}
