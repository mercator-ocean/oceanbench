// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Tiny dependency-free SVG charts for the viewer context rail (contracts.md §6:
// "quantitative curves for the current view … linked to the map state"). Two
// charts: the RMSE-vs-lead curve with bootstrap CI band (from scores-summary.json)
// and the realism PSD spectrum (challenger vs reference power, from spectra.json).
// Styling references the page CSS variables so both themes stay consistent; no
// chart library is vendored.

const VIEW_WIDTH = 320;
const VIEW_HEIGHT = 180;
const PAD_LEFT = 40;
const PAD_RIGHT = 12;
const PAD_TOP = 14;
const PAD_BOTTOM = 30;

// Reference → hue, so the same source reads identically across both charts.
export const SERIES_COLORS = {
  glorys: "#38bdf8",
  glo12: "#f0a020",
  observations: "#3ddc97",
  challenger: "#38bdf8",
  reference: "#8b97a6",
  error: "#ff6b6b",
};

function escapeText(value) {
  return String(value).replace(/[<>&]/g, (character) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[character]));
}

function plotArea() {
  return {
    x0: PAD_LEFT,
    y0: PAD_TOP,
    x1: VIEW_WIDTH - PAD_RIGHT,
    y1: VIEW_HEIGHT - PAD_BOTTOM,
    width: VIEW_WIDTH - PAD_RIGHT - PAD_LEFT,
    height: VIEW_HEIGHT - PAD_BOTTOM - PAD_TOP,
  };
}

function svgOpen(title) {
  return (
    `<svg viewBox="0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}" class="rail-chart" role="img" aria-label="${escapeText(title)}" ` +
    `preserveAspectRatio="xMidYMid meet">`
  );
}

function axes(area, xLabel, yLabel) {
  return (
    `<line x1="${area.x0}" y1="${area.y1}" x2="${area.x1}" y2="${area.y1}" class="axis"/>` +
    `<line x1="${area.x0}" y1="${area.y0}" x2="${area.x0}" y2="${area.y1}" class="axis"/>` +
    `<text x="${(area.x0 + area.x1) / 2}" y="${VIEW_HEIGHT - 6}" class="axis-label" text-anchor="middle">${escapeText(xLabel)}</text>` +
    `<text x="10" y="${(area.y0 + area.y1) / 2}" class="axis-label" text-anchor="middle" transform="rotate(-90 10 ${(area.y0 + area.y1) / 2})">${escapeText(yLabel)}</text>`
  );
}

function niceMax(value) {
  if (!(value > 0)) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  for (const step of [1, 1.5, 2, 2.5, 3, 4, 5, 7.5, 10]) {
    if (value <= step * magnitude) return step * magnitude;
  }
  return 10 * magnitude;
}

/**
 * RMSE / Class-4 RMSD vs lead day, one series per reference present, each with its
 * bootstrap-CI band. `series` is a Map(reference -> [{lead_day, mean, ci_low, ci_high}]).
 */
export function leadCurveSVG(series, { title = "Skill vs lead", unit = "" } = {}) {
  const area = plotArea();
  const references = [...series.keys()];
  if (!references.length) return emptyChart(title, "no score rows for this variable/depth");

  let maxLead = 1;
  let maxValue = 0;
  for (const rows of series.values()) {
    for (const row of rows) {
      maxLead = Math.max(maxLead, row.lead_day);
      maxValue = Math.max(maxValue, row.ci_high ?? row.mean, row.mean);
    }
  }
  const yMax = niceMax(maxValue);
  const xOf = (lead) => area.x0 + (maxLead <= 1 ? 0.5 : (lead - 1) / (maxLead - 1)) * area.width;
  const yOf = (value) => area.y1 - (value / yMax) * area.height;

  let body = "";
  // Horizontal gridlines + y ticks.
  for (let t = 0; t <= 4; t += 1) {
    const value = (yMax * t) / 4;
    const y = yOf(value);
    body += `<line x1="${area.x0}" y1="${y.toFixed(1)}" x2="${area.x1}" y2="${y.toFixed(1)}" class="grid"/>`;
    body += `<text x="${area.x0 - 4}" y="${(y + 3).toFixed(1)}" class="tick" text-anchor="end">${formatTick(value)}</text>`;
  }
  for (let lead = 1; lead <= maxLead; lead += maxLead > 6 ? 2 : 1) {
    body += `<text x="${xOf(lead).toFixed(1)}" y="${area.y1 + 12}" class="tick" text-anchor="middle">${lead}</text>`;
  }

  for (const reference of references) {
    const rows = [...series.get(reference)].sort((a, b) => a.lead_day - b.lead_day);
    const color = SERIES_COLORS[reference] || SERIES_COLORS.reference;
    const bandTop = rows.map((row) => `${xOf(row.lead_day).toFixed(1)},${yOf(row.ci_high ?? row.mean).toFixed(1)}`);
    const bandBottom = rows
      .slice()
      .reverse()
      .map((row) => `${xOf(row.lead_day).toFixed(1)},${yOf(row.ci_low ?? row.mean).toFixed(1)}`);
    body += `<polygon points="${bandTop.concat(bandBottom).join(" ")}" fill="${color}" fill-opacity="0.16" stroke="none"/>`;
    const line = rows.map((row, index) => `${index === 0 ? "M" : "L"}${xOf(row.lead_day).toFixed(1)} ${yOf(row.mean).toFixed(1)}`);
    body += `<path d="${line.join(" ")}" fill="none" stroke="${color}" stroke-width="1.8"/>`;
    for (const row of rows) {
      body += `<circle cx="${xOf(row.lead_day).toFixed(1)}" cy="${yOf(row.mean).toFixed(1)}" r="1.8" fill="${color}"/>`;
    }
  }

  const legend = references
    .map((reference, index) => {
      const color = SERIES_COLORS[reference] || SERIES_COLORS.reference;
      const x = area.x0 + index * 92;
      return (
        `<rect x="${x}" y="2" width="9" height="9" rx="2" fill="${color}"/>` +
        `<text x="${x + 12}" y="10" class="legend">${escapeText(reference)}</text>`
      );
    })
    .join("");

  return svgOpen(title) + axes(area, "lead day", unit || "RMSD") + body + legend + "</svg>";
}

/** PSD spectrum (log-log): challenger vs reference power, plus error power. */
export function spectraSVG(entry, { title = "Power spectrum" } = {}) {
  const area = plotArea();
  if (!entry || !entry.wavelength || !entry.wavelength.length) {
    return emptyChart(title, "no spectrum for this variable/region");
  }
  const wavelengths = entry.wavelength;
  const lines = [
    { key: "challenger", values: entry.challenger_power, color: SERIES_COLORS.challenger, label: "challenger" },
    { key: "reference", values: entry.reference_power, color: SERIES_COLORS.reference, label: "reference" },
    { key: "error", values: entry.error_power, color: SERIES_COLORS.error, label: "error" },
  ].filter((line) => Array.isArray(line.values) && line.values.length);

  const positive = (list) => list.filter((value) => value > 0);
  const xValues = positive(wavelengths).map((value) => Math.log10(value));
  const allPowers = lines.flatMap((line) => positive(line.values)).map((value) => Math.log10(value));
  if (!xValues.length || !allPowers.length) return emptyChart(title, "spectrum values non-positive");
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...allPowers);
  const yMax = Math.max(...allPowers);
  const xOf = (wavelength) => area.x1 - ((Math.log10(wavelength) - xMin) / (xMax - xMin || 1)) * area.width;
  const yOf = (power) => area.y1 - ((Math.log10(power) - yMin) / (yMax - yMin || 1)) * area.height;

  let body = "";
  for (let t = 0; t <= 3; t += 1) {
    const y = area.y0 + (area.height * t) / 3;
    body += `<line x1="${area.x0}" y1="${y.toFixed(1)}" x2="${area.x1}" y2="${y.toFixed(1)}" class="grid"/>`;
  }
  // Wavelength ticks (km) at decade boundaries, right-to-left (large scales left).
  for (let exponent = Math.ceil(xMin); exponent <= Math.floor(xMax); exponent += 1) {
    const wavelengthMetres = Math.pow(10, exponent);
    const x = xOf(wavelengthMetres);
    body += `<line x1="${x.toFixed(1)}" y1="${area.y0}" x2="${x.toFixed(1)}" y2="${area.y1}" class="grid"/>`;
    body += `<text x="${x.toFixed(1)}" y="${area.y1 + 12}" class="tick" text-anchor="middle">${formatKm(wavelengthMetres)}</text>`;
  }

  for (const line of lines) {
    const path = [];
    for (let i = 0; i < wavelengths.length; i += 1) {
      const wavelength = wavelengths[i];
      const power = line.values[i];
      if (!(wavelength > 0) || !(power > 0)) continue;
      path.push(`${path.length === 0 ? "M" : "L"}${xOf(wavelength).toFixed(1)} ${yOf(power).toFixed(1)}`);
    }
    body += `<path d="${path.join(" ")}" fill="none" stroke="${line.color}" stroke-width="1.6" ${
      line.key === "error" ? 'stroke-dasharray="4 3"' : ""
    }/>`;
  }

  const legend = lines
    .map((line, index) => {
      const x = area.x0 + index * 78;
      return `<rect x="${x}" y="2" width="9" height="9" rx="2" fill="${line.color}"/><text x="${x + 12}" y="10" class="legend">${line.label}</text>`;
    })
    .join("");

  return svgOpen(title) + axes(area, "wavelength", "power") + body + legend + "</svg>";
}

function emptyChart(title, message) {
  return (
    svgOpen(title) +
    `<text x="${VIEW_WIDTH / 2}" y="${VIEW_HEIGHT / 2}" class="empty" text-anchor="middle">${escapeText(message)}</text></svg>`
  );
}

function formatTick(value) {
  if (value === 0) return "0";
  if (value < 0.01 || value >= 1000) return value.toExponential(0);
  return value < 1 ? value.toFixed(2) : value.toFixed(1);
}

function formatKm(metres) {
  const km = metres / 1000;
  if (km >= 1000) return `${Math.round(km / 1000)}Mm`;
  if (km >= 1) return `${Math.round(km)}km`;
  return `${km.toFixed(1)}km`;
}
