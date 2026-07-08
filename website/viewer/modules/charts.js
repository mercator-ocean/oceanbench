// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Tiny dependency-free SVG charts for the viewer context rail (contracts.md §6:
// "quantitative curves for the current view … linked to the map state"). Two
// charts: the RMSE-vs-lead curve with bootstrap CI band (from scores-summary.json)
// and the realism PSD spectrum (challenger vs reference power, from spectra.json).
// Styling references the page CSS variables so both themes stay consistent; no
// chart library is vendored.

const VIEW_WIDTH = 360;
const VIEW_HEIGHT = 220;
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
  eastward: "#38bdf8",
  northward: "#f0a020",
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

function interactionLayer() {
  return (
    `<line class="chart-crosshair" x1="0" y1="${PAD_TOP}" x2="0" y2="${VIEW_HEIGHT - PAD_BOTTOM}" hidden/>` +
    `<g class="chart-tooltip" hidden><rect x="0" y="0" width="128" height="34" rx="4"/><text x="6" y="13"></text><text x="6" y="27"></text></g>`
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
export function leadCurveSVG(
  series,
  {
    title = "RMSD vs lead day",
    unit = "",
    labels = new Map(),
    colors = new Map(),
    emptyMessage = "no score rows for this variable/depth",
  } = {},
) {
  const area = plotArea();
  const references = [...series.keys()];
  if (!references.length) return emptyChart(title, emptyMessage);

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
  const seriesColor = (key) =>
    colors.get(key) ||
    SERIES_COLORS[key] ||
    (String(key).endsWith(":eastward") ? SERIES_COLORS.eastward : null) ||
    (String(key).endsWith(":northward") ? SERIES_COLORS.northward : null) ||
    SERIES_COLORS.reference;
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
    const color = seriesColor(reference);
    const bandTop = rows.map((row) => `${xOf(row.lead_day).toFixed(1)},${yOf(row.ci_high ?? row.mean).toFixed(1)}`);
    const bandBottom = rows
      .slice()
      .reverse()
      .map((row) => `${xOf(row.lead_day).toFixed(1)},${yOf(row.ci_low ?? row.mean).toFixed(1)}`);
    body += `<polygon points="${bandTop.concat(bandBottom).join(" ")}" fill="${color}" fill-opacity="0.16" stroke="none"/>`;
    const line = rows.map((row, index) => `${index === 0 ? "M" : "L"}${xOf(row.lead_day).toFixed(1)} ${yOf(row.mean).toFixed(1)}`);
    body += `<path d="${line.join(" ")}" fill="none" stroke="${color}" stroke-width="1.8"/>`;
    for (const row of rows) {
      const x = xOf(row.lead_day).toFixed(1);
      const y = yOf(row.mean).toFixed(1);
      body += `<circle cx="${x}" cy="${y}" r="1.8" fill="${color}"/>`;
      body += `<circle class="chart-point" data-line="${escapeText(labels.get(reference) || reference)}" data-x-label="lead day ${row.lead_day}" data-y-label="${formatValue(row.mean, unit)}" cx="${x}" cy="${y}" r="8"/>`;
    }
  }

  const legend = references
    .map((reference, index) => {
      const color = seriesColor(reference);
      const label = labels.get(reference) || reference;
      const x = area.x0 + index * 92;
      return (
        `<rect x="${x}" y="2" width="9" height="9" rx="2" fill="${color}"/>` +
        `<text x="${x + 12}" y="10" class="legend">${escapeText(label)}</text>`
      );
    })
    .join("");

  return svgOpen(title) + axes(area, "lead day", unit || "RMSD") + body + legend + interactionLayer() + "</svg>";
}

/** PSD spectrum (log-log): challenger vs reference power, plus error power. */
export function spectraSVG(entry, { title = "Power spectrum", productA = "product A", productB = "product B" } = {}) {
  const area = plotArea();
  if (!entry || !entry.wavelength || !entry.wavelength.length) {
    return emptyChart(title, "no spectrum for this variable/region");
  }
  const wavelengths = entry.wavelength;
  const lines = [
    { key: "challenger", values: entry.challenger_power, color: SERIES_COLORS.challenger, label: productA },
    { key: "reference", values: entry.reference_power, color: SERIES_COLORS.reference, label: productB },
    { key: "error", values: entry.error_power, color: SERIES_COLORS.error, label: `error (${productA}-${productB})` },
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
    for (let i = 0; i < wavelengths.length; i += 1) {
      const wavelength = wavelengths[i];
      const power = line.values[i];
      if (!(wavelength > 0) || !(power > 0)) continue;
      body += `<circle class="chart-point" data-line="${escapeText(line.label)}" data-x-label="${formatKm(wavelength)}" data-y-label="${formatPower(power)}" cx="${xOf(wavelength).toFixed(1)}" cy="${yOf(power).toFixed(1)}" r="7"/>`;
    }
  }

  const legend = lines
    .map((line, index) => {
      const x = area.x0 + index * 78;
      return `<rect x="${x}" y="2" width="9" height="9" rx="2" fill="${line.color}"/><text x="${x + 12}" y="10" class="legend">${line.label}</text>`;
    })
    .join("");

  return svgOpen(title) + axes(area, "wavelength", "power") + body + legend + interactionLayer() + "</svg>";
}

/**
 * Live PSD chart (log-log) from one or more in-browser-computed curves. `curves` is
 * an array of { label, color, wavelength: number[] (metres), power: number[], dashed }.
 * Wavelength axis in km (large scales left). Points carry per-series data attributes
 * so the cursor tooltip can report wavelength + power for the curve under the cursor.
 */
export function psdSpectraSVG(curves, { title = "Live power spectrum" } = {}) {
  const area = plotArea();
  const usable = (curves || []).filter((curve) => curve && curve.wavelength && curve.wavelength.length);
  if (!usable.length) return emptyChart(title, "no field in view for a spectrum");

  const positive = (list, pick) => {
    const out = [];
    for (let i = 0; i < list.wavelength.length; i += 1) {
      const wavelength = list.wavelength[i];
      const power = list.power[i];
      if (wavelength > 0 && power > 0) out.push(pick(wavelength, power));
    }
    return out;
  };
  const xValues = usable.flatMap((curve) => positive(curve, (wavelength) => Math.log10(wavelength)));
  const yValues = usable.flatMap((curve) => positive(curve, (_, power) => Math.log10(power)));
  if (!xValues.length || !yValues.length) return emptyChart(title, "spectrum values non-positive");
  const xMin = Math.min(...xValues);
  const xMax = Math.max(...xValues);
  const yMin = Math.min(...yValues);
  const yMax = Math.max(...yValues);
  const xOf = (wavelength) => area.x1 - ((Math.log10(wavelength) - xMin) / (xMax - xMin || 1)) * area.width;
  const yOf = (power) => area.y1 - ((Math.log10(power) - yMin) / (yMax - yMin || 1)) * area.height;

  let body = "";
  for (let t = 0; t <= 3; t += 1) {
    const y = area.y0 + (area.height * t) / 3;
    body += `<line x1="${area.x0}" y1="${y.toFixed(1)}" x2="${area.x1}" y2="${y.toFixed(1)}" class="grid"/>`;
  }
  for (let exponent = Math.ceil(xMin); exponent <= Math.floor(xMax); exponent += 1) {
    const wavelengthMetres = Math.pow(10, exponent);
    const x = xOf(wavelengthMetres);
    body += `<line x1="${x.toFixed(1)}" y1="${area.y0}" x2="${x.toFixed(1)}" y2="${area.y1}" class="grid"/>`;
    body += `<text x="${x.toFixed(1)}" y="${area.y1 + 12}" class="tick" text-anchor="middle">${formatKm(wavelengthMetres)}</text>`;
  }

  for (const curve of usable) {
    const path = [];
    for (let i = 0; i < curve.wavelength.length; i += 1) {
      const wavelength = curve.wavelength[i];
      const power = curve.power[i];
      if (!(wavelength > 0) || !(power > 0)) continue;
      path.push(`${path.length === 0 ? "M" : "L"}${xOf(wavelength).toFixed(1)} ${yOf(power).toFixed(1)}`);
    }
    body += `<path d="${path.join(" ")}" fill="none" stroke="${curve.color}" stroke-width="1.6" ${
      curve.dashed ? 'stroke-dasharray="4 3"' : ""
    }/>`;
    for (let i = 0; i < curve.wavelength.length; i += 1) {
      const wavelength = curve.wavelength[i];
      const power = curve.power[i];
      if (!(wavelength > 0) || !(power > 0)) continue;
      body += `<circle class="chart-point" data-line="${escapeText(curve.label)}" data-x-label="${formatKm(
        wavelength,
      )}" data-y-label="${formatPower(power)}" cx="${xOf(wavelength).toFixed(1)}" cy="${yOf(power).toFixed(1)}" r="6"/>`;
    }
  }

  const legend = usable
    .map((curve, index) => {
      const x = area.x0 + index * 96;
      return `<rect x="${x}" y="2" width="9" height="9" rx="2" fill="${curve.color}"/><text x="${
        x + 12
      }" y="10" class="legend">${escapeText(curve.label)}</text>`;
    })
    .join("");

  return svgOpen(title) + axes(area, "wavelength (km)", "power") + body + legend + interactionLayer() + "</svg>";
}

/**
 * RMSD by start date, one line per forecast. `series` is an array of
 * { label, color, dates: string[], rmsd: number[] }. Points carry a `data-date`
 * and `data-series-index` so the host can drill down into single-forecast scope.
 * The x-axis is the union of start dates (chronological); y is RMSD at the
 * selected lead. Values are the Class-4 RMSD pooled over all match-ups for each
 * start — the same method as the official scores — so this never shares an axis
 * with the skill-vs-lead chart (which plots RMSD against lead, not start date).
 * `yBound` (optional): a fixed y-extent — [0, niceMax(yBound)] for RMSD, ±niceMax(yBound)
 * when signed — so the axis stays STABLE across lead-day scrubs (the caller passes the
 * max across ALL leads). Without it the axis fits the plotted series, as before.
 */
export function rmsdByStartSVG(series, { title = "RMSD by start date", unit = "", signed = false, yBound = 0 } = {}) {
  const area = plotArea();
  const usable = (series || []).filter((line) => line && line.dates && line.dates.length);
  if (!usable.length) return emptyChart(title, "no year RMSD for this variable");

  const allDates = [...new Set(usable.flatMap((line) => line.dates))].sort();
  const indexOfDate = new Map(allDates.map((date, index) => [date, index]));
  const lastIndex = Math.max(1, allDates.length - 1);
  const xOf = (date) => area.x0 + (indexOfDate.get(date) / lastIndex) * area.width;

  let body = "";
  // Signed (bias) mode: symmetric y-axis centred on 0, with negative values plotted
  // below a zero baseline. |error|/RMSD mode keeps the original [0, niceMax] scale.
  if (signed) {
    let magnitude = yBound > 0 ? yBound : 0;
    if (!magnitude) {
      for (const line of usable) {
        for (const value of line.rmsd) if (Number.isFinite(value) && Math.abs(value) > magnitude) magnitude = Math.abs(value);
      }
    }
    const bound = niceMax(magnitude);
    const yOf = (value) => area.y1 - ((value + bound) / (2 * bound)) * area.height;
    for (let t = 0; t <= 4; t += 1) {
      const value = bound - (2 * bound * t) / 4;
      const y = yOf(value);
      body += `<line x1="${area.x0}" y1="${y.toFixed(1)}" x2="${area.x1}" y2="${y.toFixed(1)}" class="${
        value === 0 ? "axis" : "grid"
      }"/>`;
      body += `<text x="${area.x0 - 4}" y="${(y + 3).toFixed(1)}" class="tick" text-anchor="end">${formatTick(value)}</text>`;
    }
    const tickStep = Math.max(1, Math.round(allDates.length / 6));
    for (let i = 0; i < allDates.length; i += tickStep) {
      const date = allDates[i];
      body += `<text x="${xOf(date).toFixed(1)}" y="${area.y1 + 12}" class="tick" text-anchor="middle">${date.slice(5)}</text>`;
    }
    for (const line of usable) {
      const band = ciBandPolygon(line, xOf, yOf);
      if (band) body += `<polygon points="${band}" fill="${line.color}" fill-opacity="0.16" stroke="none"/>`;
      const points = line.dates
        .map((date, index) => ({ date, value: line.rmsd[index] }))
        .filter((point) => Number.isFinite(point.value));
      const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${xOf(point.date).toFixed(1)} ${yOf(point.value).toFixed(1)}`);
      body += `<path d="${path.join(" ")}" fill="none" stroke="${line.color}" stroke-width="1.6"/>`;
      for (const point of points) {
        const x = xOf(point.date).toFixed(1);
        const y = yOf(point.value).toFixed(1);
        body += `<circle cx="${x}" cy="${y}" r="1.8" fill="${line.color}"/>`;
        body +=
          `<circle class="chart-point year-point" data-date="${escapeText(point.date)}" ` +
          `data-line="${escapeText(line.label)}" data-x-label="${escapeText(point.date)}" ` +
          `data-y-label="${formatValue(point.value, unit)}" cx="${x}" cy="${y}" r="7"/>`;
      }
    }
    const legend = usable
      .map((line, index) => {
        const x = area.x0 + index * 118;
        return `<rect x="${x}" y="2" width="9" height="9" rx="2" fill="${line.color}"/><text x="${x + 12}" y="10" class="legend">${escapeText(line.label)}</text>`;
      })
      .join("");
    return svgOpen(title) + axes(area, "start date", unit ? `bias (${unit})` : "bias") + body + legend + interactionLayer() + "</svg>";
  }

  let maxValue = yBound > 0 ? yBound : 0;
  if (!maxValue) {
    for (const line of usable) {
      for (const value of line.rmsd) if (Number.isFinite(value) && value > maxValue) maxValue = value;
    }
  }
  const yMax = niceMax(maxValue);
  const yOf = (value) => area.y1 - (value / yMax) * area.height;

  for (let t = 0; t <= 4; t += 1) {
    const value = (yMax * t) / 4;
    const y = yOf(value);
    body += `<line x1="${area.x0}" y1="${y.toFixed(1)}" x2="${area.x1}" y2="${y.toFixed(1)}" class="grid"/>`;
    body += `<text x="${area.x0 - 4}" y="${(y + 3).toFixed(1)}" class="tick" text-anchor="end">${formatTick(value)}</text>`;
  }
  // Month-ish ticks: a handful of evenly spaced start dates.
  const tickStep = Math.max(1, Math.round(allDates.length / 6));
  for (let i = 0; i < allDates.length; i += tickStep) {
    const date = allDates[i];
    body += `<text x="${xOf(date).toFixed(1)}" y="${area.y1 + 12}" class="tick" text-anchor="middle">${date.slice(5)}</text>`;
  }

  for (const line of usable) {
    const band = ciBandPolygon(line, xOf, yOf);
    if (band) body += `<polygon points="${band}" fill="${line.color}" fill-opacity="0.16" stroke="none"/>`;
    const points = line.dates
      .map((date, index) => ({ date, value: line.rmsd[index] }))
      .filter((point) => Number.isFinite(point.value));
    const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${xOf(point.date).toFixed(1)} ${yOf(point.value).toFixed(1)}`);
    body += `<path d="${path.join(" ")}" fill="none" stroke="${line.color}" stroke-width="1.6"/>`;
    for (const point of points) {
      const x = xOf(point.date).toFixed(1);
      const y = yOf(point.value).toFixed(1);
      body += `<circle cx="${x}" cy="${y}" r="1.8" fill="${line.color}"/>`;
      body +=
        `<circle class="chart-point year-point" data-date="${escapeText(point.date)}" ` +
        `data-line="${escapeText(line.label)}" data-x-label="${escapeText(point.date)}" ` +
        `data-y-label="${formatValue(point.value, unit)}" cx="${x}" cy="${y}" r="7"/>`;
    }
  }

  const legend = usable
    .map((line, index) => {
      const x = area.x0 + index * 118;
      return `<rect x="${x}" y="2" width="9" height="9" rx="2" fill="${line.color}"/><text x="${x + 12}" y="10" class="legend">${escapeText(line.label)}</text>`;
    })
    .join("");

  return svgOpen(title) + axes(area, "start date", unit || "RMSD") + body + legend + interactionLayer() + "</svg>";
}

// 95% CI band polygon for a start-date line, in the same visual idiom as the lead-curve band.
// `line.ciLow`/`line.ciHigh` are parallel to `line.dates` (the caller selects the RMSD or bias
// pair for the active metric). Returns null when the arrays are absent (old artifacts) or too
// sparse to shade, so the chart degrades to a plain line.
function ciBandPolygon(line, xOf, yOf) {
  const low = line.ciLow;
  const high = line.ciHigh;
  if (!Array.isArray(low) || !Array.isArray(high)) return null;
  const top = [];
  const bottom = [];
  for (let index = 0; index < line.dates.length; index += 1) {
    const lo = low[index];
    const hi = high[index];
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) continue;
    const x = xOf(line.dates[index]).toFixed(1);
    top.push(`${x},${yOf(hi).toFixed(1)}`);
    bottom.push(`${x},${yOf(lo).toFixed(1)}`);
  }
  if (top.length < 2) return null;
  return top.concat(bottom.reverse()).join(" ");
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
  if (km >= 1) return `${Math.round(km).toLocaleString("en-US")} km`;
  return `${km.toFixed(1)} km`;
}

function formatValue(value, unit) {
  const formatted = Math.abs(value) < 0.01 || Math.abs(value) >= 1000 ? value.toExponential(3) : value.toFixed(4);
  return unit ? `${formatted} ${unit}` : formatted;
}

function formatPower(value) {
  return Math.abs(value) < 0.01 || Math.abs(value) >= 1000 ? value.toExponential(3) : value.toPrecision(4);
}
