// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Tiny dependency-free SVG charts for the viewer context rail (contracts.md §6:
// "quantitative curves for the current view … linked to the map state"). Two
// charts: the RMSE-vs-lead curve with bootstrap CI band (from scores-summary.json)
// and the realism PSD spectrum (challenger vs reference power, from spectra.json).
// Styling references the page CSS variables so both themes stay consistent; no
// chart library is vendored.

import { formatFixed } from "./render.js";

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

function plotArea(extraTop = 0) {
  return {
    x0: PAD_LEFT,
    y0: PAD_TOP + extraTop,
    x1: VIEW_WIDTH - PAD_RIGHT,
    y1: VIEW_HEIGHT - PAD_BOTTOM,
    width: VIEW_WIDTH - PAD_RIGHT - PAD_LEFT,
    height: VIEW_HEIGHT - PAD_BOTTOM - PAD_TOP - extraTop,
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

function renderLegend(area, entries, columnStride = 0) {
  return entries
    .map((entry, index) => {
      const x = area.x0 + index * columnStride;
      const y = 2 + index * (columnStride ? 0 : 14);
      return (
        `<rect x="${x}" y="${y}" width="9" height="9" rx="2" fill="${entry.color}"/>` +
        `<text x="${x + 12}" y="${y + 8}" class="legend">${escapeText(entry.label)}</text>`
      );
    })
    .join("");
}

/**
 * RMSE / Class-4 RMSE vs lead day, one series per reference present, each with its
 * bootstrap-CI band. `series` is a Map(reference -> [{lead_day, mean, ci_low, ci_high}]).
 *
 * The in-SVG legend lays entries out on a single row sized for the rail's two or three
 * series. `legend: false` suppresses it for a caller that renders its own legend in HTML;
 * no caller in this tree does today.
 */
export function leadCurveSVG(
  series,
  {
    title = "RMSE vs lead day",
    unit = "",
    labels = new Map(),
    colors = new Map(),
    legend = true,
    emptyMessage = "no score rows for this variable/depth",
  } = {},
) {
  const references = [...series.keys()];
  if (!references.length) return emptyChart(title, emptyMessage);
  const extraTop = legend && references.length > 1 ? (references.length - 1) * 14 : 0;
  const area = plotArea(extraTop);

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

  const legendMarkup = legend
    ? renderLegend(area, references.map((reference) => ({ color: seriesColor(reference), label: labels.get(reference) || reference })))
    : "";

  return svgOpen(title) + axes(area, "lead day", unit || "RMSE") + body + legendMarkup + interactionLayer() + "</svg>";
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

  const legend = renderLegend(area, lines, 78);

  return svgOpen(title) + axes(area, "wavelength", "power") + body + legend + interactionLayer() + "</svg>";
}

/**
 * Live PSD chart (log-log) from one or more in-browser-computed curves. `curves` is
 * an array of { label, color, wavelength: number[] (metres), power: number[], dashed }.
 * Wavelength axis in km (large scales left). Points carry per-series data attributes
 * so the cursor tooltip can report wavelength + power for the curve under the cursor.
 */
export function psdSpectraSVG(curves, { title = "Live power spectrum", xBounds = null, yBounds = null } = {}) {
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
  // Both axes are log10. `xBounds`/`yBounds` are the caller's grow-only decade window for
  // the current selection and box, in the same log10 units, so the spectrum slides inside
  // a still frame as the lead changes. They only ever widen what this lead's data needs.
  const widen = (low, high, bound) =>
    Array.isArray(bound) && Number.isFinite(bound[0]) && Number.isFinite(bound[1])
      ? [Math.min(low, bound[0]), Math.max(high, bound[1])]
      : [low, high];
  const [xMin, xMax] = widen(Math.min(...xValues), Math.max(...xValues), xBounds);
  const [yMin, yMax] = widen(Math.min(...yValues), Math.max(...yValues), yBounds);
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

  const legend = renderLegend(area, usable, 96);

  return svgOpen(title) + axes(area, "wavelength (km)", "power") + body + legend + interactionLayer() + "</svg>";
}

/**
 * RMSE by start date, one line per forecast. `series` is an array of
 * { label, color, dates: string[], rmsd: number[] }. Points carry a `data-date`
 * and `data-series-index` so the host can drill down into single-forecast scope.
 * The x-axis is the union of start dates (chronological); y is RMSE at the
 * selected lead. Values are the Class-4 RMSE pooled over all match-ups for each
 * start — the same method as the official scores — so this never shares an axis
 * with the skill-vs-lead chart (which plots RMSE against lead, not start date).
 * `yBound` (optional): a fixed y-extent — [0, niceMax(yBound)] for RMSE, ±niceMax(yBound)
 * when signed — so the axis stays STABLE across lead-day scrubs (the caller passes the
 * max across ALL leads). Without it the axis fits the plotted series, as before.
 */
export function rmsdByStartSVG(series, { title = "RMSE by start date", unit = "", signed = false, yBound = 0 } = {}) {
  const usable = (series || []).filter((line) => line && line.dates && line.dates.length);
  if (!usable.length) return emptyChart(title, "no year RMSE for this variable");
  const area = plotArea(usable.length > 1 ? (usable.length - 1) * 14 : 0);

  const allDates = [...new Set(usable.flatMap((line) => line.dates))].sort();
  const indexOfDate = new Map(allDates.map((date, index) => [date, index]));
  const lastIndex = Math.max(1, allDates.length - 1);
  const xOf = (date) => area.x0 + (indexOfDate.get(date) / lastIndex) * area.width;

  let body = "";
  // Signed (bias) mode: symmetric y-axis centred on 0, with negative values plotted
  // below a zero baseline. |error|/RMSE mode keeps the original [0, niceMax] scale.
  // Only the y scale and its gridlines differ between the two; the start-date ticks, the
  // lines and their CI bands are one piece of code reading whichever `yOf` was built.
  let yOf;
  if (signed) {
    let magnitude = yBound > 0 ? yBound : 0;
    if (!magnitude) {
      // The band is drawn, so the axis has to hold it: sizing on the line alone clipped the
      // CI at the top of the plot and showed a confident interval where the data is wide.
      for (const line of usable) {
        for (const value of seriesExtentValues(line)) {
          if (Number.isFinite(value) && Math.abs(value) > magnitude) magnitude = Math.abs(value);
        }
      }
    }
    const bound = niceMax(magnitude);
    yOf = (value) => area.y1 - ((value + bound) / (2 * bound)) * area.height;
    for (let t = 0; t <= 4; t += 1) {
      const value = bound - (2 * bound * t) / 4;
      const y = yOf(value);
      body += `<line x1="${area.x0}" y1="${y.toFixed(1)}" x2="${area.x1}" y2="${y.toFixed(1)}" class="${
        value === 0 ? "axis" : "grid"
      }"/>`;
      body += `<text x="${area.x0 - 4}" y="${(y + 3).toFixed(1)}" class="tick" text-anchor="end">${formatTick(value)}</text>`;
    }
  } else {
    let maxValue = yBound > 0 ? yBound : 0;
    if (!maxValue) {
      // Same as the signed branch: the axis is sized on everything that gets drawn, band included.
      for (const line of usable) {
        for (const value of seriesExtentValues(line)) if (Number.isFinite(value) && value > maxValue) maxValue = value;
      }
    }
    const yMax = niceMax(maxValue);
    yOf = (value) => area.y1 - (value / yMax) * area.height;
    for (let t = 0; t <= 4; t += 1) {
      const value = (yMax * t) / 4;
      const y = yOf(value);
      body += `<line x1="${area.x0}" y1="${y.toFixed(1)}" x2="${area.x1}" y2="${y.toFixed(1)}" class="grid"/>`;
      body += `<text x="${area.x0 - 4}" y="${(y + 3).toFixed(1)}" class="tick" text-anchor="end">${formatTick(value)}</text>`;
    }
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

  const legend = renderLegend(area, usable);
  const yLabel = signed ? (unit ? `bias (${unit})` : "bias") : unit || "RMSE";

  return svgOpen(title) + axes(area, "start date", yLabel) + body + legend + interactionLayer() + "</svg>";
}

/**
 * RMSE vertical profile: RMSE on the x-axis, DEPTH on the y-axis increasing DOWNWARD
 * (surface at the top), one line per forecast. `series` is an array of
 * { label, color, bins: [{ label, rmsd, n }] } where every bin's `label` is a depth-bin
 * label ordered surface→deep. Bins are placed as evenly spaced ordinal rows (the labels
 * are opaque strings, not numeric depths), so both forecasts share the depth ordering of
 * the longest series. Points carry `data-line`/`data-x-label`/`data-y-label` so the shared
 * cursor tooltip reports the depth bin, the RMSE, and the observation count on hover.
 */
export function rmsdByDepthSVG(
  series,
  { title = "RMSE vs depth", unit = "", emptyMessage = "no depth profile for this variable", xBound = 0 } = {},
) {
  // Depth-bin labels ("1500-3000 m") are wider than the numeric ticks the other charts use,
  // so this profile gets a roomier left gutter than the shared plotArea() default.
  const usable = (series || []).filter((line) => line && Array.isArray(line.bins) && line.bins.some((bin) => Number.isFinite(bin.rmsd)));
  if (!usable.length) return emptyChart(title, emptyMessage);
  const base = plotArea(usable.length > 1 ? (usable.length - 1) * 14 : 0);
  const DEPTH_PAD_LEFT = 62;
  const area = { ...base, x0: DEPTH_PAD_LEFT, width: base.x1 - DEPTH_PAD_LEFT };

  // Depth ordering (surface→deep) from the series with the most bins, so a shorter
  // profile still aligns onto the shared axis by label.
  const depthLabels = usable.reduce((best, line) => (line.bins.length > best.length ? line.bins : best), usable[0].bins).map((bin) => bin.label);
  const rowOfLabel = new Map(depthLabels.map((label, index) => [label, index]));
  const lastRow = Math.max(1, depthLabels.length - 1);
  const yOf = (index) => area.y0 + (index / lastRow) * area.height;

  let maxValue = 0;
  for (const line of usable) {
    for (const bin of line.bins) if (Number.isFinite(bin.rmsd) && bin.rmsd > maxValue) maxValue = bin.rmsd;
  }
  // `xBound` is the caller's lead-independent extent (the max over EVERY lead of this
  // artifact), so the axis stays put while the lead slider scrubs. It can only widen the
  // frame, never narrow it, so the drawn profile is always contained.
  const xMax = niceMax(Math.max(maxValue, Number.isFinite(xBound) ? xBound : 0));
  const xOf = (value) => area.x0 + (value / xMax) * area.width;

  let body = "";
  // Vertical gridlines + x ticks (RMSE).
  for (let t = 0; t <= 4; t += 1) {
    const value = (xMax * t) / 4;
    const x = xOf(value);
    body += `<line x1="${x.toFixed(1)}" y1="${area.y0}" x2="${x.toFixed(1)}" y2="${area.y1}" class="grid"/>`;
    body += `<text x="${x.toFixed(1)}" y="${area.y1 + 12}" class="tick" text-anchor="middle">${formatTick(value)}</text>`;
  }
  // Horizontal gridlines + depth-bin labels (thinned when many bins).
  const labelStep = Math.max(1, Math.round(depthLabels.length / 8));
  depthLabels.forEach((label, index) => {
    const y = yOf(index);
    body += `<line x1="${area.x0}" y1="${y.toFixed(1)}" x2="${area.x1}" y2="${y.toFixed(1)}" class="grid"/>`;
    if (index % labelStep === 0 || index === depthLabels.length - 1) {
      body += `<text x="${area.x0 - 4}" y="${(y + 3).toFixed(1)}" class="tick" text-anchor="end">${escapeText(label)}</text>`;
    }
  });

  for (const line of usable) {
    const points = line.bins
      .filter((bin) => Number.isFinite(bin.rmsd) && rowOfLabel.has(bin.label))
      .map((bin) => ({ ...bin, row: rowOfLabel.get(bin.label) }))
      .sort((a, b) => a.row - b.row);
    const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${xOf(point.rmsd).toFixed(1)} ${yOf(point.row).toFixed(1)}`);
    body += `<path d="${path.join(" ")}" fill="none" stroke="${line.color}" stroke-width="1.8"/>`;
    for (const point of points) {
      const x = xOf(point.rmsd).toFixed(1);
      const y = yOf(point.row).toFixed(1);
      const count = Number.isFinite(point.n) ? ` · n=${Number(point.n).toLocaleString("en-US")}` : "";
      body += `<circle cx="${x}" cy="${y}" r="1.8" fill="${line.color}"/>`;
      body +=
        `<circle class="chart-point" data-line="${escapeText(line.label)}" ` +
        `data-x-label="${escapeText(point.label)}${count}" data-y-label="${formatValue(point.rmsd, unit)}" ` +
        `cx="${x}" cy="${y}" r="8"/>`;
    }
  }

  const legend = renderLegend(area, usable);

  // Depth axis reads downward; the tooltip's crosshair (a vertical line) is not meaningful
  // for a profile, so only the point readout is used (interactionLayer supplies the tooltip).
  return svgOpen(title) + axes(area, unit || "RMSE", "depth") + body + legend + interactionLayer() + "</svg>";
}

/**
 * Water-column profile: a model variable's value on the x-axis against DEPTH on the
 * y-axis increasing DOWNWARD (surface at the top), one line per forecast. `series` is
 * an array of { label, color, points: [{ depth, value }] } where `depth` is in metres
 * and `value` is the model's temperature/salinity at that depth. Unlike the RMSE-vs-depth
 * chart, the y-axis carries REAL numeric depths (each forecast may have its own depth
 * levels; both are plotted faithfully on a shared metric axis) and the x-axis is a plain
 * value scale padded to the data. Points carry `data-line`/`data-x-label`/`data-y-label`
 * so the shared cursor tooltip reports the depth and the value on hover.
 */
export function columnProfileSVG(
  series,
  { title = "Water column", unit = "", xLabel = "value", emptyMessage = "no water column at this point", valueBound = null, depthBound = 0 } = {},
) {
  const usable = (series || []).filter(
    (line) => line && Array.isArray(line.points) && line.points.some((point) => Number.isFinite(point.value) && Number.isFinite(point.depth)),
  );
  if (!usable.length) return emptyChart(title, emptyMessage);
  const base = plotArea(usable.length > 1 ? (usable.length - 1) * 14 : 0);
  const DEPTH_PAD_LEFT = 52;
  const area = { ...base, x0: DEPTH_PAD_LEFT, width: base.x1 - DEPTH_PAD_LEFT };

  let depthMax = 0;
  let valueMin = Infinity;
  let valueMax = -Infinity;
  for (const line of usable) {
    for (const point of line.points) {
      if (!Number.isFinite(point.value) || !Number.isFinite(point.depth)) continue;
      depthMax = Math.max(depthMax, point.depth);
      valueMin = Math.min(valueMin, point.value);
      valueMax = Math.max(valueMax, point.value);
    }
  }
  // `valueBound`/`depthBound` are the caller's extent over EVERY lead of the column it
  // already holds in memory, so scrubbing the lead moves the profile inside a fixed
  // frame. They widen the bounds this lead's own data asks for and never narrow them.
  if (Array.isArray(valueBound) && Number.isFinite(valueBound[0]) && Number.isFinite(valueBound[1])) {
    valueMin = Math.min(valueMin, valueBound[0]);
    valueMax = Math.max(valueMax, valueBound[1]);
  }
  if (Number.isFinite(depthBound)) depthMax = Math.max(depthMax, depthBound);
  if (!(valueMax > valueMin)) valueMax = valueMin + 1;
  const padding = (valueMax - valueMin) * 0.06 || 1;
  const xLo = valueMin - padding;
  const xHi = valueMax + padding;
  const yOf = (depth) => area.y0 + (depthMax > 0 ? depth / depthMax : 0) * area.height;
  const xOf = (value) => area.x0 + ((value - xLo) / (xHi - xLo)) * area.width;

  let body = "";
  for (let t = 0; t <= 4; t += 1) {
    const value = xLo + ((xHi - xLo) * t) / 4;
    const x = xOf(value);
    body += `<line x1="${x.toFixed(1)}" y1="${area.y0}" x2="${x.toFixed(1)}" y2="${area.y1}" class="grid"/>`;
    body += `<text x="${x.toFixed(1)}" y="${area.y1 + 12}" class="tick" text-anchor="middle">${formatTick(value)}</text>`;
  }
  for (let t = 0; t <= 4; t += 1) {
    const depth = (depthMax * t) / 4;
    const y = yOf(depth);
    body += `<line x1="${area.x0}" y1="${y.toFixed(1)}" x2="${area.x1}" y2="${y.toFixed(1)}" class="grid"/>`;
    body += `<text x="${area.x0 - 4}" y="${(y + 3).toFixed(1)}" class="tick" text-anchor="end">${Math.round(depth).toLocaleString("en-US")}</text>`;
  }

  for (const line of usable) {
    const points = line.points
      .filter((point) => Number.isFinite(point.value) && Number.isFinite(point.depth))
      .slice()
      .sort((a, b) => a.depth - b.depth);
    const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${xOf(point.value).toFixed(1)} ${yOf(point.depth).toFixed(1)}`);
    body += `<path d="${path.join(" ")}" fill="none" stroke="${line.color}" stroke-width="1.8"/>`;
    for (const point of points) {
      const x = xOf(point.value).toFixed(1);
      const y = yOf(point.depth).toFixed(1);
      body += `<circle cx="${x}" cy="${y}" r="1.8" fill="${line.color}"/>`;
      body +=
        `<circle class="chart-point" data-line="${escapeText(line.label)}" ` +
        `data-x-label="${formatValue(point.value, unit)}" data-y-label="${Math.round(point.depth).toLocaleString("en-US")} m" ` +
        `cx="${x}" cy="${y}" r="8"/>`;
    }
  }

  const legend = renderLegend(area, usable);

  return svgOpen(title) + axes(area, unit ? `${xLabel} (${unit})` : xLabel, "depth (m)") + body + legend + interactionLayer() + "</svg>";
}

// Everything of a start-date line that is drawn inside the plot area: the line itself and,
// when the artifact carries them, the CI edges the band is shaded between.
function seriesExtentValues(line) {
  const values = Array.isArray(line.rmsd) ? line.rmsd.slice() : [];
  if (Array.isArray(line.ciHigh)) values.push(...line.ciHigh);
  if (Array.isArray(line.ciLow)) values.push(...line.ciLow);
  return values;
}

// 95% CI band polygon for a start-date line, in the same visual idiom as the lead-curve band.
// `line.ciLow`/`line.ciHigh` are parallel to `line.dates` (the caller selects the RMSE or bias
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

// An empty section is worth one line of explanation, not a full-height blank chart
// area: the placeholder svg is only as tall as the sentence it carries.
const EMPTY_VIEW_HEIGHT = 24;

function emptyChart(title, message) {
  return (
    `<svg viewBox="0 0 ${VIEW_WIDTH} ${EMPTY_VIEW_HEIGHT}" class="rail-chart" role="img" aria-label="${escapeText(title)}" ` +
    `preserveAspectRatio="xMidYMid meet">` +
    `<text x="0" y="${EMPTY_VIEW_HEIGHT / 2 + 3}" class="empty">${escapeText(message)}</text></svg>`
  );
}

// A tick labels the gridline it sits on, so it has to be that line's value. Gridlines are
// quarters of a "nice" maximum (1, 1.5, 2, 2.5, 3, 4, 5, 7.5 times a power of ten), and every
// such quarter is exact in four significant digits. Rounding to two decimals printed 0.07 on
// the line at 0.075 and 2e+3 on the line at 1875, which is a reader misreading the whole chart
// by up to 7% off the axis alone.
function formatTick(value) {
  if (value === 0) return "0";
  // Magnitude, not signed value: a signed comparison sent every negative tick to
  // exponential notation.
  const magnitude = Math.abs(value);
  if (Number.isInteger(value) && magnitude < 1e6) return String(value);
  if (magnitude < 0.001) return trimTrailingZeros(value.toExponential(3));
  return trimTrailingZeros(value.toPrecision(4));
}

function trimTrailingZeros(text) {
  if (!text.includes(".")) return text;
  const [mantissa, exponent] = text.split("e");
  const trimmed = mantissa.replace(/0+$/, "").replace(/\.$/, "");
  return exponent ? `${trimmed}e${exponent}` : trimmed;
}

function formatKm(metres) {
  const km = metres / 1000;
  if (km >= 1) return `${Math.round(km).toLocaleString("en-US")} km`;
  return `${km.toFixed(1)} km`;
}

function formatValue(value, unit) {
  const formatted = Math.abs(value) < 0.01 || Math.abs(value) >= 1000 ? value.toExponential(3) : formatFixed(value, 4);
  return unit ? `${formatted} ${unit}` : formatted;
}

function formatPower(value) {
  return Math.abs(value) < 0.01 || Math.abs(value) >= 1000 ? value.toExponential(3) : value.toPrecision(4);
}
