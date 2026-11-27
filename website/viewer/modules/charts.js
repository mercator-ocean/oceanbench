// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Tiny dependency-free SVG charts for the viewer context rail (contracts.md §6:
// "quantitative curves for the current view … linked to the map state"), starting
// with the RMSE-vs-lead curve with bootstrap CI band (from scores-summary.json).
// Styling references the page CSS variables so both themes stay consistent; no
// chart library is vendored.

import { formatFixed } from "./render.js";

const VIEW_WIDTH = 360;
const VIEW_HEIGHT = 220;
// Room for a five-character tick. The y quantity and unit live in the HTML chart title,
// so there is no rotated axis title in this gutter.
const PAD_LEFT = 38;
const PAD_RIGHT = 12;
// Gap above the plot (and below any legend rows), so the top tick never touches the edge.
const PAD_TOP = 8;
const PAD_BOTTOM = 18;

// Left gutter for a y axis whose tick labels are wider than the shared PAD_LEFT allows.
// Tick type is pinned in px, so in view units it grows by typeScale as the slot narrows:
// a character is about 6 units at typeScale 1, and each label sits 4 units left of the axis.
function labelGutter(labels, typeScale, minimum) {
  const longest = Math.max(0, ...labels.map((label) => String(label).length));
  return Math.max(minimum, 8 + longest * 6 * typeScale);
}

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

// `legendRows` rows of legend sit above the plot; a single-series chart has none.
function plotArea(legendRows = 0, endLabelWidth = 0) {
  const top = legendRows * LEGEND_ROW + PAD_TOP;
  const right = PAD_RIGHT + endLabelWidth;
  return {
    x0: PAD_LEFT,
    y0: top,
    x1: VIEW_WIDTH - right,
    y1: VIEW_HEIGHT - PAD_BOTTOM,
    width: VIEW_WIDTH - right - PAD_LEFT,
    height: VIEW_HEIGHT - PAD_BOTTOM - top,
  };
}

// `titleUnit` is shown after the HTML chart title, since the chart draws no y-axis title.
function svgOpen(title, titleUnit = "") {
  return (
    `<svg viewBox="0 0 ${VIEW_WIDTH} ${VIEW_HEIGHT}" class="rail-chart" role="img" aria-label="${escapeText(title)}" ` +
    `data-title-unit="${escapeText(titleUnit)}" preserveAspectRatio="xMidYMid meet">`
  );
}

function unitInParentheses(label) {
  const match = /\(([^()]*)\)\s*$/.exec(String(label || ""));
  return match ? match[1] : "";
}

function interactionLayer() {
  return (
    `<line class="chart-crosshair" x1="0" y1="${PAD_TOP}" x2="0" y2="${VIEW_HEIGHT - PAD_BOTTOM}" hidden/>` +
    `<g class="chart-tooltip" hidden><rect x="0" y="0" width="128" height="34" rx="4"/></g>`
  );
}

function axes(area, xLabel) {
  return (
    `<line x1="${area.x0}" y1="${area.y1}" x2="${area.x1}" y2="${area.y1}" class="axis"/>` +
    `<line x1="${area.x0}" y1="${area.y0}" x2="${area.x0}" y2="${area.y1}" class="axis"/>` +
    `<title>${escapeText(xLabel)}</title>`
  );
}

// Maxima whose quarters are short numbers (0.3, 0.6, 0.9, 1.2 rather than 0.1875), since
// the gridlines sit at quarters of this value and each one carries a tick label.
function niceMax(value) {
  if (!(value > 0)) return 1;
  const magnitude = Math.pow(10, Math.floor(Math.log10(value)));
  for (const step of [1, 1.2, 1.6, 2, 2.4, 3.2, 4, 6, 8, 10]) {
    if (value <= step * magnitude) return step * magnitude;
  }
  return 10 * magnitude;
}

// The v component shares its forecast's colour with u, so it is told apart by a dash.
const DASH_PATTERN = "5 3";
const DASH_PATTERN_SHORT = "3 2";

function isDashedSeries(key) {
  return key === "northward" || String(key).endsWith(":northward");
}

// Legend rows, in viewBox units. Label type is pinned at 10 px whatever the slot width,
// so in viewBox units it grows as the rail narrows; these sizes hold down to a 288 px
// slot (10 px text drawn at 12.5 units, about 7.2 units per character).
const LEGEND_ROW = 15;
const LEGEND_CHAR = 7.2;

// Line-end names in a right gutter, for every multi-series chart: the gutter is sized on
// the longest name, and names that would overlap are pushed apart by one legend row.
// Every rail chart reserves the same right gutter, named lines or not, so the plots
// share one width and their x axes line up.
const END_LABEL_GUTTER = 18 + 7 * LEGEND_CHAR;
function gutterWidth() {
  return END_LABEL_GUTTER;
}

function renderGutterLabels(area, items) {
  const placed = items.filter((item) => Number.isFinite(item.y)).sort((a, b) => a.y - b.y);
  for (let i = 1; i < placed.length; i += 1) placed[i].y = Math.max(placed[i].y, placed[i - 1].y + LEGEND_ROW - 2);
  const overflow = placed.length ? placed.at(-1).y - (area.y1 + 4) : 0;
  if (overflow > 0) for (const item of placed) item.y -= overflow;
  return placed
    .map((item) => {
      const x = area.x1 + 4;
      const dash = item.dashed ? ` stroke-dasharray="${DASH_PATTERN_SHORT}"` : "";
      return (
        `<line x1="${x}" y1="${item.y.toFixed(1)}" x2="${x + 8}" y2="${item.y.toFixed(1)}" stroke="${item.color}" stroke-width="2"${dash}/>` +
        `<text x="${x + 11}" y="${(item.y + 4).toFixed(1)}" class="legend">${escapeText(item.label)}</text>`
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
    yLabel = unit ? `RMSE (${unit})` : "RMSE",
    heading = "",
    headingSuffix = "",
  } = {},
) {
  const references = [...series.keys()];
  if (!references.length) return emptyChart(title, emptyMessage);
  const seriesColor = (key) =>
    colors.get(key) ||
    SERIES_COLORS[key] ||
    (String(key).endsWith(":eastward") ? SERIES_COLORS.eastward : null) ||
    (String(key).endsWith(":northward") ? SERIES_COLORS.northward : null) ||
    SERIES_COLORS.reference;
  const legendEntries = references.map((reference) => ({
    color: seriesColor(reference),
    label: labels.get(reference) || reference,
    dashed: isDashedSeries(reference),
  }));
  const headingRows = heading ? 1 : 0;

  let maxLead = 1;
  let maxValue = 0;
  for (const rows of series.values()) {
    for (const row of rows) {
      maxLead = Math.max(maxLead, row.lead_day);
      maxValue = Math.max(maxValue, row.ci_high ?? row.mean, row.mean);
    }
  }
  const yMax = niceMax(maxValue);
  // Two or more lines are named at their right end; when two names would collide the
  // chart falls back to one legend row above the plot.
  const named = legend && references.length > 1;
  const area = plotArea(headingRows, gutterWidth());
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
    const color = seriesColor(reference);
    const bandTop = rows.map((row) => `${xOf(row.lead_day).toFixed(1)},${yOf(row.ci_high ?? row.mean).toFixed(1)}`);
    const bandBottom = rows
      .slice()
      .reverse()
      .map((row) => `${xOf(row.lead_day).toFixed(1)},${yOf(row.ci_low ?? row.mean).toFixed(1)}`);
    body += `<polygon points="${bandTop.concat(bandBottom).join(" ")}" fill="${color}" fill-opacity="0.16" stroke="none"/>`;
    const line = rows.map((row, index) => `${index === 0 ? "M" : "L"}${xOf(row.lead_day).toFixed(1)} ${yOf(row.mean).toFixed(1)}`);
    const dash = isDashedSeries(reference) ? ` stroke-dasharray="${DASH_PATTERN}"` : "";
    body += `<path d="${line.join(" ")}" fill="none" stroke="${color}" stroke-width="1.8"${dash}/>`;
    for (const row of rows) {
      const x = xOf(row.lead_day).toFixed(1);
      const y = yOf(row.mean).toFixed(1);
      body += `<circle cx="${x}" cy="${y}" r="1.8" fill="${color}"/>`;
      body += `<circle class="chart-point" data-line="${escapeText(labels.get(reference) || reference)}" data-x-label="lead day ${row.lead_day}" data-y-label="${formatValue(row.mean, unit)}" cx="${x}" cy="${y}" r="8"/>`;
    }
  }

  const legendMarkup = named
    ? renderGutterLabels(
        area,
        references.map((reference, index) => {
          const last = [...series.get(reference)].sort((a, b) => a.lead_day - b.lead_day).at(-1);
          return { ...legendEntries[index], y: last ? yOf(last.mean) : NaN };
        }),
      )
    : "";
  // A chart that shares its section with another one names itself in one horizontal row
  // above the plot instead of in the section title.
  const suffixMarkup = headingSuffix ? ` <tspan class="heading-suffix">${escapeText(headingSuffix)}</tspan>` : "";
  const headingMarkup = heading ? `<text x="${area.x0}" y="9" class="legend">${escapeText(heading)}${suffixMarkup}</text>` : "";

  return (
    svgOpen(title, heading ? "" : unitInParentheses(yLabel) || unit) +
    axes(area, "lead day") + body + headingMarkup + legendMarkup + interactionLayer() + "</svg>"
  );
}

/**
 * Live PSD chart (log-log) from one or more in-browser-computed curves. `curves` is
 * an array of { label, color, wavelength: number[] (metres), power: number[], dashed }.
 * Wavelength axis in km (large scales left). Points carry per-series data attributes
 * so the cursor tooltip can report wavelength + power for the curve under the cursor.
 * `yLabel` names the y quantity and its units; `marker` is an optional
 * { wavelength (metres), label } vertical line, used for the scale of disagreement.
 */
export function psdSpectraSVG(
  curves,
  { title = "Live power spectrum", xBounds = null, yBounds = null, yLabel = "power", marker = null } = {},
) {
  const usable = (curves || []).filter((curve) => curve && curve.wavelength && curve.wavelength.length);
  const area = plotArea(0, gutterWidth());
  if (!usable.length) return emptyChart(title, "no field in view for a spectrum");
  const ends = [];

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
    let endIndex = -1;
    for (let i = 0; i < curve.wavelength.length; i += 1) {
      if (curve.wavelength[i] > 0 && curve.power[i] > 0 && (endIndex < 0 || curve.wavelength[i] < curve.wavelength[endIndex])) endIndex = i;
    }
    if (endIndex >= 0) ends.push({ label: curve.label, color: curve.color, dashed: curve.dashed, y: yOf(curve.power[endIndex]) });
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

  // Effective-resolution marker: a labelled vertical line at one wavelength, drawn last
  // so it sits over the curves. `marker` is { wavelength (metres), label }.
  if (marker && marker.wavelength > 0) {
    const x = xOf(marker.wavelength);
    if (x >= area.x0 && x <= area.x1) {
      body += `<line x1="${x.toFixed(1)}" y1="${area.y0}" x2="${x.toFixed(1)}" y2="${area.y1}" class="axis" stroke-dasharray="2 3"/>`;
      const anchor = x > area.x0 + area.width / 2 ? "end" : "start";
      const offset = anchor === "end" ? -4 : 4;
      body += `<text x="${(x + offset).toFixed(1)}" y="${area.y0 + 10}" class="tick" text-anchor="${anchor}">${escapeText(
        marker.label,
      )}</text>`;
    }
  }

  const legend = usable.length > 1 ? renderGutterLabels(area, ends) : "";

  return svgOpen(title, unitInParentheses(yLabel)) + axes(area, "wavelength (km)") + body + legend + interactionLayer() + "</svg>";
}

/**
 * RMSE by start date, one line per forecast. `series` is an array of
 * { label, color, dates: string[], rmsd: number[] }. Points carry a `data-date`
 * and `data-series-index` so the host can drill down into single-forecast scope.
 * The x-axis is the union of start dates (chronological); y is RMSE at the
 * selected lead. Values are the Class-4 RMSE pooled over all match-ups for each
 * start, the same method as the official scores, so this never shares an axis
 * with the skill-vs-lead chart (which plots RMSE against lead, not start date).
 * `yBound` (optional): a fixed y-extent, [0, niceMax(yBound)] for RMSE, ±niceMax(yBound)
 * when signed, so the axis stays STABLE across lead-day scrubs (the caller passes the
 * max across ALL leads). Without it the axis fits the plotted series, as before.
 */
export function rmsdByStartSVG(series, { title = "RMSE by start date", unit = "", signed = false, yBound = 0, typeScale = 1 } = {}) {
  const usable = (series || []).filter((line) => line && line.dates && line.dates.length);
  if (!usable.length) return emptyChart(title, "no RMSE by start for this variable");
  const area = plotArea(0, gutterWidth());
  const ends = [];

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

  // Month-ish ticks: a handful of evenly spaced start dates, spaced by the rendered label
  // width so a short series or a narrow slot does not crowd its labels. An
  // "MM-DD" label is 5 monospace characters (see labelGutter); the first label starts at
  // its tick and a right-edge one ends at it, so neighbours sit a label and a half plus
  // one character apart.
  const tickSpacing = (1.5 * 5 + 1) * 6 * typeScale;
  const tickStep = Math.max(1, Math.round(allDates.length / 6), Math.ceil(tickSpacing / (area.width / lastIndex)));
  for (let i = 0; i < allDates.length; i += tickStep) {
    const date = allDates[i];
    // The first date sits on the left edge, so it starts there instead of centring
    // across the lowest y tick; a date on the right edge ends there instead of clipping.
    const anchor = i === 0 && xOf(date) - area.x0 < 12 ? "start" : area.x1 - xOf(date) < 12 ? "end" : "middle";
    body += `<text x="${xOf(date).toFixed(1)}" y="${area.y1 + 12}" class="tick" text-anchor="${anchor}">${date.slice(5)}</text>`;
  }

  for (const line of usable) {
    const band = ciBandPolygon(line, xOf, yOf);
    if (band) body += `<polygon points="${band}" fill="${line.color}" fill-opacity="0.16" stroke="none"/>`;
    const points = line.dates
      .map((date, index) => ({ date, value: line.rmsd[index] }))
      .filter((point) => Number.isFinite(point.value));
    const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${xOf(point.date).toFixed(1)} ${yOf(point.value).toFixed(1)}`);
    if (points.length) ends.push({ label: line.label, color: line.color, y: yOf(points.at(-1).value) });
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

  const legend = usable.length > 1 ? renderGutterLabels(area, ends) : "";

  return svgOpen(title, unit) + axes(area, "start date") + body + legend + interactionLayer() + "</svg>";
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
  { title = "RMSE vs depth", unit = "", emptyMessage = "no depth profile for this variable", xBound = 0, typeScale = 1 } = {},
) {
  const usable = (series || []).filter((line) => line && Array.isArray(line.bins) && line.bins.some((bin) => Number.isFinite(bin.rmsd)));
  if (!usable.length) return emptyChart(title, emptyMessage);
  // Depth ordering (surface→deep) from the series with the most bins, so a shorter
  // profile still aligns onto the shared axis by label.
  const depthLabels = usable.reduce((best, line) => (line.bins.length > best.length ? line.bins : best), usable[0].bins).map((bin) => bin.label);
  // Depth-bin labels ("1500-3000 m") are wider than the numeric ticks the other charts use,
  // so this profile gets a gutter sized to its longest label at the rendered type size.
  const DEPTH_PAD_LEFT = labelGutter(depthLabels, typeScale, 44);
  const base = plotArea(0, gutterWidth());
  const area = { ...base, x0: DEPTH_PAD_LEFT, width: base.x1 - DEPTH_PAD_LEFT };
  const ends = [];

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
    // End ticks align inward, so the last one stays inside the chart and the first one
    // clears the deepest bin's label.
    const anchor = t === 4 ? "end" : t === 0 ? "start" : "middle";
    body += `<text x="${x.toFixed(1)}" y="${area.y1 + 12}" class="tick" text-anchor="${anchor}">${formatTick(value)}</text>`;
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
    if (points.length) ends.push({ label: line.label, color: line.color, y: Math.min(...points.map((point) => yOf(point.row))) });
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

  const legend = usable.length > 1 ? renderGutterLabels(area, ends) : "";

  // Depth axis reads downward; the tooltip's crosshair (a vertical line) is not meaningful
  // for a profile, so only the point readout is used (interactionLayer supplies the tooltip).
  return svgOpen(title, unit) + axes(area, unit ? `RMSE (${unit})` : "RMSE") + body + legend + interactionLayer() + "</svg>";
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
  { title = "Water column", unit = "", xLabel = "value", emptyMessage = "no water column at this point", valueBound = null, depthBound = 0, typeScale = 1 } = {},
) {
  const usable = (series || []).filter(
    (line) => line && Array.isArray(line.points) && line.points.some((point) => Number.isFinite(point.value) && Number.isFinite(point.depth)),
  );
  if (!usable.length) return emptyChart(title, emptyMessage);

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
  const depthTicks = [0, 1, 2, 3, 4].map((t) => Math.round((depthMax * t) / 4).toLocaleString("en-US"));
  const DEPTH_PAD_LEFT = labelGutter(depthTicks, typeScale, 36);
  const base = plotArea(0, gutterWidth());
  const area = { ...base, x0: DEPTH_PAD_LEFT, width: base.x1 - DEPTH_PAD_LEFT };
  const ends = [];
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
    // The end ticks sit on the plot edges, so they are aligned inward instead of
    // centring half a label outside the chart or across the corner's y tick.
    const anchor = t === 4 ? "end" : t === 0 ? "start" : "middle";
    body += `<text x="${x.toFixed(1)}" y="${area.y1 + 12}" class="tick" text-anchor="${anchor}">${formatTick(value)}</text>`;
  }
  for (let t = 0; t <= 4; t += 1) {
    const depth = (depthMax * t) / 4;
    const y = yOf(depth);
    body += `<line x1="${area.x0}" y1="${y.toFixed(1)}" x2="${area.x1}" y2="${y.toFixed(1)}" class="grid"/>`;
    body += `<text x="${area.x0 - 4}" y="${(y + 3).toFixed(1)}" class="tick" text-anchor="end">${depthTicks[t]}</text>`;
  }

  for (const line of usable) {
    const points = line.points
      .filter((point) => Number.isFinite(point.value) && Number.isFinite(point.depth))
      .slice()
      .sort((a, b) => a.depth - b.depth);
    const path = points.map((point, index) => `${index === 0 ? "M" : "L"}${xOf(point.value).toFixed(1)} ${yOf(point.depth).toFixed(1)}`);
    if (points.length) ends.push({ label: line.label, color: line.color, y: yOf(points.at(-1).depth) });
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

  const legend = usable.length > 1 ? renderGutterLabels(area, ends) : "";

  return svgOpen(title, "depth in m") + axes(area, unit ? `${xLabel} (${unit})` : xLabel) + body + legend + interactionLayer() + "</svg>";
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
// quarters of a "nice" maximum (1, 1.2, 1.6, 2, 2.4, 3.2, 4, 6, 8 times a power of ten), and every
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
