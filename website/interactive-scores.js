// SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

const MAX_PERCENT_DIFF = 50;
const DISPLAY_LEAD_DAYS = [
  { preferred: "1", fallback: "2" },
  { preferred: "3" },
  { preferred: "5" },
  { preferred: "7" },
  { preferred: "10", fallback: "9" },
];

const PALETTE = {
  ends: [[0, 0, 255], [255, 0, 0]],
  light: [255, 255, 255],
  dark: [40, 40, 40],
};

function isDarkMode() {
  return document.body.classList.contains("quarto-dark");
}

function getPaletteColors() {
  const neutralColor = isDarkMode() ? PALETTE.dark : PALETTE.light;
  return [PALETTE.ends[0], neutralColor, PALETTE.ends[1]];
}

const METRIC_TITLES = {
  rmsd_variables_glorys: "RMSD of Variables",
  rmsd_mld_glorys: "RMSD of Mixed Layer Depth",
  rmsd_geostrophic_glorys: "RMSD of Geostrophic Currents",
  lagrangian_glorys: "Lagrangian Trajectory Deviation",
  rmsd_variables_glo12: "RMSD of Variables",
  rmsd_mld_glo12: "RMSD of Mixed Layer Depth",
  rmsd_geostrophic_glo12: "RMSD of Geostrophic Currents",
  lagrangian_glo12: "Lagrangian Trajectory Deviation",
};

const REANALYSIS_DEPTH_METRIC = "rmsd_variables_glorys";
const ANALYSIS_DEPTH_METRIC = "rmsd_variables_glo12";

const REANALYSIS_FLAT_METRICS = [
  "rmsd_mld_glorys",
  "rmsd_geostrophic_glorys",
  "lagrangian_glorys",
];

const ANALYSIS_FLAT_METRICS = [
  "rmsd_mld_glo12",
  "rmsd_geostrophic_glo12",
  "lagrangian_glo12",
];

let selectedDepths = new Set();
let availableDepths = [];
let showAllMode = true;

function interpolateColor(startColor, endColor, ratio) {
  return [
    Math.round(startColor[0] + (endColor[0] - startColor[0]) * ratio),
    Math.round(startColor[1] + (endColor[1] - startColor[1]) * ratio),
    Math.round(startColor[2] + (endColor[2] - startColor[2]) * ratio),
  ];
}

function textColorForBackground(rgb) {
  const luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
  return luminance > 140 ? "black" : "white";
}

function getCellStyle(referenceValue, comparedValue) {
  const palette = getPaletteColors();
  const percentDiff = referenceValue === 0 ? 0 : ((comparedValue - referenceValue) / Math.abs(referenceValue)) * 100;
  const clamped = Math.max(-MAX_PERCENT_DIFF, Math.min(MAX_PERCENT_DIFF, percentDiff));
  const normalized = (clamped + MAX_PERCENT_DIFF) / (MAX_PERCENT_DIFF * 2);
  const segmentCount = palette.length - 1;
  const segment = Math.min(Math.floor(normalized * segmentCount), segmentCount - 1);
  const localRatio = normalized * segmentCount - segment;
  const color = interpolateColor(palette[segment], palette[segment + 1], localRatio);
  return `background-color:rgb(${color[0]}, ${color[1]}, ${color[2]}); color: ${textColorForBackground(color)}`;
}

function formatPercentDiff(referenceValue, comparedValue) {
  if (referenceValue === 0) return comparedValue === 0 ? "0%" : "N/A";
  const percent = ((comparedValue - referenceValue) / Math.abs(referenceValue)) * 100;
  const sign = percent > 0 ? "+" : "";
  return `${sign}${percent.toFixed(1)}%`;
}

function getValue(scoreData, depth, variable, leadDay) {
  try {
    return scoreData.depths[depth].variables[variable].data[leadDay];
  } catch {
    return null;
  }
}

function getLeadDays(scoreData, depth) {
  const firstVariable = Object.keys(scoreData.depths[depth].variables)[0];
  const allDays = Object.keys(scoreData.depths[depth].variables[firstVariable].data);
  const resolvedDays = [];
  for (const entry of DISPLAY_LEAD_DAYS) {
    if (allDays.includes(entry.preferred)) {
      resolvedDays.push(entry.preferred);
    } else if (entry.fallback && allDays.includes(entry.fallback)) {
      resolvedDays.push(entry.fallback);
    }
  }
  return resolvedDays;
}

function getUnit(scoreData, depth, variable) {
  try {
    return scoreData.depths[depth].variables[variable].unit || "";
  } catch {
    return "";
  }
}

function getCfName(scoreData, depth, variable) {
  try {
    return scoreData.depths[depth].variables[variable].cf_name || "";
  } catch {
    return "";
  }
}

function titleCase(str) {
  return str.replace(/(^|\s)\w/g, (c) => c.toUpperCase());
}

function formatVariableHeader(variable, unit, cfName) {
  const displayName = titleCase(variable);
  let header = displayName;
  if (unit && !variable.includes(`(${unit})`)) {
    header = `${displayName} (${unit})`;
  }
  if (cfName && cfName !== "unknown") {
    header += `<br><span class="cf-name">${cfName}</span>`;
  }
  return header;
}

function cellTooltip(variable, unit, day, value, referenceValue, isBaseline, baselineName) {
  const unitSuffix = unit ? ` ${unit}` : "";
  let tooltip = `${titleCase(variable)}, lead day ${day}\nValue: ${value.toFixed(2)}${unitSuffix}`;
  if (!isBaseline && referenceValue !== null) {
    tooltip += `\nvs ${baselineName}: ${formatPercentDiff(referenceValue, value)}`;
  }
  return tooltip;
}

function buildDataRows(
  orderedNames,
  challengers,
  metricKey,
  depth,
  variables,
  leadDays,
  baseline,
) {
  const baselineScore = challengers[baseline][metricKey];
  let rows = "";
  for (const name of orderedNames) {
    const score = challengers[name][metricKey];
    if (!score || !score.depths[depth]) continue;
    const isBaseline = name === baseline;
    const rowClass = isBaseline ? ' class="baseline-row"' : "";
    rows += `<tr${rowClass}><th class="model-col"><a href="reports/${name}.report.html">${name}</a></th>`;
    for (const variable of variables) {
      const unit = getUnit(baselineScore, depth, variable);
      for (const day of leadDays) {
        const value = getValue(score, depth, variable, day);
        const referenceValue = getValue(baselineScore, depth, variable, day);
        let style = "";
        if (!isBaseline && value !== null && referenceValue !== null) {
          style = getCellStyle(referenceValue, value);
        }
        const display = value !== null ? value.toFixed(2) : "";
        const title = value !== null
          ? cellTooltip(variable, unit, day, value, referenceValue, isBaseline, baseline)
          : "";
        rows += `<td style="${style}" title="${title}">${display}</td>`;
      }
    }
    rows += "</tr>";
  }
  return rows;
}

function buildCombinedDataRows(
  orderedNames,
  challengers,
  metricSpecs,
  baseline,
) {
  let rows = "";
  for (const name of orderedNames) {
    const isBaseline = name === baseline;
    const rowClass = isBaseline ? ' class="baseline-row"' : "";
    rows += `<tr${rowClass}><th class="model-col"><a href="reports/${name}.report.html">${name}</a></th>`;
    for (const { metricKey, variables, leadDays } of metricSpecs) {
      const score = challengers[name][metricKey];
      const baselineScore = challengers[baseline][metricKey];
      for (const variable of variables) {
        const unit = baselineScore ? getUnit(baselineScore, "flat", variable) : "";
        for (const day of leadDays) {
          const value = score ? getValue(score, "flat", variable, day) : null;
          const referenceValue = baselineScore
            ? getValue(baselineScore, "flat", variable, day)
            : null;
          let style = "";
          if (!isBaseline && value !== null && referenceValue !== null) {
            style = getCellStyle(referenceValue, value);
          }
          const display = value !== null ? value.toFixed(2) : "";
          const title = value !== null
            ? cellTooltip(variable, unit, day, value, referenceValue, isBaseline, baseline)
            : "";
          rows += `<td style="${style}" title="${title}">${display}</td>`;
        }
      }
    }
    rows += "</tr>";
  }
  return rows;
}

function buildControlsInnerHtml(challengerNames, baseline, depths) {
  let html = "";

  html += '<label>Baseline: <select id="baseline-select">';
  for (const name of challengerNames) {
    const selected = name === baseline ? " selected" : "";
    html += `<option value="${name}"${selected}>${name}</option>`;
  }
  html += "</select></label>";

  html += '<span class="depth-toggle">';
  html += `<button class="depth-toggle-btn${showAllMode ? " active" : ""}" data-depth="all">All</button>`;
  html += '<span class="depth-pills">';
  for (const depth of depths) {
    const active = !showAllMode && selectedDepths.has(depth) ? " active" : "";
    html += `<button class="depth-toggle-btn${active}" data-depth="${depth}">${depth}</button>`;
  }
  html += "</span></span>";

  html += '<div id="color-legend" class="color-legend"></div>';

  return html;
}

function ensureControlsElement() {
  let el = document.getElementById("score-controls");
  if (el) return el;

  // Remove any stale controls div from server-rendered HTML (freeze cache)
  const wrapper = document.getElementById("all-scores");
  if (wrapper) {
    const stale = wrapper.querySelector(".controls:not(#score-controls)");
    if (stale) stale.remove();
  }

  el = document.createElement("div");
  el.id = "score-controls";
  el.className = "controls";

  if (wrapper) {
    wrapper.insertBefore(el, wrapper.firstElementChild);
  } else {
    const reanalysis = document.getElementById("reanalysis-scores");
    if (reanalysis) {
      reanalysis.parentNode.insertBefore(el, reanalysis);
    }
  }

  return el;
}

function renderDepthMetric(
  challengers,
  challengerNames,
  metricKey,
  baseline,
) {
  const baselineScore = challengers[baseline][metricKey];
  if (!baselineScore) return "";

  const depths = Object.keys(baselineScore.depths);
  const visibleDepths = showAllMode ? depths : depths.filter((d) => selectedDepths.has(d));
  if (visibleDepths.length === 0) return "";
  const orderedNames = [
    baseline,
    ...challengerNames.filter(
      (name) => name !== baseline && challengers[name][metricKey],
    ),
  ];

  // Split variables: common (all depths) vs surface-only
  const headerDepth = depths[0];
  const allHeaderVars = Object.keys(baselineScore.depths[headerDepth].variables);
  const deeperVarSets = depths.slice(1).map(
    (d) => new Set(Object.keys(baselineScore.depths[d]?.variables || {})),
  );
  const surfaceOnlyVars = allHeaderVars.filter(
    (v) => deeperVarSets.length > 0 && deeperVarSets.some((s) => !s.has(v)),
  );
  const commonVars = allHeaderVars.filter((v) => !surfaceOnlyVars.includes(v));
  // If only surface-level depths selected, show surface-only vars too
  const hasDeepDepth = visibleDepths.some((d) => d !== depths[0]);
  const variables = hasDeepDepth ? commonVars : [...commonVars, ...surfaceOnlyVars];
  const leadDays = getLeadDays(baselineScore, headerDepth);
  const totalCols = 1 + variables.length * leadDays.length;

  let thead = "<thead>";
  thead += `<tr><th class="model-col">Models</th>`;
  for (const variable of variables) {
    const unit = getUnit(baselineScore, headerDepth, variable);
    const cfName = getCfName(baselineScore, headerDepth, variable);
    thead += `<th class="var-header" colspan="${leadDays.length}">${formatVariableHeader(variable, unit, cfName)}</th>`;
  }
  thead += `</tr><tr><th class="model-col lead-day-label">Lead days</th>`;
  for (const variable of variables) {
    for (const day of leadDays) {
      thead += `<th class="lead-day">${day}</th>`;
    }
  }
  thead += "</tr></thead>";

  let tbody = "<tbody>";
  for (const depth of visibleDepths) {
    if (visibleDepths.length > 1) {
      tbody += `<tr class="depth-separator"><td class="depth-separator-cell" colspan="${totalCols}">${depth}</td></tr>`;
    }
    tbody += buildDataRows(
      orderedNames,
      challengers,
      metricKey,
      depth,
      variables,
      leadDays,
      baseline,
    );
  }
  tbody += "</tbody>";

  const tableClass = visibleDepths.length > 1 ? "score-table depth-table" : "score-table";
  return `<table class="${tableClass}">${thead}${tbody}</table>`;
}

function renderCombinedFlatMetrics(
  challengers,
  challengerNames,
  metricKeys,
  baseline,
) {
  const metricSpecs = [];
  for (const metricKey of metricKeys) {
    const baselineScore = challengers[baseline][metricKey];
    if (!baselineScore || !baselineScore.depths.flat) continue;
    const variables = Object.keys(baselineScore.depths.flat.variables);
    const leadDays = getLeadDays(baselineScore, "flat");
    metricSpecs.push({ metricKey, variables, leadDays });
  }
  if (metricSpecs.length === 0) return "";

  const orderedNames = [
    baseline,
    ...challengerNames.filter((name) => {
      return (
        name !== baseline &&
        metricSpecs.some((spec) => challengers[name][spec.metricKey])
      );
    }),
  ];

  let thead = "<thead>";
  thead += `<tr><th class="model-col">Models</th>`;
  for (const { metricKey, variables, leadDays } of metricSpecs) {
    const baselineScore = challengers[baseline][metricKey];
    for (const variable of variables) {
      const unit = getUnit(baselineScore, "flat", variable);
      const cfName = getCfName(baselineScore, "flat", variable);
      thead += `<th class="var-header" colspan="${leadDays.length}">${formatVariableHeader(variable, unit, cfName)}</th>`;
    }
  }
  thead += "</tr>";

  thead += `<tr><th class="model-col lead-day-label">Lead days</th>`;
  for (const { variables, leadDays } of metricSpecs) {
    for (const variable of variables) {
      for (const day of leadDays) {
        thead += `<th class="lead-day">${day}</th>`;
      }
    }
  }
  thead += "</tr></thead>";

  const tbody =
    "<tbody>" +
    buildCombinedDataRows(orderedNames, challengers, metricSpecs, baseline) +
    "</tbody>";

  return `<div class="score-table-wrapper"><table class="score-table">${thead}${tbody}</table></div>`;
}

function renderMetricSection(
  containerId,
  depthMetric,
  flatMetrics,
  challengers,
  challengerNames,
  baseline,
) {
  const container = document.getElementById(containerId);
  if (!container) return;

  let html = "";

  html += '<div class="depth-section">';
  html += `<h3>${METRIC_TITLES[depthMetric]}</h3>`;
  html += renderDepthMetric(
    challengers,
    challengerNames,
    depthMetric,
    baseline,
  );
  html += "</div>";

  html += `<h3>Diagnostic Metrics</h3>`;
  html += renderCombinedFlatMetrics(
    challengers,
    challengerNames,
    flatMetrics,
    baseline,
  );

  container.innerHTML = html;
}

function formatRgb(color) {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

function legendGradientCSS() {
  const colors = [...getPaletteColors()].reverse();
  const stops = colors.map(
    (color, index) => `${formatRgb(color)} ${(index / (colors.length - 1)) * 100}%`,
  );
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

function updateColorLegend(baseline) {
  const legend = document.getElementById("color-legend");
  if (!legend) return;
  legend.innerHTML =
    `<span class="legend-label">Worse</span>` +
    `<span class="legend-bar" style="background: ${legendGradientCSS()}"></span>` +
    `<span class="legend-label">Better</span>` +
    `<span class="legend-label">(vs. ${baseline})</span>`;
}

function setupCellHighlight() {
  document.querySelectorAll(".score-table").forEach((table) => {
    table.addEventListener("mouseover", (e) => {
      const td = e.target.closest("td");
      if (!td || td.classList.contains("depth-separator-cell")) return;
      td.classList.add("highlight-cell");
      // Highlight model name
      const modelCell = td.parentElement.querySelector("th.model-col");
      if (modelCell) modelCell.classList.add("highlight-label");
      // Highlight lead day header
      const colIdx = td.cellIndex;
      const lastHeadRow = table.querySelector("thead tr:last-child");
      if (lastHeadRow && lastHeadRow.cells[colIdx]) {
        lastHeadRow.cells[colIdx].classList.add("highlight-label");
      }
    });
    table.addEventListener("mouseout", (e) => {
      const td = e.target.closest("td");
      if (!td) return;
      table.querySelectorAll(".highlight-cell, .highlight-label").forEach((c) => {
        c.classList.remove("highlight-cell", "highlight-label");
      });
    });
  });
}

function updateStickyOffsets() {
  const controls = document.getElementById("score-controls");
  if (controls) {
    const h = controls.getBoundingClientRect().height;
    document.documentElement.style.setProperty("--controls-height", h + "px");
  }
}

function attachControlListeners() {
  const baselineSelect = document.getElementById("baseline-select");
  if (baselineSelect) {
    baselineSelect.addEventListener("change", renderAllTables);
  }

  // "All" button — simple click
  const allBtn = document.querySelector('.depth-toggle-btn[data-depth="all"]');
  if (allBtn) {
    allBtn.addEventListener("click", () => {
      if (showAllMode) {
        showAllMode = false;
        selectedDepths = new Set();
      } else {
        showAllMode = true;
      }
      renderAllTables();
    });
  }

  // Depth pills — drag-select (click-hold-drag to select/deselect range)
  let dragAction = null;

  function applyToBtn(btn) {
    const depth = btn.dataset.depth;
    if (!depth || depth === "all") return;
    if (dragAction === "select") {
      if (showAllMode) {
        showAllMode = false;
        selectedDepths = new Set([depth]);
      } else {
        selectedDepths.add(depth);
      }
      btn.classList.add("active");
    } else {
      if (!showAllMode) {
        selectedDepths.delete(depth);
        btn.classList.remove("active");
      }
    }
    if (allBtn) allBtn.classList.toggle("active", showAllMode);
  }

  document.querySelectorAll(".depth-pills").forEach((container) => {
    container.addEventListener("mousedown", (e) => {
      const btn = e.target.closest(".depth-toggle-btn");
      if (!btn) return;
      e.preventDefault();
      const depth = btn.dataset.depth;
      const isActive = !showAllMode && selectedDepths.has(depth);
      dragAction = isActive ? "deselect" : "select";
      applyToBtn(btn);
      document.addEventListener("mouseup", () => {
        dragAction = null;
        renderAllTables();
      }, { once: true });
    });
    container.addEventListener("mouseover", (e) => {
      if (!dragAction) return;
      const btn = e.target.closest(".depth-toggle-btn");
      if (!btn) return;
      applyToBtn(btn);
    });
  });
}

function renderAllTables() {
  const dataElement = document.getElementById("scores-data");
  if (!dataElement) return;
  const data = JSON.parse(dataElement.textContent);
  const { challengers, challenger_names: challengerNames } = data;

  // Preserve baseline across re-renders
  const existingSelect = document.getElementById("baseline-select");
  const baseline = existingSelect?.value
    || (challengerNames.includes("glo12") ? "glo12" : challengerNames[0]);

  // Discover available depths
  const refScore = challengers[baseline]?.[REANALYSIS_DEPTH_METRIC]
    || challengers[baseline]?.[ANALYSIS_DEPTH_METRIC];
  if (refScore) {
    availableDepths = Object.keys(refScore.depths);
  }

  renderMetricSection(
    "reanalysis-scores",
    REANALYSIS_DEPTH_METRIC,
    REANALYSIS_FLAT_METRICS,
    challengers,
    challengerNames,
    baseline,
  );
  renderMetricSection(
    "analysis-scores",
    ANALYSIS_DEPTH_METRIC,
    ANALYSIS_FLAT_METRICS,
    challengers,
    challengerNames,
    baseline,
  );

  const controlsEl = ensureControlsElement();
  controlsEl.innerHTML = buildControlsInnerHtml(challengerNames, baseline, availableDepths);

  updateColorLegend(baseline);
  updateStickyOffsets();
  attachControlListeners();
  setupCellHighlight();
}

function init() {
  if (!document.getElementById("scores-data")) return;
  renderAllTables();
  let wasDark = document.body.classList.contains("quarto-dark");
  new MutationObserver(() => {
    const isDark = document.body.classList.contains("quarto-dark");
    if (isDark !== wasDark) {
      wasDark = isDark;
      renderAllTables();
    }
  }).observe(document.body, {
    attributes: true,
    attributeFilter: ["class"],
  });

  const header = document.getElementById("quarto-header")
    || document.querySelector(".headroom");
  if (header) {
    const navbar = header.querySelector(".navbar") || header;
    const h = navbar.offsetHeight;
    document.documentElement.style.setProperty("--navbar-full-height", h + "px");
    new MutationObserver(() => {
      const hidden = header.classList.contains("headroom--unpinned");
      document.body.classList.toggle("nav-hidden", hidden);
    }).observe(header, { attributes: true, attributeFilter: ["class"] });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
