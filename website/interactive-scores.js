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
  // ColorBrewer RdBu diverging — perceptually uniform, colorblind-safe
  blue: { end: [33, 102, 172], mid: [146, 197, 222] },
  red:  { mid: [244, 165, 130], end: [178, 24, 43] },
  light: [255, 255, 255],
  dark: [40, 40, 40],
};

function isDarkMode() {
  return document.body.classList.contains("quarto-dark");
}

function getPaletteColors() {
  const neutral = isDarkMode() ? PALETTE.dark : PALETTE.light;
  return [PALETTE.blue.end, PALETTE.blue.mid, neutral, PALETTE.red.mid, PALETTE.red.end];
}

let selectedDepths = new Set();
let availableDepths = [];
let showAllMode = true;
let showPercentDiff = false;
let parsedData = null;

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
  return `${sign}${Math.round(percent)}%`;
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

function titleCase(text) {
  return text.replace(/(^|\s)\w/g, (character) => character.toUpperCase());
}

function formatVariableHeader(variable, unit, cfName, metricKey) {
  const displayName = titleCase(variable);
  const metricLabel = metricKey.startsWith("rmsd") ? `RMSE (${unit})` : `(${unit})`;
  let header = `${displayName}<br><span class="metric-label">${metricLabel}</span>`;
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
      const referenceValues = {};
      for (const day of leadDays) {
        referenceValues[day] = getValue(baselineScore, depth, variable, day);
      }
      for (const day of leadDays) {
        const value = getValue(score, depth, variable, day);
        const referenceValue = referenceValues[day];
        let style = "";
        if (!isBaseline && value !== null && referenceValue !== null) {
          style = getCellStyle(referenceValue, value);
        }
        let display = "";
        if (value !== null) {
          if (showPercentDiff && !isBaseline && referenceValue !== null) {
            display = formatPercentDiff(referenceValue, value);
          } else {
            display = value.toFixed(2);
          }
        }
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
          let display = "";
          if (value !== null) {
            if (showPercentDiff && !isBaseline && referenceValue !== null) {
              display = formatPercentDiff(referenceValue, value);
            } else {
              display = value.toFixed(2);
            }
          }
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

  html += '<span class="display-toggle">';
  html += `<button class="display-toggle-btn${!showPercentDiff ? " active" : ""}" data-display="values">Values</button>`;
  html += `<button class="display-toggle-btn${showPercentDiff ? " active" : ""}" data-display="percent-diff">% diff</button>`;
  html += '</span>';

  html += '<div id="color-legend" class="color-legend"></div>';

  return html;
}

function ensureControlsElement() {
  let element = document.getElementById("score-controls");
  if (element) return element;

  // Quarto freeze cache can leave a stale controls div in the rendered HTML
  const wrapper = document.getElementById("all-scores");
  if (wrapper) {
    const stale = wrapper.querySelector(".controls:not(#score-controls)");
    if (stale) stale.remove();
  }

  element = document.createElement("div");
  element.id = "score-controls";
  element.className = "controls";

  if (wrapper) {
    wrapper.insertBefore(element, wrapper.firstElementChild);
  } else {
    const reanalysis = document.getElementById("reanalysis-scores");
    if (reanalysis) {
      reanalysis.parentNode.insertBefore(element, reanalysis);
    }
  }

  return element;
}

function groupDepthsByVariables(baselineScore, depths) {
  const groups = [];
  for (const depth of depths) {
    const variables = Object.keys(baselineScore.depths[depth]?.variables || {});
    const signature = variables.join(",");
    const last = groups[groups.length - 1];
    if (last && last.signature === signature) {
      last.depths.push(depth);
    } else {
      groups.push({ signature, depths: [depth], variables });
    }
  }
  return groups;
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
  const filteredDepths = showAllMode ? depths : depths.filter((depth) => selectedDepths.has(depth));
  const visibleDepths = filteredDepths.length > 0 ? filteredDepths : depths;
  const orderedNames = [
    baseline,
    ...challengerNames.filter(
      (name) => name !== baseline && challengers[name][metricKey],
    ),
  ];

  const depthGroups = groupDepthsByVariables(baselineScore, visibleDepths);
  if (depthGroups.length > 1) {
    return depthGroups.map((group) => renderDepthGroup(
      baselineScore, orderedNames, challengers, metricKey, group.depths, group.variables, baseline,
    )).join("");
  }

  const headerDepth = depths[0];
  const allHeaderVariables = Object.keys(baselineScore.depths[headerDepth].variables);
  const deeperVariableSets = depths.slice(1).map(
    (depth) => new Set(Object.keys(baselineScore.depths[depth]?.variables || {})),
  );
  const surfaceOnlyVariables = allHeaderVariables.filter(
    (variable) => deeperVariableSets.length > 0 && deeperVariableSets.some((set) => !set.has(variable)),
  );
  const commonVariables = allHeaderVariables.filter((variable) => !surfaceOnlyVariables.includes(variable));
  const hasDeepDepth = visibleDepths.some((depth) => depth !== depths[0]);
  const variables = hasDeepDepth ? commonVariables : [...commonVariables, ...surfaceOnlyVariables];

  return renderDepthGroup(
    baselineScore, orderedNames, challengers, metricKey, visibleDepths, variables, baseline,
  );
}

function renderDepthGroup(
  baselineScore, orderedNames, challengers, metricKey, depths, variables, baseline,
) {
  if (variables.length === 0 || depths.length === 0) return "";

  const refDepth = depths[0];
  const leadDays = getLeadDays(baselineScore, refDepth);
  const totalColumns = 1 + variables.length * leadDays.length;

  let thead = "<thead>";
  thead += `<tr><th class="model-col">Models</th>`;
  for (const variable of variables) {
    const unit = getUnit(baselineScore, refDepth, variable);
    const cfName = getCfName(baselineScore, refDepth, variable);
    thead += `<th class="var-header" colspan="${leadDays.length}">${formatVariableHeader(variable, unit, cfName, metricKey)}</th>`;
  }
  thead += `</tr><tr><th class="model-col lead-day-label">Lead days</th>`;
  for (const variable of variables) {
    for (const day of leadDays) {
      thead += `<th class="lead-day">${day}</th>`;
    }
  }
  thead += "</tr></thead>";

  let tbody = "<tbody>";
  for (const depth of depths) {
    if (depths.length > 1) {
      tbody += `<tr class="depth-separator"><td class="depth-separator-cell" colspan="${totalColumns}">${depth}</td></tr>`;
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

  const tableClass = depths.length > 1 ? "score-table depth-table" : "score-table";
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
      thead += `<th class="var-header" colspan="${leadDays.length}">${formatVariableHeader(variable, unit, cfName, metricKey)}</th>`;
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
  metricTitles,
) {
  const container = document.getElementById(containerId);
  if (!container) return;

  let html = "";

  html += '<div class="depth-section">';
  html += `<h3>${metricTitles[depthMetric]}</h3>`;
  html += renderDepthMetric(
    challengers,
    challengerNames,
    depthMetric,
    baseline,
  );
  html += "</div>";

  const flatHtml = renderCombinedFlatMetrics(
    challengers,
    challengerNames,
    flatMetrics,
    baseline,
  );
  if (flatHtml) {
    html += `<h3>Diagnostic Metrics</h3>`;
    html += flatHtml;
  }

  container.innerHTML = html;
}

function formatRgb(color) {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

function legendGradientCSS() {
  // Reverse: palette is ordered better→worse, legend displays worse→better (left→right)
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
    table.addEventListener("mouseover", (event) => {
      const cell = event.target.closest("td");
      if (!cell || cell.classList.contains("depth-separator-cell")) return;
      cell.classList.add("highlight-cell");
      const modelCell = cell.parentElement.querySelector("th.model-col");
      if (modelCell) modelCell.classList.add("highlight-label");
      const columnIndex = cell.cellIndex;
      const lastHeadRow = table.querySelector("thead tr:last-child");
      if (lastHeadRow && lastHeadRow.cells[columnIndex]) {
        lastHeadRow.cells[columnIndex].classList.add("highlight-label");
      }
    });
    table.addEventListener("mouseout", (event) => {
      const cell = event.target.closest("td");
      if (!cell) return;
      table.querySelectorAll(".highlight-cell, .highlight-label").forEach((highlighted) => {
        highlighted.classList.remove("highlight-cell", "highlight-label");
      });
    });
  });
}

function updateStickyOffsets() {
  const controls = document.getElementById("score-controls");
  if (controls) {
    const controlsHeight = controls.getBoundingClientRect().height;
    document.documentElement.style.setProperty("--controls-height", controlsHeight + "px");
  }
}

function attachControlListeners() {
  const baselineSelect = document.getElementById("baseline-select");
  if (baselineSelect) {
    baselineSelect.addEventListener("change", renderAllTables);
  }

  document.querySelectorAll(".display-toggle-btn").forEach((button) => {
    button.addEventListener("click", () => {
      showPercentDiff = button.dataset.display === "percent-diff";
      renderAllTables();
    });
  });

  const allButton = document.querySelector('.depth-toggle-btn[data-depth="all"]');
  if (allButton) {
    allButton.addEventListener("click", () => {
      if (showAllMode) {
        showAllMode = false;
        selectedDepths = new Set();
      } else {
        showAllMode = true;
      }
      renderAllTables();
    });
  }

  // Click-hold-drag to select/deselect a range of depth pills
  let dragAction = null;

  function applyDragToButton(button) {
    const depth = button.dataset.depth;
    if (!depth || depth === "all") return;
    if (dragAction === "select") {
      if (showAllMode) {
        showAllMode = false;
        selectedDepths = new Set([depth]);
      } else {
        selectedDepths.add(depth);
      }
      button.classList.add("active");
    } else {
      if (!showAllMode) {
        selectedDepths.delete(depth);
        button.classList.remove("active");
      }
    }
    if (allButton) allButton.classList.toggle("active", showAllMode);
  }

  document.querySelectorAll(".depth-pills").forEach((container) => {
    container.addEventListener("mousedown", (event) => {
      const button = event.target.closest(".depth-toggle-btn");
      if (!button) return;
      event.preventDefault();
      const depth = button.dataset.depth;
      const isActive = !showAllMode && selectedDepths.has(depth);
      dragAction = isActive ? "deselect" : "select";
      applyDragToButton(button);
      document.addEventListener("mouseup", () => {
        dragAction = null;
        renderAllTables();
      }, { once: true });
    });
    container.addEventListener("mouseover", (event) => {
      if (!dragAction) return;
      const button = event.target.closest(".depth-toggle-btn");
      if (!button) return;
      applyDragToButton(button);
    });
  });
}

function renderAllTables() {
  if (!parsedData) {
    const dataElement = document.getElementById("scores-data");
    if (!dataElement) return;
    parsedData = JSON.parse(dataElement.textContent);
  }
  const {
    challengers,
    challenger_names: challengerNames,
    metric_titles: metricTitles,
    sections,
  } = parsedData;

  const existingSelect = document.getElementById("baseline-select");
  const baseline = existingSelect?.value
    || (challengerNames.includes("glo12") ? "glo12" : challengerNames[0]);

  for (const [sectionKey, sectionConfig] of Object.entries(sections)) {
    const depthScore = challengers[baseline]?.[sectionConfig.depth_metric];
    if (depthScore && availableDepths.length === 0) {
      availableDepths = Object.keys(depthScore.depths);
    }
    renderMetricSection(
      `${sectionKey}-scores`,
      sectionConfig.depth_metric,
      sectionConfig.flat_metrics,
      challengers,
      challengerNames,
      baseline,
      metricTitles,
    );
  }

  const controlsElement = ensureControlsElement();
  controlsElement.innerHTML = buildControlsInnerHtml(challengerNames, baseline, availableDepths);

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
    const navbarHeight = navbar.offsetHeight;
    document.documentElement.style.setProperty("--navbar-full-height", navbarHeight + "px");
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
