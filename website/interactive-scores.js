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

const PALETTES = {
  "Teal-Orange":  { ends: [[0, 128, 128],  [230, 97, 1]],   light: [240, 240, 240], dark: [45, 45, 45] },
  "Blue-Red":     { ends: [[0, 0, 255],    [255, 0, 0]],     light: [255, 255, 255], dark: [40, 40, 40] },
  "Green-Red":    { ends: [[26, 152, 80],  [215, 48, 39]],   light: [254, 224, 79],  dark: [80, 70, 20] },
  "Purple-Orange":{ ends: [[94, 60, 153],  [230, 97, 1]],    light: [247, 247, 247], dark: [45, 45, 45] },
  "Green-Purple": { ends: [[27, 120, 55],  [118, 42, 131]],  light: [247, 247, 247], dark: [45, 45, 45] },
};

function isDarkMode() {
  return document.body.classList.contains("quarto-dark");
}

function getPaletteColors(paletteName) {
  const palette = PALETTES[paletteName];
  const neutralColor = isDarkMode() ? palette.dark : palette.light;
  return [palette.ends[0], neutralColor, palette.ends[1]];
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

const activeDepthTab = {};

function interpolateColor(startColor, endColor, ratio) {
  return [
    Math.round(startColor[0] + (endColor[0] - startColor[0]) * ratio),
    Math.round(startColor[1] + (endColor[1] - startColor[1]) * ratio),
    Math.round(startColor[2] + (endColor[2] - startColor[2]) * ratio),
  ];
}

function getActivePaletteColors() {
  const colors = getPaletteColors(currentPalette);
  return paletteReversed ? [...colors].reverse() : colors;
}

function textColorForBackground(rgb) {
  const luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2];
  return luminance > 140 ? "black" : "white";
}

function getCellStyle(referenceValue, comparedValue) {
  const palette = getActivePaletteColors();
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

function formatVariableHeader(variable, unit, cfName) {
  let header = variable;
  if (unit && !variable.includes(`(${unit})`)) {
    header = `${variable} (${unit})`;
  }
  if (cfName && cfName !== "unknown") {
    header += `<br><span class="cf-name">${cfName}</span>`;
  }
  return header;
}

function cellTooltip(variable, unit, day, value, referenceValue, isBaseline, baselineName) {
  const unitSuffix = unit ? ` ${unit}` : "";
  let tooltip = `${variable}, lead day ${day}\nValue: ${value.toFixed(2)}${unitSuffix}`;
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
        const display = value !== null ? value.toFixed(2) : "NaN";
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
          const display = value !== null ? value.toFixed(2) : "NaN";
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

function renderTabbedDepthMetric(
  challengers,
  challengerNames,
  metricKey,
  baseline,
  idPrefix,
) {
  const baselineScore = challengers[baseline][metricKey];
  if (!baselineScore) return "";

  const depths = Object.keys(baselineScore.depths);
  const orderedNames = [
    baseline,
    ...challengerNames.filter(
      (name) => name !== baseline && challengers[name][metricKey],
    ),
  ];

  const savedDepth = activeDepthTab[idPrefix];
  const activeIndex = savedDepth ? Math.max(depths.indexOf(savedDepth), 0) : 0;

  let tabsHtml = `<div class="depth-tabs" role="tablist">`;
  for (let i = 0; i < depths.length; i++) {
    const depth = depths[i];
    const active = i === activeIndex ? " active" : "";
    const selected = i === activeIndex ? "true" : "false";
    tabsHtml += `<button class="depth-tab${active}" role="tab" aria-selected="${selected}" data-depth="${depth}" data-prefix="${idPrefix}">${depth}</button>`;
  }
  tabsHtml += `</div>`;

  const variableSets = depths.map((depth) => new Set(Object.keys(baselineScore.depths[depth].variables)));
  const commonVariables = [];
  const allSeenVariables = new Set();
  for (const depth of depths) {
    for (const variable of Object.keys(baselineScore.depths[depth].variables)) {
      if (!allSeenVariables.has(variable) && variableSets.every((variableSet) => variableSet.has(variable))) {
        commonVariables.push(variable);
      }
      allSeenVariables.add(variable);
    }
  }
  const commonVariableSet = new Set(commonVariables);

  let panelsHtml = "";
  for (let i = 0; i < depths.length; i++) {
    const depth = depths[i];
    const depthVariables = Object.keys(baselineScore.depths[depth].variables);
    const depthSpecificVariables = depthVariables.filter((variable) => !commonVariableSet.has(variable));
    const orderedVariables = [...commonVariables.filter((variable) => depthVariables.includes(variable)), ...depthSpecificVariables];
    if (orderedVariables.length === 0) continue;
    const leadDays = getLeadDays(baselineScore, depth);
    const hiddenAttribute = i === activeIndex ? "" : " hidden";

    let thead = "<thead>";
    thead += `<tr><th class="model-col">Models</th>`;
    for (const variable of orderedVariables) {
      const unit = getUnit(baselineScore, depth, variable);
      const cfName = getCfName(baselineScore, depth, variable);
      thead += `<th class="var-header" colspan="${leadDays.length}">${formatVariableHeader(variable, unit, cfName)}</th>`;
    }
    thead += `</tr><tr><th class="model-col lead-day-label">Lead days</th>`;
    for (const variable of orderedVariables) {
      for (const day of leadDays) {
        thead += `<th class="lead-day">${day}</th>`;
      }
    }
    thead += "</tr></thead>";

    const tbody =
      "<tbody>" +
      buildDataRows(
        orderedNames,
        challengers,
        metricKey,
        depth,
        orderedVariables,
        leadDays,
        baseline,
      ) +
      "</tbody>";

    panelsHtml += `<div class="depth-panel"${hiddenAttribute} role="tabpanel" data-depth="${depth}" data-prefix="${idPrefix}">`;
    panelsHtml += `<div class="score-table-wrapper"><table class="score-table">${thead}${tbody}</table></div>`;
    panelsHtml += `</div>`;
  }

  return tabsHtml + panelsHtml;
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

  const idPrefix = containerId;
  let html = "";

  html += `<h3>${METRIC_TITLES[depthMetric]}</h3>`;
  html += renderTabbedDepthMetric(
    challengers,
    challengerNames,
    depthMetric,
    baseline,
    idPrefix,
  );

  html += `<h3>Diagnostic Metrics</h3>`;
  html += renderCombinedFlatMetrics(
    challengers,
    challengerNames,
    flatMetrics,
    baseline,
  );

  container.innerHTML = html;

  container.querySelectorAll(".depth-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const prefix = tab.dataset.prefix;
      const depth = tab.dataset.depth;

      activeDepthTab[prefix] = depth;

      container.querySelectorAll(`.depth-tab[data-prefix="${prefix}"]`).forEach((tabButton) => {
        tabButton.classList.remove("active");
        tabButton.setAttribute("aria-selected", "false");
      });
      tab.classList.add("active");
      tab.setAttribute("aria-selected", "true");

      container.querySelectorAll(`.depth-panel[data-prefix="${prefix}"]`).forEach((panel) => {
        panel.hidden = panel.dataset.depth !== depth;
      });
    });
  });
}

let currentPalette = "Teal-Orange";
let paletteReversed = false;

function formatRgb(color) {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

function activeGradientCSS() {
  const colors = getActivePaletteColors();
  const stops = colors.map(
    (color, index) => `${formatRgb(color)} ${(index / (colors.length - 1)) * 100}%`,
  );
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

function paletteGradientCSS(paletteName, reversed) {
  const paletteColors = getPaletteColors(paletteName);
  const colors = reversed ? [...paletteColors].reverse() : paletteColors;
  const stops = colors.map(
    (color, index) => `${formatRgb(color)} ${(index / (colors.length - 1)) * 100}%`,
  );
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

function buildPaletteButtons() {
  const container = document.getElementById("palette-buttons");
  if (!container) return;
  container.innerHTML = "";
  for (const name of Object.keys(PALETTES)) {
    const button = document.createElement("button");
    const isActive = name === currentPalette;
    const showReversed = isActive && paletteReversed;
    button.className = "palette-btn" + (isActive ? " active" : "");
    button.style.background = paletteGradientCSS(name, showReversed);
    button.title = name + (showReversed ? " (reversed)" : "");
    if (showReversed) {
      button.textContent = "\u21C4";
    }
    button.addEventListener("click", () => {
      if (currentPalette === name) {
        paletteReversed = !paletteReversed;
      } else {
        currentPalette = name;
        paletteReversed = false;
      }
      renderAllTables();
    });
    container.appendChild(button);
  }
}

function updateColorLegend(baseline) {
  const legend = document.getElementById("color-legend");
  if (!legend) return;
  legend.innerHTML =
    `<span class="legend-label">Worse</span>` +
    `<span class="legend-bar" style="background: ${activeGradientCSS()}"></span>` +
    `<span class="legend-label">Better</span>` +
    `<span class="legend-label">(vs. ${baseline})</span>`;
}

function populateBaselineSelect(challengerNames) {
  const select = document.getElementById("baseline-select");
  if (!select || select.options.length > 0) return;
  for (const name of challengerNames) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = name;
    if (name === "glo12") option.selected = true;
    select.appendChild(option);
  }
}

function renderAllTables() {
  const dataElement = document.getElementById("scores-data");
  if (!dataElement) return;
  const data = JSON.parse(dataElement.textContent);
  const { challengers, challenger_names: challengerNames } = data;

  populateBaselineSelect(challengerNames);
  buildPaletteButtons();

  const baseline =
    document.getElementById("baseline-select").value || challengerNames[0];

  updateColorLegend(baseline);

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
}

function init() {
  if (!document.getElementById("scores-data")) return;
  renderAllTables();
  const baselineSelect = document.getElementById("baseline-select");
  if (baselineSelect)
    baselineSelect.addEventListener("change", renderAllTables);
  new MutationObserver(() => renderAllTables()).observe(document.body, {
    attributes: true,
    attributeFilter: ["class"],
  });
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
