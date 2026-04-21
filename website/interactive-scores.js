// SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

function buildBinEdges(maximumPercent) {
  const positive = [];
  let value = maximumPercent;
  for (let i = 0; i < 5; i++) {
    positive.push(value);
    value = value / 2;
  }
  positive.reverse();
  const negative = positive.map((v) => -v).reverse();
  return [-Infinity, ...negative, 0, ...positive, Infinity];
}

const SCALE_SNAPS = [5, 10, 20, 40, 80, 100, 200, 500, 1000];
let maxScale = 80;
let colorScaleEdges = buildBinEdges(maxScale);
const DISPLAY_LEAD_DAYS = [
  { preferred: "1", fallback: "2" },
  { preferred: "3" },
  { preferred: "5" },
  { preferred: "7" },
  { preferred: "10", fallback: "9" },
];

const PALETTE = {
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
let showAllMode = true;
let showPercentDiff = false;
let parsedData = null;
let challengerLabels = {};
let regionLabels = {};
let regionMetadata = {};
let activeSection = "observations";
let activeRegion = null;
let isScrollRefreshScheduled = false;

const SECTION_ORDER = ["observations", "reanalysis", "analysis"];

const SECTION_ID_MAP = {
  observations: "comparison-to-observations",
  reanalysis: "comparison-to-reanalysis",
  analysis: "comparison-to-analysis",
};

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

function getBinIndex(percentDiff) {
  const binCount = colorScaleEdges.length - 1;
  for (let i = 0; i < binCount; i++) {
    if (percentDiff <= colorScaleEdges[i + 1]) return i;
  }
  return binCount - 1;
}

function getBinColor(binIndex) {
  const palette = getPaletteColors();
  const binCount = colorScaleEdges.length - 1;
  const normalized = (binIndex + 0.5) / binCount;
  const segmentCount = palette.length - 1;
  const segment = Math.min(Math.floor(normalized * segmentCount), segmentCount - 1);
  const localRatio = normalized * segmentCount - segment;
  return interpolateColor(palette[segment], palette[segment + 1], localRatio);
}

function getCellStyle(referenceValue, comparedValue) {
  const percentDiff = referenceValue === 0 ? 0 : ((comparedValue - referenceValue) / Math.abs(referenceValue)) * 100;
  const binIndex = getBinIndex(percentDiff);
  const color = getBinColor(binIndex);
  return `background-color:rgb(${color[0]}, ${color[1]}, ${color[2]}); color: ${textColorForBackground(color)}`;
}

function formatPercentDiff(referenceValue, comparedValue) {
  if (referenceValue === 0) return comparedValue === 0 ? "0%" : "N/A";
  const percent = ((comparedValue - referenceValue) / Math.abs(referenceValue)) * 100;
  const sign = percent > 0 ? "+" : "";
  return `${sign}${Math.round(percent)}%`;
}

function formatPercentDiffForCell(referenceValue, comparedValue) {
  if (referenceValue === 0) return comparedValue === 0 ? "0%" : "N/A";
  const percent = ((comparedValue - referenceValue) / Math.abs(referenceValue)) * 100;
  if (percent > 999) return ">999%";
  if (percent < -999) return "<-999%";
  return `${Math.round(percent)}%`;
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

function getStandardName(scoreData, depth, variable) {
  try {
    return scoreData.depths[depth].variables[variable].standard_name || "";
  } catch {
    return "";
  }
}

function displayName(name) {
  return challengerLabels[name] || name;
}

function titleCase(text) {
  return text.replace(/(^|\s)\w/g, (character) => character.toUpperCase());
}

function formatVariableHeader(variable, unit, standardName, metricKey) {
  const displayName = titleCase(variable);
  const metricLabel = metricKey.startsWith("rmsd") ? `RMSE (${unit})` : `(${unit})`;
  let header = `${displayName}<br><span class="metric-label">${metricLabel}</span>`;
  if (standardName && standardName !== "unknown") {
    header += `<br><span class="standard-name">${standardName}</span>`;
  }
  return header;
}

function cellTooltip(variable, unit, day, value, referenceValue, isBaseline, baselineName) {
  const unitSuffix = unit ? ` ${unit}` : "";
  let tooltip = `${titleCase(variable)}, lead day ${day}\nValue: ${value.toFixed(2)}${unitSuffix}`;
  if (!isBaseline && referenceValue !== null) {
    tooltip += `\nvs ${displayName(baselineName)}: ${formatPercentDiff(referenceValue, value)}`;
  }
  return tooltip;
}

function buildDataRows(
  orderedNames,
  challengers,
  regionId,
  metricKey,
  depth,
  variables,
  leadDays,
  baseline,
  depthVariables,
) {
  const baselineScore = challengers[baseline][metricKey];
  let rows = "";
  for (const name of orderedNames) {
    const score = challengers[name][metricKey];
    if (!score || !score.depths[depth]) continue;
    const isBaseline = name === baseline;
    const rowClass = isBaseline ? ' class="baseline-row"' : "";
    rows += `<tr${rowClass}><th class="model-col"><a href="reports/${name}.${regionId}.report.html">${displayName(name)}</a></th>`;
    for (const variable of variables) {
      if (depthVariables && !depthVariables.has(variable)) {
        for (const day of leadDays) {
          rows += `<td class="no-var-cell"></td>`;
        }
        continue;
      }
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
            display = formatPercentDiffForCell(referenceValue, value);
          } else {
            display = value.toFixed(2);
          }
        }
        const title = value !== null
          ? cellTooltip(variable, unit, day, value, referenceValue, isBaseline, baseline)
          : "";
        rows += `<td class="score-value-cell" style="${style}" title="${title}">${display}</td>`;
      }
    }
    rows += "</tr>";
  }
  return rows;
}

function buildCombinedDataRows(
  orderedNames,
  challengers,
  regionId,
  metricSpecs,
  baseline,
) {
  let rows = "";
  for (const name of orderedNames) {
    const isBaseline = name === baseline;
    const rowClass = isBaseline ? ' class="baseline-row"' : "";
    rows += `<tr${rowClass}><th class="model-col"><a href="reports/${name}.${regionId}.report.html">${displayName(name)}</a></th>`;
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
              display = formatPercentDiffForCell(referenceValue, value);
            } else {
              display = value.toFixed(2);
            }
          }
          const title = value !== null
            ? cellTooltip(variable, unit, day, value, referenceValue, isBaseline, baseline)
            : "";
          rows += `<td class="score-value-cell" style="${style}" title="${title}">${display}</td>`;
        }
      }
    }
    rows += "</tr>";
  }
  return rows;
}

function readSectionFromHash() {
  const hash = window.location.hash.slice(1);
  if (SECTION_ID_MAP[hash]) return hash;
  for (const [sectionKey, sectionId] of Object.entries(SECTION_ID_MAP)) {
    if (sectionId === hash) return sectionKey;
  }
  return null;
}

function getSectionElement(sectionKey) {
  const sectionId = SECTION_ID_MAP[sectionKey];
  if (!sectionId) return null;
  return document.getElementById(sectionId);
}

function updateSectionHash(sectionKey, replaceHistory) {
  const sectionId = SECTION_ID_MAP[sectionKey];
  if (!sectionId) return;
  const hash = `#${sectionId}`;
  if (window.location.hash === hash) return;
  if (replaceHistory) {
    window.history.replaceState(null, "", hash);
  } else {
    window.history.pushState(null, "", hash);
  }
}

function setActiveSection(sectionKey, options = {}) {
  if (!SECTION_ID_MAP[sectionKey]) return;
  const { updateHash = false, replaceHistory = true } = options;
  activeSection = sectionKey;
  const depthToggle = document.getElementById("depth-toggle");
  if (depthToggle) {
    if (activeSection === "observations") {
      depthToggle.setAttribute("hidden", "");
      depthToggle.setAttribute("aria-hidden", "true");
    } else {
      depthToggle.removeAttribute("hidden");
      depthToggle.removeAttribute("aria-hidden");
    }
  }
  document.querySelectorAll(".score-track-link").forEach((link) => {
    const isActive = link.dataset.section === activeSection;
    link.classList.toggle("active", isActive);
    if (isActive) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  });
  if (updateHash) {
    updateSectionHash(sectionKey, replaceHistory);
  }
}

function getStickyBottomOffset() {
  const scoreHeader = document.getElementById("score-header");
  if (!scoreHeader) return 0;
  return Math.max(scoreHeader.getBoundingClientRect().bottom, 0);
}

function scrollToSection(sectionKey) {
  const target = getSectionElement(sectionKey);
  if (!target) return false;
  const desiredTop = getStickyBottomOffset() + 4;
  const delta = target.getBoundingClientRect().top - desiredTop;
  if (Math.abs(delta) < 2) return false;
  const top = Math.max(0, window.scrollY + delta);
  window.scrollTo({ top, behavior: "auto" });
  return true;
}

function orderedSectionKeys(sections) {
  const availableSections = sections ? new Set(Object.keys(sections)) : null;
  return SECTION_ORDER.filter(
    (sectionKey) => SECTION_ID_MAP[sectionKey] && (!availableSections || availableSections.has(sectionKey)),
  );
}

function navigateToSection(
  sectionKey,
  { replaceHistory = false, updateHash = true } = {},
) {
  if (!SECTION_ID_MAP[sectionKey]) return;
  scrollToSection(sectionKey);
  setActiveSection(sectionKey, { updateHash, replaceHistory });
}

function sectionInView() {
  const marker = getStickyBottomOffset() + 12;
  let firstSection = null;
  let lastPassed = null;
  for (const sectionKey of orderedSectionKeys()) {
    const section = getSectionElement(sectionKey);
    if (!section) continue;
    if (!firstSection) firstSection = sectionKey;
    const top = section.getBoundingClientRect().top;
    if (top <= marker) {
      lastPassed = sectionKey;
    }
  }
  return lastPassed || firstSection;
}

function refreshScrollSpy() {
  const currentSection = sectionInView();
  if (currentSection) {
    setActiveSection(currentSection, { updateHash: false });
  }
}

function scheduleScrollSpyRefresh() {
  if (isScrollRefreshScheduled) return;
  isScrollRefreshScheduled = true;
  window.requestAnimationFrame(() => {
    isScrollRefreshScheduled = false;
    refreshScrollSpy();
  });
}

function buildTabsInnerHtml(sections) {
  let markup = "";
  for (const sectionKey of orderedSectionKeys(sections)) {
    const isActive = sectionKey === activeSection;
    markup += `<a class="score-tab score-track-link${isActive ? " active" : ""}" data-section="${sectionKey}" href="#${SECTION_ID_MAP[sectionKey]}"${isActive ? ' aria-current="page"' : ""}>${titleCase(sectionKey)}</a>`;
  }
  return markup;
}

function attachTabListeners() {
  document.querySelectorAll(".score-track-link").forEach((link) => {
    link.addEventListener("click", (event) => {
      event.preventDefault();
      const sectionKey = link.dataset.section;
      if (sectionKey === activeSection) return;
      navigateToSection(sectionKey, { replaceHistory: false, updateHash: true });
    });
  });
}

function buildControlsInnerHtml(challengerNames, baseline, depths) {
  let markup = "";

  markup += '<label>Baseline: <select id="baseline-select">';
  for (const name of challengerNames) {
    const selected = name === baseline ? " selected" : "";
    markup += `<option value="${name}"${selected}>${displayName(name)}</option>`;
  }
  markup += "</select></label>";

  markup += '<span id="depth-toggle" class="depth-toggle">';
  markup += `<button class="depth-toggle-btn${showAllMode ? " active" : ""}" data-depth="all">All</button>`;
  markup += '<span class="depth-pills">';
  for (const depth of depths) {
    const active = !showAllMode && selectedDepths.has(depth) ? " active" : "";
    markup += `<button class="depth-toggle-btn${active}" data-depth="${depth}">${depth}</button>`;
  }
  markup += "</span></span>";

  markup += '<span class="display-toggle">';
  markup += `<button class="display-toggle-btn${!showPercentDiff ? " active" : ""}" data-display="values">Values</button>`;
  markup += `<button class="display-toggle-btn${showPercentDiff ? " active" : ""}" data-display="percent-diff">% diff</button>`;
  markup += '</span>';

  markup += '<div id="color-legend" class="color-legend"></div>';

  markup += `<label>\u00b1 <input id="scale-input" type="number" min="1" step="1" value="${maxScale}"> %</label>`;

  return markup;
}

function buildRegionSelectorInnerHtml(regionIds) {
  if (regionIds.length <= 1) return "";

  let markup = '<div class="region-selector-layout">';
  markup += '<div class="region-selector-copy">';
  markup += '<div class="region-selector-row">';
  markup += '<span class="region-selector-label">Region</span>';
  markup += '<div class="region-chip-group" role="group" aria-label="Evaluation region">';
  for (const regionId of regionIds) {
    const active = regionId === activeRegion ? " active" : "";
    const label = regionLabels[regionId] || titleCase(regionId);
    const description = regionMetadata[regionId]?.description || "";
    markup += `<button type="button" class="region-chip${active}" data-region="${regionId}" aria-pressed="${regionId === activeRegion}" title="${description}">${label}</button>`;
  }
  markup += "</div></div>";

  const activeDescription = regionMetadata[activeRegion]?.description;
  if (activeDescription) {
    markup += `<div class="region-selector-description">${activeDescription}</div>`;
  }
  markup += "</div>";
  markup += '<div id="region-globe" class="region-globe" aria-live="polite"></div>';
  markup += "</div>";

  return markup;
}

function renderRegionGlobe(regionIds) {
  if (!window.OceanBenchRegionGlobe) return;
  window.OceanBenchRegionGlobe.render({
    activeRegion,
    regionIds,
    regionLabels,
    regionMetadata,
  });
}

function renderRegionSelector(regionIds) {
  const existing = document.getElementById("region-selector");
  if (regionIds.length <= 1) {
    if (existing) existing.remove();
    return;
  }

  const wrapper = document.getElementById("all-scores");
  if (!wrapper) return;

  const regionSelector = existing || document.createElement("div");
  regionSelector.id = "region-selector";
  regionSelector.innerHTML = buildRegionSelectorInnerHtml(regionIds);

  if (!existing) {
    wrapper.parentNode.insertBefore(regionSelector, wrapper);
  }

  renderRegionGlobe(regionIds);
}

function ensureHeaderElement() {
  const existing = document.getElementById("score-header");
  if (existing) return document.getElementById("score-controls");

  const wrapper = document.getElementById("all-scores");
  if (wrapper) {
    const stale = wrapper.querySelector(".controls:not(#score-controls)");
    if (stale) stale.remove();
  }

  const header = document.createElement("div");
  header.id = "score-header";

  const tabNavigation = document.createElement("nav");
  tabNavigation.id = "score-tabs";
  tabNavigation.setAttribute("aria-label", "Score sections");

  const controlsElement = document.createElement("div");
  controlsElement.id = "score-controls";
  controlsElement.className = "controls";

  header.appendChild(tabNavigation);
  header.appendChild(controlsElement);

  if (wrapper) {
    wrapper.insertBefore(header, wrapper.firstElementChild);
  }

  return controlsElement;
}

function getAvailableDepths(challengers, baseline, sections) {
  for (const sectionConfig of Object.values(sections)) {
    const depthScore = challengers[baseline]?.[sectionConfig.depth_metric];
    if (depthScore) {
      return Object.keys(depthScore.depths);
    }
  }
  return [];
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
  regionId,
  metricKey,
  baseline,
  unifyVariables,
  depthGroupsConfig = null,
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

  if (unifyVariables) {
    const seen = new Set();
    const allVariables = [];
    for (const depth of visibleDepths) {
      for (const variable of Object.keys(baselineScore.depths[depth]?.variables || {})) {
        if (!seen.has(variable)) {
          seen.add(variable);
          allVariables.push(variable);
        }
      }
    }
    const depthSets = visibleDepths.map(
      (d) => new Set(Object.keys(baselineScore.depths[d]?.variables || {})),
    );
    const commonVars = allVariables.filter((v) => depthSets.every((s) => s.has(v)));
    const partialVars = allVariables.filter((v) => !depthSets.every((s) => s.has(v)));
    const unionVariables = [...commonVars, ...partialVars];
    return renderDepthGroup(
      baselineScore, orderedNames, challengers, regionId, metricKey, visibleDepths, unionVariables, baseline,
    );
  }

  if (depthGroupsConfig) {
    const visibleDepthSet = new Set(visibleDepths);
    return depthGroupsConfig.map((group) => {
      const groupDepths = group.depths.filter((depth) => visibleDepthSet.has(depth));
      if (groupDepths.length === 0) return "";
      const groupVariables = group.variables.filter((variable) => groupDepths.some(
        (depth) => baselineScore.depths[depth]?.variables?.[variable],
      ));
      return renderDepthGroup(
        baselineScore,
        orderedNames,
        challengers,
        regionId,
        metricKey,
        groupDepths,
        groupVariables,
        baseline,
        group.show_depth_label || false,
      );
    }).join("");
  }

  const depthGroups = groupDepthsByVariables(baselineScore, visibleDepths);
  return depthGroups.map((group) => renderDepthGroup(
    baselineScore, orderedNames, challengers, regionId, metricKey, group.depths, group.variables, baseline,
  )).join("");
}

function renderDepthGroup(
  baselineScore, orderedNames, challengers, regionId, metricKey, depths, variables, baseline, showDepthLabelForSingleDepth = false,
) {
  if (variables.length === 0 || depths.length === 0) return "";

  const referenceDepth = depths[0];
  const leadDays = getLeadDays(baselineScore, referenceDepth);
  const totalColumns = 1 + variables.length * leadDays.length;

  let thead = "<thead>";
  thead += `<tr><th class="model-col">Models</th>`;
  for (const variable of variables) {
    const sourceDepth = depths.find(
      (d) => baselineScore.depths[d]?.variables?.[variable],
    ) || referenceDepth;
    const unit = getUnit(baselineScore, sourceDepth, variable);
    const standardName = getStandardName(baselineScore, sourceDepth, variable);
    thead += `<th class="var-header" colspan="${leadDays.length}">${formatVariableHeader(variable, unit, standardName, metricKey)}</th>`;
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
    if (depths.length > 1 || showDepthLabelForSingleDepth) {
      tbody += `<tr class="depth-separator"><th class="depth-separator-cell">${depth}</th><td colspan="${totalColumns - 1}" style="border: none;"></td></tr>`;
    }
    const depthVariables = new Set(Object.keys(baselineScore.depths[depth]?.variables || {}));
    tbody += buildDataRows(
      orderedNames,
      challengers,
      regionId,
      metricKey,
      depth,
      variables,
      leadDays,
      baseline,
      depthVariables,
    );
  }
  tbody += "</tbody>";

  const tableClass = depths.length > 1 ? "score-table depth-table" : "score-table";
  return `<table class="${tableClass}">${thead}${tbody}</table>`;
}

function renderCombinedFlatMetrics(
  challengers,
  challengerNames,
  regionId,
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
      const standardName = getStandardName(baselineScore, "flat", variable);
      thead += `<th class="var-header" colspan="${leadDays.length}">${formatVariableHeader(variable, unit, standardName, metricKey)}</th>`;
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
    buildCombinedDataRows(orderedNames, challengers, regionId, metricSpecs, baseline) +
    "</tbody>";

  return `<div class="score-table-wrapper"><table class="score-table">${thead}${tbody}</table></div>`;
}

function renderMetricSection(
  containerId,
  depthMetric,
  flatMetrics,
  challengers,
  challengerNames,
  regionId,
  baseline,
  metricTitles,
  unifyVariables,
  depthGroups,
) {
  const container = document.getElementById(containerId);
  if (!container) return;

  let markup = "";

  markup += '<div class="depth-section">';
  markup += `<h3>${metricTitles[depthMetric]}</h3>`;
  markup += renderDepthMetric(
    challengers,
    challengerNames,
    regionId,
    depthMetric,
    baseline,
    unifyVariables,
    depthGroups,
  );
  markup += "</div>";

  const flatMarkup = renderCombinedFlatMetrics(
    challengers,
    challengerNames,
    regionId,
    flatMetrics,
    baseline,
  );
  if (flatMarkup) {
    markup += `<h3>Physically consistent diagnostic variables</h3>`;
    markup += flatMarkup;
  }

  container.innerHTML = markup;
}

function formatRgb(color) {
  return `rgb(${color[0]}, ${color[1]}, ${color[2]})`;
}

function formatTickLabel(value) {
  if (value === 0) return "0%";
  const formatted = Number.isInteger(value) ? `${value}` : value.toFixed(1).replace(/\.0$/, "");
  return value > 0 ? `+${formatted}%` : `${formatted}%`;
}

function updateColorLegend() {
  const legend = document.getElementById("color-legend");
  if (!legend) return;

  const binCount = colorScaleEdges.length - 1;

  let binsHtml = "";
  for (let i = binCount - 1; i >= 0; i--) {
    binsHtml += `<span class="legend-bin" style="background:${formatRgb(getBinColor(i))}"></span>`;
  }

  let ticksHtml = "";
  for (let i = 0; i < colorScaleEdges.length; i++) {
    if (!isFinite(colorScaleEdges[i])) continue;
    const percent = ((binCount - i) / binCount) * 100;
    const label = formatTickLabel(colorScaleEdges[i]);
    ticksHtml += `<span class="legend-tick-label" style="left:${percent}%">${label}</span>`;
  }

  legend.innerHTML =
    `<span class="legend-label">Worse</span>` +
    `<span class="legend-bar-wrapper">` +
      `<span class="legend-bar">${binsHtml}</span>` +
      `<span class="legend-ticks">${ticksHtml}</span>` +
    `</span>` +
    `<span class="legend-label">Better</span>`;
}

function setupCellHighlight() {
  document.querySelectorAll(".score-table").forEach((table) => {
    table.addEventListener("mouseover", (event) => {
      const cell = event.target.closest("td");
      if (!cell || cell.classList.contains("no-var-cell") || cell.closest("tr").classList.contains("depth-separator")) return;
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
  const header = document.getElementById("score-header");
  if (header) {
    const headerHeight = header.getBoundingClientRect().height;
    document.documentElement.style.setProperty("--controls-height", headerHeight + "px");
  }
}

function attachControlListeners() {
  document.querySelectorAll(".region-chip").forEach((button) => {
    button.addEventListener("click", () => {
      if (button.dataset.region === activeRegion) return;
      activeRegion = button.dataset.region;
      renderAllTables();
    });
  });

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

  const scaleInput = document.getElementById("scale-input");
  if (scaleInput) {
    function applyScale(value) {
      if (value > 0 && isFinite(value)) {
        maxScale = value;
        scaleInput.value = maxScale;
        colorScaleEdges = buildBinEdges(maxScale);
        renderTablesOnly();
      }
    }

    function snapScale(direction) {
      const current = maxScale;
      if (direction > 0) {
        const next = SCALE_SNAPS.find((v) => v > current);
        return next || current;
      }
      for (let i = SCALE_SNAPS.length - 1; i >= 0; i--) {
        if (SCALE_SNAPS[i] < current) return SCALE_SNAPS[i];
      }
      return current;
    }

    scaleInput.addEventListener("change", () => applyScale(Number(scaleInput.value)));

    scaleInput.addEventListener("input", () => {
      const enteredValue = Number(scaleInput.value);
      const direction = enteredValue > maxScale ? 1 : -1;
      applyScale(snapScale(direction));
    });

    scaleInput.addEventListener("wheel", (event) => {
      event.preventDefault();
      applyScale(snapScale(event.deltaY < 0 ? 1 : -1));
    }, { passive: false });

    scaleInput.addEventListener("keydown", (event) => {
      if (event.key === "ArrowUp" || event.key === "ArrowDown") {
        event.preventDefault();
        applyScale(snapScale(event.key === "ArrowUp" ? 1 : -1));
      }
    });
  }

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

function ensureParsedData() {
  if (!parsedData) {
    const dataElement = document.getElementById("scores-data");
    if (!dataElement) return null;
    parsedData = JSON.parse(dataElement.textContent);
    challengerLabels = parsedData.challenger_labels || {};
    regionLabels = parsedData.region_labels || {};
    regionMetadata = parsedData.region_metadata || {};
    const regionIds = parsedData.region_order || Object.keys(parsedData.regions || {});
    if (!activeRegion || !regionIds.includes(activeRegion)) {
      activeRegion = regionIds[0] || null;
    }
  }
  return parsedData;
}

function getRegionIds(data) {
  return data.region_order || Object.keys(data.regions || {});
}

function getCurrentRegionData(data) {
  if (!activeRegion) return null;
  return data.regions?.[activeRegion] || null;
}

function resolveBaselineSelection(challengerNames, selectedBaseline) {
  if (selectedBaseline && challengerNames.includes(selectedBaseline)) {
    return selectedBaseline;
  }
  if (challengerNames.includes("glo12")) {
    return "glo12";
  }
  return challengerNames[0];
}

function renderTablesOnly() {
  const data = ensureParsedData();
  if (!data) return;
  const regionData = getCurrentRegionData(data);
  if (!regionData) return;
  const { challengers, challenger_names: challengerNames } = regionData;
  if (!challengerNames || challengerNames.length === 0) return;
  const { metric_titles: metricTitles, sections } = data;

  const existingSelect = document.getElementById("baseline-select");
  const baseline = resolveBaselineSelection(challengerNames, existingSelect?.value);

  for (const [sectionKey, sectionConfig] of Object.entries(sections)) {
    renderMetricSection(
      `${sectionKey}-scores`,
      sectionConfig.depth_metric,
      sectionConfig.flat_metrics,
      challengers,
      challengerNames,
      activeRegion,
      baseline,
      metricTitles,
      sectionKey !== "observations",
      sectionConfig.depth_groups || null,
    );
  }

  updateColorLegend();
  setupCellHighlight();
}

function renderAllTables() {
  const data = ensureParsedData();
  if (!data) return;
  const regionData = getCurrentRegionData(data);
  if (!regionData) return;
  const { challengers, challenger_names: challengerNames } = regionData;
  const { metric_titles: metricTitles, sections } = data;
  const regionIds = getRegionIds(data);
  if (!challengerNames || challengerNames.length === 0) return;

  const existingSelect = document.getElementById("baseline-select");
  const baseline = resolveBaselineSelection(challengerNames, existingSelect?.value);
  const availableDepths = getAvailableDepths(challengers, baseline, sections);

  for (const [sectionKey, sectionConfig] of Object.entries(sections)) {
    renderMetricSection(
      `${sectionKey}-scores`,
      sectionConfig.depth_metric,
      sectionConfig.flat_metrics,
      challengers,
      challengerNames,
      activeRegion,
      baseline,
      metricTitles,
      sectionKey !== "observations",
      sectionConfig.depth_groups || null,
    );
  }

  const controlsElement = ensureHeaderElement();
  controlsElement.innerHTML = buildControlsInnerHtml(challengerNames, baseline, availableDepths);
  renderRegionSelector(regionIds);

  const tabNavigation = document.getElementById("score-tabs");
  if (tabNavigation) {
    tabNavigation.innerHTML = buildTabsInnerHtml(sections);
  }

  updateColorLegend();
  updateStickyOffsets();
  attachControlListeners();
  attachTabListeners();
  setActiveSection(activeSection);
  refreshScrollSpy();
  setupCellHighlight();
}

function init() {
  if (!document.getElementById("scores-data")) return;
  const initialSection = readSectionFromHash();
  if (initialSection) {
    activeSection = initialSection;
  }
  renderAllTables();

  if (initialSection) {
    navigateToSection(initialSection, { replaceHistory: true, updateHash: true });
  }

  window.addEventListener("hashchange", () => {
    const newSection = readSectionFromHash();
    if (newSection) {
      navigateToSection(newSection, { updateHash: false });
    }
  });

  window.addEventListener("scroll", scheduleScrollSpyRefresh, { passive: true });

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
    const syncHeaderState = () => {
      const hidden = header.classList.contains("headroom--unpinned");
      document.body.classList.toggle("nav-hidden", hidden);
      updateStickyOffsets();
      refreshScrollSpy();
    };
    document.documentElement.style.setProperty("--navbar-full-height", `${navbar.offsetHeight}px`);
    syncHeaderState();

    new MutationObserver(() => {
      syncHeaderState();
    }).observe(header, { attributes: true, attributeFilter: ["class"] });

    window.addEventListener("resize", () => {
      document.documentElement.style.setProperty("--navbar-full-height", `${navbar.offsetHeight}px`);
      syncHeaderState();
    });
  } else {
    window.addEventListener("resize", () => {
      updateStickyOffsets();
      refreshScrollSpy();
    });
  }
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
