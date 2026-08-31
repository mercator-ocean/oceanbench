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

const SCORE_DECIMALS = 2;

let selectedDepths = new Set();
let showAllMode = true;
let showPercentDiff = false;
let parsedData = null;
let parsedDataByView = {};
let ensembleSections = [];
let challengerLabels = {};
let regionLabels = {};
let regionMetadata = {};
let activeView = "deterministic";
let activeTrack = "high_resolution";
let activeSection = "observations";
let activeRegion = null;
let activeVersion = null;
let isScrollRefreshScheduled = false;

let selectedBaseline = null;
let defaultVersionValue = null;
let defaultRegionValue = null;
let defaultBaselineValue = null;

const SECTION_ORDER = ["observations", "reanalysis", "analysis"];

const SECTION_ID_MAP = {
  observations: "comparison-to-observations",
  reanalysis: "comparison-to-reanalysis",
  analysis: "comparison-to-analysis",
};

const TRACK_LABELS = {
  high_resolution: "High resolution",
  one_degree: "1 degree",
};

const TRACK_NOTES = {
  high_resolution: "Models evaluated at their native high resolution.",
  one_degree:
    "Non-one-degree base models whose forecasts are interpolated to the one degree resolution.",
};

const VIEW_ORDER = ["deterministic", "ensemble"];

const VIEW_LABELS = {
  deterministic: "Deterministic",
  ensemble: "Ensemble",
};

const VIEW_NOTES = {
  deterministic: "One forecast per system, scored against observations, reanalysis and analysis.",
  ensemble: "Ensemble means and probabilistic skill of the ensemble systems, global, year 2024.",
};

const VIEW_DATA_ELEMENT_IDS = {
  deterministic: "scores-data",
  ensemble: "ensemble-scores-data",
};

const viewState = {
  deterministic: {},
  ensemble: {},
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

// A table whose metric has no "lower is better" direction sets a transform here, so that its
// cells are shaded on the ramp below by whatever does read as better for that metric. The
// default is no transform, which is what every deterministic table uses.
let colorTransform = null;

function colorScore(value) {
  if (colorTransform === "closeness_to_one") return Math.abs(value - 1);
  return value;
}

// The percent a cell reads compares the very numbers its shading compares, so a cell shaded as
// better can never print a worse percent. For a ratio those numbers are the distances from one.
function comparedPercent(referenceValue, comparedValue) {
  const reference = colorScore(referenceValue);
  const compared = colorScore(comparedValue);
  if (reference === 0) return compared === 0 ? 0 : null;
  return ((compared - reference) / Math.abs(reference)) * 100;
}

function comparisonBasisLabel() {
  return colorTransform === "closeness_to_one" ? " (distance from 1)" : "";
}

function getCellStyle(referenceValue, comparedValue) {
  const percentDiff = comparedPercent(referenceValue, comparedValue) ?? 0;
  const binIndex = getBinIndex(percentDiff);
  const color = getBinColor(binIndex);
  return `background-color:rgb(${color[0]}, ${color[1]}, ${color[2]}); color: ${textColorForBackground(color)}`;
}

function formatScoreValue(value) {
  // Two decimals everywhere, like the deterministic score tables. Scores that only differ past
  // the second decimal print the same; the shading and the tooltip still carry the difference.
  if (!Number.isFinite(value)) return "";
  return value.toFixed(SCORE_DECIMALS);
}

function formatPercentDiff(referenceValue, comparedValue) {
  const percent = comparedPercent(referenceValue, comparedValue);
  if (percent === null) return "N/A";
  const sign = percent > 0 ? "+" : "";
  return `${sign}${Math.round(percent)}%`;
}

function formatPercentDiffForCell(referenceValue, comparedValue) {
  const percent = comparedPercent(referenceValue, comparedValue);
  if (percent === null) return "N/A";
  if (percent > 999) return ">999%";
  if (percent < -999) return "<-999%";
  return `${Math.round(percent)}%`;
}

function getValue(scoreData, depth, variable, leadDay) {
  try {
    return scoreData.depths[depth].variables[variable].data[leadDay] ?? null;
  } catch {
    return null;
  }
}

function getLeadDays(scoreData, depth) {
  const firstVariable = Object.keys(scoreData.depths[depth].variables)[0];
  const allDays = Object.keys(scoreData.depths[depth].variables[firstVariable].data);
  if (activeView === "ensemble") return allDays;
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

function trackKeyForChallenger(name) {
  return name.endsWith("_1_degree") ? "one_degree" : "high_resolution";
}

function getTrackChallengerNames(challengerNames, trackKey) {
  return challengerNames.filter((name) => trackKeyForChallenger(name) === trackKey);
}

function getAvailableTracks(challengerNames) {
  const tracks = [];
  if (getTrackChallengerNames(challengerNames, "high_resolution").length > 0) {
    tracks.push("high_resolution");
  }
  if (getTrackChallengerNames(challengerNames, "one_degree").length > 0) {
    tracks.push("one_degree");
  }
  return tracks;
}

function titleCase(text) {
  return text.replace(/(^|\s)\w/g, (character) => character.toUpperCase());
}

function metricHeaderLabel(metricKey, unit) {
  if (metricKey.startsWith("rmsd")) return `RMSE (${unit})`;
  if (metricKey.startsWith("crps")) return `Fair CRPS (${unit})`;
  if (metricKey.startsWith("spread_error_ratio")) return "Spread error ratio";
  return `(${unit})`;
}

function formatVariableHeader(variable, unit, standardName, metricKey) {
  const displayName = titleCase(variable);
  const metricLabel = metricHeaderLabel(metricKey, unit);
  let header = `${displayName}<br><span class="metric-label">${metricLabel}</span>`;
  if (standardName && standardName !== "unknown") {
    header += `<br><span class="standard-name">${standardName}</span>`;
  }
  return header;
}

function modelHeaderCell(name, regionId) {
  if (activeView === "ensemble") {
    return `<th class="model-col">${displayName(name)}</th>`;
  }
  return `<th class="model-col"><a href="reports/${activeVersion}/${name}.${regionId}.report.html">${displayName(name)}</a></th>`;
}

function cellTooltip(variable, unit, day, value, referenceValue, isBaseline, baselineName) {
  const unitSuffix = unit ? ` ${unit}` : "";
  let tooltip = `${titleCase(variable)}, lead day ${day}\nValue: ${value}${unitSuffix}`;
  if (!isBaseline && referenceValue !== null) {
    tooltip += `\nvs ${displayName(baselineName)}${comparisonBasisLabel()}: ${formatPercentDiff(referenceValue, value)}`;
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
    rows += `<tr${rowClass}>${modelHeaderCell(name, regionId)}`;
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
            display = formatScoreValue(value);
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
    rows += `<tr${rowClass}>${modelHeaderCell(name, regionId)}`;
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
              display = formatScoreValue(value);
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

function currentSectionIds() {
  if (activeView === "ensemble") {
    return Object.fromEntries(ensembleSections.map((section) => [section.key, section.key]));
  }
  return SECTION_ID_MAP;
}

function currentSectionOrder() {
  if (activeView === "ensemble") {
    return ensembleSections.map((section) => section.key);
  }
  return SECTION_ORDER;
}

function currentSectionLabel(sectionKey) {
  if (activeView === "ensemble") {
    const section = ensembleSections.find((candidate) => candidate.key === sectionKey);
    return section ? section.label : titleCase(sectionKey);
  }
  return titleCase(sectionKey);
}

function readSectionFromHash() {
  const hash = window.location.hash.slice(1);
  const sectionIds = currentSectionIds();
  if (sectionIds[hash]) return hash;
  for (const [sectionKey, sectionId] of Object.entries(sectionIds)) {
    if (sectionId === hash) return sectionKey;
  }
  return null;
}

function getSectionElement(sectionKey) {
  const sectionId = currentSectionIds()[sectionKey];
  if (!sectionId) return null;
  return document.getElementById(sectionId);
}

function updateSectionHash(sectionKey, replaceHistory) {
  const sectionId = currentSectionIds()[sectionKey];
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
  if (!currentSectionIds()[sectionKey]) return;
  const { updateHash = false, replaceHistory = true } = options;
  activeSection = sectionKey;
  const depthToggle = document.getElementById("depth-toggle");
  if (depthToggle) {
    if (activeView === "ensemble" || activeSection === "observations") {
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
  if (activeView === "ensemble") return currentSectionOrder();
  const availableSections = sections ? new Set(Object.keys(sections)) : null;
  return SECTION_ORDER.filter(
    (sectionKey) => SECTION_ID_MAP[sectionKey] && (!availableSections || availableSections.has(sectionKey)),
  );
}

function navigateToSection(
  sectionKey,
  { replaceHistory = false, updateHash = true } = {},
) {
  if (!currentSectionIds()[sectionKey]) return;
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
  const sectionIds = currentSectionIds();
  let markup = "";
  for (const sectionKey of orderedSectionKeys(sections)) {
    const isActive = sectionKey === activeSection;
    markup += `<a class="score-tab score-track-link${isActive ? " active" : ""}" data-section="${sectionKey}" href="#${sectionIds[sectionKey]}"${isActive ? ' aria-current="page"' : ""}>${currentSectionLabel(sectionKey)}</a>`;
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

function buildSelectorChips(labelOf, values, activeValue, dataAttribute, ariaLabel, isDisabled = () => false) {
  const chips = values
    .map((value) => {
      const active = value === activeValue ? " active" : "";
      const disabled = isDisabled(value);
      const disabledClass = disabled ? " disabled" : "";
      const disabledAttributes = disabled ? ' disabled aria-disabled="true"' : "";
      return `<button type="button" class="selector-chip${active}${disabledClass}" ${dataAttribute}="${value}" aria-pressed="${value === activeValue}"${disabledAttributes}>${labelOf(value)}</button>`;
    })
    .join("");
  return `<div class="selector-chips" role="group" aria-label="${ariaLabel}">${chips}</div>`;
}

function buildSelectorRow(labelText, chipsHtml, description, descriptionClass = "") {
  let markup = '<div class="selector-row">';
  markup += `<span class="selector-label">${labelText}</span>`;
  markup += chipsHtml;
  if (description) {
    markup += `<p class="selector-description${descriptionClass ? ` ${descriptionClass}` : ""}">${description}</p>`;
  }
  markup += "</div>";
  return markup;
}

function hasEnsembleBundle() {
  return Boolean(document.getElementById(VIEW_DATA_ELEMENT_IDS.ensemble));
}

function buildRegionSelectorInnerHtml(regionIds, versionTracks, regionTracks) {
  if (regionIds.length === 0) return "";

  let controls = "";
  if (hasEnsembleBundle()) {
    const viewChips = buildSelectorChips(
      (viewKey) => VIEW_LABELS[viewKey],
      VIEW_ORDER,
      activeView,
      "data-view",
      "Forecast type",
    );
    controls += buildSelectorRow("Forecast", viewChips, VIEW_NOTES[activeView]);
  }

  const regionChips = buildSelectorChips(
    (regionId) => regionLabels[regionId] || titleCase(regionId),
    regionIds,
    activeRegion,
    "data-region",
    "Evaluation region",
  );
  controls += buildSelectorRow("Region", regionChips, regionMetadata[activeRegion]?.description);

  if (activeView !== "ensemble" && versionTracks.length > 1) {
    const trackChips = buildSelectorChips(
      (trackKey) => TRACK_LABELS[trackKey],
      versionTracks,
      activeTrack,
      "data-track",
      "Model resolution track",
      (trackKey) => !regionTracks.includes(trackKey),
    );
    controls += buildSelectorRow("Track", trackChips, TRACK_NOTES[activeTrack], "selector-description--track");
  }

  let markup = '<div id="region-globe" class="region-globe" aria-live="polite"></div>';
  markup += `<div class="region-selector-controls">${controls}</div>`;
  return markup;
}

function getVersionTracks(versionData) {
  const challengerNames = Object.values(versionData?.regions || {})
    .flatMap((region) => region.challenger_names || []);
  return getAvailableTracks(challengerNames);
}

function buildVersionSelectorInnerHtml(versions) {
  let markup = '<span class="version-selector-label">Evaluation version</span>';
  markup += '<select id="version-select" class="version-select" aria-label="Evaluation report version">';
  for (const version of versions) {
    const selected = version === activeVersion ? " selected" : "";
    markup += `<option value="${version}"${selected}>${version}</option>`;
  }
  markup += "</select>";
  markup += '<a class="version-changelog-link" href="https://github.com/mercator-ocean/oceanbench/blob/main/CHANGELOG.md" target="_blank" rel="noopener">Changelog</a>';
  return markup;
}

function renderVersionSelector(versions) {
  const wrapper = document.getElementById("all-scores");
  const existing = document.getElementById("version-selector");
  if (!wrapper || !versions || versions.length <= 1) {
    if (existing) existing.remove();
    return;
  }

  const versionSelector = existing || document.createElement("div");
  versionSelector.id = "version-selector";
  versionSelector.className = "version-selector";
  versionSelector.innerHTML = buildVersionSelectorInnerHtml(versions);

  if (!existing) {
    const anchor = document.getElementById("region-selector") || wrapper;
    anchor.parentNode.insertBefore(versionSelector, anchor);
  }
}

function renderRegionGlobe(regionIds) {
  if (!window.OceanBenchRegionGlobe) return;
  window.OceanBenchRegionGlobe.render({
    activeRegion,
    activeTrack,
    regionIds,
    regionLabels,
    regionMetadata,
  });
}

function renderRegionSelector(regionIds, versionTracks, regionTracks) {
  const existing = document.getElementById("region-selector");
  if (regionIds.length === 0) {
    if (existing) existing.remove();
    return;
  }

  const wrapper = document.getElementById("all-scores");
  if (!wrapper) return;

  const regionSelector = existing || document.createElement("div");
  regionSelector.id = "region-selector";
  regionSelector.innerHTML = buildRegionSelectorInnerHtml(regionIds, versionTracks, regionTracks);

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
  layoutScore = null,
) {
  const baselineScore = layoutScore || challengers[baseline][metricKey];
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

// The ensemble systems stop at different lead days, so a table can end up with a lead column that
// every one of its rows leaves empty. Such a column carries nothing and is dropped. The
// deterministic view keeps whatever columns it asked for, so its markup never moves.
function dropLeadDaysNoSystemReaches(leadDays, orderedNames, challengers, metricKey, depths, variables) {
  if (activeView !== "ensemble") return leadDays;
  return leadDays.filter((day) => orderedNames.some((name) => {
    const score = challengers[name][metricKey];
    if (!score) return false;
    return depths.some((depth) => variables.some((variable) => getValue(score, depth, variable, day) !== null));
  }));
}

function renderDepthGroup(
  baselineScore, orderedNames, challengers, regionId, metricKey, depths, variables, baseline, showDepthLabelForSingleDepth = false,
) {
  if (variables.length === 0 || depths.length === 0) return "";

  const referenceDepth = depths[0];
  const leadDays = dropLeadDaysNoSystemReaches(
    getLeadDays(baselineScore, referenceDepth), orderedNames, challengers, metricKey, depths, variables,
  );
  if (leadDays.length === 0) return "";
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

function resolveMetricBaseline(challengers, challengerNames, metricKey, baseline) {
  if (challengers[baseline]?.[metricKey]) return baseline;
  return challengerNames.find((name) => challengers[name]?.[metricKey]) || null;
}

function renderEnsembleMetric(challengers, challengerNames, regionId, metric, baseline) {
  const metricBaseline = resolveMetricBaseline(challengers, challengerNames, metric.metric_key, baseline);
  if (!metricBaseline) return "";

  colorTransform = metric.color_transform || null;
  const tables = renderDepthMetric(
    challengers,
    challengerNames,
    regionId,
    metric.metric_key,
    metricBaseline,
    metric.unify_variables || false,
    metric.depth_groups || null,
    metric.layout,
  );
  colorTransform = null;
  if (!tables) return "";

  let markup = '<div class="depth-section">';
  markup += `<h3>${metric.title}</h3>`;
  if (metricBaseline !== baseline) {
    // The selector still names the system the reader chose, so a table that had to fall back onto
    // another baseline says so rather than letting the shading and the percentages read as if the
    // chosen system were the reference.
    markup += `<p class="ensemble-table-note">${displayName(baseline)} carries no score in this table,`
      + ` so the cells are shaded and compared against ${displayName(metricBaseline)}.</p>`;
  }
  markup += tables;
  if (metric.note) {
    markup += `<p class="ensemble-table-note">${metric.note}</p>`;
  }
  markup += "</div>";
  return markup;
}

function renderEnsembleSections(challengers, challengerNames, regionId, baseline) {
  for (const section of ensembleSections) {
    const container = document.getElementById(section.container);
    if (!container) continue;
    container.innerHTML = section.metrics
      .map((metric) => renderEnsembleMetric(challengers, challengerNames, regionId, metric, baseline))
      .join("");
  }
}

function renderActiveSections(data, challengers, challengerNames, baseline) {
  if (activeView === "ensemble") {
    renderEnsembleSections(challengers, challengerNames, activeRegion, baseline);
    return;
  }
  for (const [sectionKey, sectionConfig] of Object.entries(data.sections)) {
    renderMetricSection(
      `${sectionKey}-scores`,
      sectionConfig.depth_metric,
      sectionConfig.flat_metrics,
      challengers,
      challengerNames,
      activeRegion,
      baseline,
      data.metric_titles,
      sectionKey !== "observations",
      sectionConfig.depth_groups || null,
    );
  }
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

function applyViewVisibility() {
  for (const viewKey of VIEW_ORDER) {
    const viewElement = document.getElementById(`${viewKey}-view`);
    if (!viewElement) continue;
    if (viewKey === activeView) {
      viewElement.removeAttribute("hidden");
    } else {
      viewElement.setAttribute("hidden", "");
    }
  }
}

function switchView(viewKey) {
  viewState[activeView] = {
    version: activeVersion,
    region: activeRegion,
    track: activeTrack,
    baseline: selectedBaseline,
    section: activeSection,
  };
  activeView = viewKey;
  const restored = viewState[activeView];
  activeVersion = restored.version || null;
  activeRegion = restored.region || null;
  activeTrack = restored.track || "high_resolution";
  selectedBaseline = restored.baseline || null;
  activeSection = restored.section || "";
  const staleBaselineSelect = document.getElementById("baseline-select");
  if (staleBaselineSelect) staleBaselineSelect.remove();
  applyViewVisibility();
  renderAllTables();
  updateSectionHash(activeSection, true);
}

function attachSelectorListeners() {
  document.querySelectorAll("#region-selector .selector-chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const { region, track, view } = chip.dataset;
      if (view && view !== activeView) {
        switchView(view);
      } else if (region && region !== activeRegion) {
        activeRegion = region;
        renderAllTables();
      } else if (track && track !== activeTrack) {
        activeTrack = track;
        selectedDepths = new Set();
        showAllMode = true;
        renderAllTables();
      }
    });
  });
}

function attachControlListeners() {
  const versionSelect = document.getElementById("version-select");
  if (versionSelect) {
    versionSelect.addEventListener("change", () => {
      const selectedVersion = versionSelect.value;
      if (!selectedVersion || selectedVersion === activeVersion) return;
      activeVersion = selectedVersion;
      applyActiveVersion();
      renderAllTables();
    });
  }

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
  if (parsedDataByView[activeView] === undefined) {
    const dataElement = document.getElementById(VIEW_DATA_ELEMENT_IDS[activeView]);
    parsedDataByView[activeView] = dataElement ? JSON.parse(dataElement.textContent) : null;
  }
  const viewData = parsedDataByView[activeView];
  if (!viewData) return null;
  if (parsedData !== viewData) {
    parsedData = viewData;
    ensembleSections = viewData.ensemble_sections || [];
    const versions = getVersions(parsedData);
    if (!activeVersion || !versions.includes(activeVersion)) {
      activeVersion = resolveDefaultVersion(parsedData);
    }
    applyActiveVersion();
    if (!currentSectionIds()[activeSection]) {
      activeSection = currentSectionOrder()[0] || activeSection;
    }
  }
  return parsedData;
}

function getVersions(data) {
  return data.version_order || Object.keys(data.versions || {});
}

function resolveDefaultVersion(data) {
  const versions = getVersions(data);
  return versions.includes(data.default_version) ? data.default_version : versions[0] || null;
}

function getActiveVersionData(data) {
  if (!data || !data.versions) return null;
  return data.versions[activeVersion] || null;
}

function applyActiveVersion() {
  const versionData = getActiveVersionData(parsedData) || {};
  challengerLabels = versionData.challenger_labels || {};
  regionLabels = versionData.region_labels || {};
  regionMetadata = versionData.region_metadata || {};
  const regionIds = versionData.region_order || Object.keys(versionData.regions || {});
  if (!activeRegion || !regionIds.includes(activeRegion)) {
    activeRegion = regionIds[0] || null;
  }
}

function getRegionIds(data) {
  const versionData = getActiveVersionData(data) || {};
  return versionData.region_order || Object.keys(versionData.regions || {});
}

function getCurrentRegionData(data) {
  const versionData = getActiveVersionData(data);
  if (!versionData || !activeRegion) return null;
  return versionData.regions?.[activeRegion] || null;
}

function resolveBaselineSelection(challengerNames, selectedBaseline, trackKey) {
  const trackChallengerNames = getTrackChallengerNames(challengerNames, trackKey);
  if (trackChallengerNames.length === 0) {
    return null;
  }
  if (selectedBaseline && trackChallengerNames.includes(selectedBaseline)) {
    return selectedBaseline;
  }
  const preferredBaseline = trackKey === "one_degree" ? "glo12_1_degree" : "glo12";
  if (trackChallengerNames.includes(preferredBaseline)) {
    return preferredBaseline;
  }
  return trackChallengerNames[0];
}

function resolveTrackSelection(challengerNames) {
  const availableTracks = getAvailableTracks(challengerNames);
  if (!availableTracks.includes(activeTrack)) {
    activeTrack = availableTracks[0] || "high_resolution";
  }
  return availableTracks;
}

function resolveVisibleChallengerNames(challengerNames) {
  const availableTracks = resolveTrackSelection(challengerNames);
  return {
    availableTracks,
    visibleChallengerNames: getTrackChallengerNames(challengerNames, activeTrack),
  };
}

function resolveBaselineSelectionForTrack(challengerNames, selectedBaseline) {
  if (selectedBaseline && challengerNames.includes(selectedBaseline)) {
    return selectedBaseline;
  }
  return resolveBaselineSelection(challengerNames, selectedBaseline, activeTrack);
}

function renderTablesOnly() {
  const data = ensureParsedData();
  if (!data) return;
  const regionData = getCurrentRegionData(data);
  if (!regionData) return;
  const { challengers, challenger_names: challengerNames } = regionData;
  if (!challengerNames || challengerNames.length === 0) return;
  const { visibleChallengerNames } = resolveVisibleChallengerNames(challengerNames);
  if (visibleChallengerNames.length === 0) return;

  const existingSelect = document.getElementById("baseline-select");
  const baseline = resolveBaselineSelectionForTrack(visibleChallengerNames, existingSelect?.value ?? selectedBaseline);
  if (!baseline) return;

  renderActiveSections(data, challengers, visibleChallengerNames, baseline);

  updateColorLegend();
  setupCellHighlight();

  selectedBaseline = baseline;
  writeUrlState();
}

function renderAllTables() {
  const data = ensureParsedData();
  if (!data) return;
  const regionData = getCurrentRegionData(data);
  if (!regionData) return;
  const { challengers, challenger_names: challengerNames } = regionData;
  const { sections } = data;
  const regionIds = getRegionIds(data);
  if (!challengerNames || challengerNames.length === 0) return;
  const { availableTracks, visibleChallengerNames } = resolveVisibleChallengerNames(challengerNames);
  if (visibleChallengerNames.length === 0) return;

  const existingSelect = document.getElementById("baseline-select");
  const baseline = resolveBaselineSelectionForTrack(visibleChallengerNames, existingSelect?.value ?? selectedBaseline);
  if (!baseline) return;
  const availableDepths = getAvailableDepths(challengers, baseline, sections);

  renderActiveSections(data, challengers, visibleChallengerNames, baseline);

  const versionTracks = getVersionTracks(getActiveVersionData(data));

  const controlsElement = ensureHeaderElement();
  controlsElement.innerHTML = buildControlsInnerHtml(visibleChallengerNames, baseline, availableDepths);
  renderRegionSelector(regionIds, versionTracks, availableTracks);
  renderVersionSelector(getVersions(data));

  const tabNavigation = document.getElementById("score-tabs");
  if (tabNavigation) {
    tabNavigation.innerHTML = buildTabsInnerHtml(sections);
  }

  updateColorLegend();
  updateStickyOffsets();
  attachSelectorListeners();
  attachControlListeners();
  attachTabListeners();
  setActiveSection(activeSection);
  refreshScrollSpy();
  setupCellHighlight();

  defaultVersionValue = resolveDefaultVersion(data);
  defaultRegionValue = regionIds[0] || null;
  defaultBaselineValue = resolveBaselineSelectionForTrack(visibleChallengerNames, null);
  selectedBaseline = baseline;
  writeUrlState();
}

function applyUrlStateFromLocation() {
  const parameters = new URLSearchParams(window.location.search);

  const viewParameter = parameters.get("view");
  if (viewParameter && VIEW_ORDER.includes(viewParameter)) activeView = viewParameter;

  const versionParameter = parameters.get("version");
  if (versionParameter) activeVersion = versionParameter;

  const regionParameter = parameters.get("region");
  if (regionParameter) activeRegion = regionParameter;

  const trackParameter = parameters.get("track");
  if (trackParameter) activeTrack = trackParameter;

  const baselineParameter = parameters.get("baseline");
  if (baselineParameter) selectedBaseline = baselineParameter;

  const displayParameter = parameters.get("display");
  if (displayParameter) showPercentDiff = displayParameter === "percent";

  const depthsParameter = parameters.get("depths");
  if (depthsParameter) {
    if (depthsParameter === "all") {
      showAllMode = true;
      selectedDepths = new Set();
    } else {
      showAllMode = false;
      selectedDepths = new Set(depthsParameter.split(",").filter((depth) => depth !== ""));
    }
  }

  const scaleParameter = parameters.get("scale");
  if (scaleParameter) {
    const parsedScale = Number(scaleParameter);
    if (isFinite(parsedScale) && parsedScale > 0) {
      maxScale = parsedScale;
      colorScaleEdges = buildBinEdges(maxScale);
    }
  }
}

function writeUrlState() {
  if (!parsedData) return;
  const parameters = new URLSearchParams();

  if (activeView !== "deterministic") {
    parameters.set("view", activeView);
  }
  if (activeVersion && activeVersion !== defaultVersionValue) {
    parameters.set("version", activeVersion);
  }
  if (activeRegion && activeRegion !== defaultRegionValue) {
    parameters.set("region", activeRegion);
  }
  if (activeTrack && activeTrack !== "high_resolution") {
    parameters.set("track", activeTrack);
  }
  if (selectedBaseline && selectedBaseline !== defaultBaselineValue) {
    parameters.set("baseline", selectedBaseline);
  }
  if (showPercentDiff) {
    parameters.set("display", "percent");
  }
  if (!showAllMode && selectedDepths.size > 0) {
    const orderedDepths = [...selectedDepths].sort((first, second) => Number(first) - Number(second));
    parameters.set("depths", orderedDepths.join(","));
  }
  if (maxScale !== 80) {
    parameters.set("scale", `${maxScale}`);
  }

  const queryString = parameters.toString();
  const newRelativeUrl =
    window.location.pathname + (queryString ? `?${queryString}` : "") + window.location.hash;
  window.history.replaceState(null, "", newRelativeUrl);
}

function init() {
  if (!document.getElementById("scores-data")) return;
  applyUrlStateFromLocation();
  applyViewVisibility();
  ensureParsedData();
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
