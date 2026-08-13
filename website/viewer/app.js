// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// OceanBench viewer — comparison-first field explorer (contracts.md §6).
//
// Comparison is the primitive: 1 or 2 synchronized panels (Forecast 1 / Forecast 2)
// sharing viewport, lead day and start date; each panel is {dataset, variable, mode}.
// Modes are field and first-class difference (A − B in a diverging colormap centred on
// 0). Currents are a variable (speed magnitude √(u²+v²)) with an optional particle
// animation. In single-panel mode a field can A/B compare a second forecast by a swipe
// divider or a blink key. Insight overlays (eddy census, Class-4 obs error)
// attach as purpose-modes, never all at once. A context rail carries
// the quantitative curves (skill vs lead, PSD spectrum) for the active view. Every bit
// of view state lives in the URL hash.

import { loadStore, loadManifest, readLayer, readLayerWindow, readCoordinate, readRootCoordinate, readColumn, prefetchLayer, isLayerCached } from "./modules/zarr.js";
import { COLORMAP_NAMES, DIVERGING } from "./vendor/cmocean/colormaps.js";
import {
  fieldToImageData,
  landStencilImageData,
  areaWeightedMean,
  symmetricRange,
  differenceField,
  resampleOntoGrid,
  drawColorbar,
  landColor,
  noObsColor,
  formatFixed,
} from "./modules/render.js";
import { startParticleField, makeVelocitySampler, speedMagnitudeField } from "./modules/particles.js";
import {
  loadInsightIndex,
  loadEddies,
  loadScoresSummary,
  loadRmsdByDepth,
  rmsdDepthProfile,
  loadClass4,
  prefetchClass4,
  cancelClass4Prefetches,
  insightsFor,
  eddyCensus,
  alignedEddyCensuses,
  class4Points,
  class4ParquetVariable,
} from "./modules/insights.js";
import {
  drawEddyDetections,
  matchCensuses,
  drawClass4Points,
  drawClass4Screen,
  class4AbsoluteError,
  class4ErrorScale,
  numericOrNaN,
  EDDY_MATCHED_COLOR,
  CLASS4_COLORMAP,
} from "./modules/overlays.js";
import { leadCurveSVG, psdSpectraSVG, rmsdByStartSVG, rmsdByDepthSVG, columnProfileSVG, SERIES_COLORS } from "./modules/charts.js";
import {
  loadYearGeography,
  loadYearRmsd,
  yearVariableMapping,
  buildYearGeographyField,
  buildYearBiasField,
  buildYearObservationCounts,
  buildYearBiasStandardError,
  yearGeographyMax,
  yearBiasMax,
  yearRmsdSeries,
  yearRmsdSeriesMax,
} from "./modules/year.js";
import { attachMethodNote, attachEddyMethodNote } from "./modules/method-popover.js";
import { boxPowerSpectrum, differenceBoxSpectrum } from "./modules/psd.js";
import { TRAJECTORY_COLORS, trajectorySeparationSVG } from "./modules/trajectories.js";
import { forecastColor } from "./modules/forecast-colors.js";
import { resolveViewerDataUrl, resolveColumnStoreUrl, initializeViewerConfig } from "./config.js";

// Resolved lazily: the data root is only final after initializeViewerConfig() has had a
// chance to apply an optional viewer-config.json.
const DATASETS_PATH = "./data/datasets.json";
const DIFFERENCE_COLORMAP = "balance";
const SPEED_COLORMAP = "speed";
const YEAR_ERROR_COLORMAP = "dense"; // sequential map for time-mean |obs − model|
const CURRENTS_MAX_SPEED = 1.2; // m/s mapping to the top of the speed colormap
const REGION_BOUNDS = {
  ibi: { west: -19.08, east: 5.08, south: 26.17, north: 56.08 },
};
// The global default view is Pacific-centred (central meridian 180°), matching the
// OceanBench site globe (region-globe.js rotates the global view to [180, 0, 0]).
// centerNX = (lon + 180) / 360, so lon 180° ≡ nx 1.0 ≡ nx 0.0 after the periodic
// wrap the global map applies. Rendering already tiles wrapped longitude copies.
const GLOBAL_DEFAULT_CENTER_NX = 0.0;
const PARTICLE_MAGNITUDE_SCALE = 1.0;
const CLASS4_DISPLAY_POINT_BUDGET = 18000;
const CLASS4_FULL_DENSITY_ZOOM = 12;

// Currents are a synthetic variable (speed magnitude √(u²+v²)) built from the u/v
// velocity components, one per available depth, so they sit in the variable dropdown
// like any other channel. The particle animation is an optional overlay on top.
const CURRENTS_VARIABLE_SURFACE = "current_speed";
const CURRENTS_VARIABLE_15M = "current_speed_15m";

function isCurrentsVariable(key) {
  return key === CURRENTS_VARIABLE_SURFACE || key === CURRENTS_VARIABLE_15M;
}

function currentsVariableDepth(key) {
  return key === CURRENTS_VARIABLE_15M ? "15m" : "surface";
}

// Human-facing depth label ("15 m" with a thin space), separate from the "15m" token
// used for depth-bin logic elsewhere.
function currentsDepthLabel(key) {
  return key === CURRENTS_VARIABLE_15M ? "15 m" : "surface";
}

// Class-4 current observations are surface drifters drogued at 15 m: obs and skill for
// velocities exist ONLY at the "15m" depth. A surface current selection (surface u, v,
// or derived surface current_speed) therefore has no honest obs to compare against.
function isVelocityFamilyVariable(key) {
  return isCurrentsVariable(key) || String(key).includes("sea_water_velocity");
}

function isSurfaceCurrentVariable(key) {
  return isVelocityFamilyVariable(key) && !String(key).endsWith("_15m");
}

// Matching 15 m variable for a surface current selection (u→u_15m, current_speed→…_15m).
function matching15mCurrentVariable(key) {
  return isSurfaceCurrentVariable(key) ? `${key}_15m` : key;
}

function syntheticCurrentsEntry(key) {
  return {
    standard_name: "sea_water_speed",
    units: "m/s",
    depth: currentsVariableDepth(key),
    default_colormap: SPEED_COLORMAP,
    default_range: [0, CURRENTS_MAX_SPEED],
  };
}

// Real manifest entry, or a synthetic descriptor for the currents variables.
function variableEntry(manifest, key) {
  if (isCurrentsVariable(key)) return syntheticCurrentsEntry(key);
  return manifest && manifest.variables[key];
}

function variableExists(manifest, key) {
  if (isCurrentsVariable(key)) return currentsVariableOptions(manifest).some((option) => option.value === key);
  return Boolean(manifest && key in manifest.variables);
}

// Currents variable options available for this manifest, gated on the u/v components.
function currentsVariableOptions(manifest) {
  if (!manifest || !manifest.variables) return [];
  const options = [];
  if ("eastward_sea_water_velocity" in manifest.variables && "northward_sea_water_velocity" in manifest.variables) {
    options.push({ value: CURRENTS_VARIABLE_SURFACE, label: "Currents · surface" });
  }
  if ("eastward_sea_water_velocity_15m" in manifest.variables && "northward_sea_water_velocity_15m" in manifest.variables) {
    options.push({ value: CURRENTS_VARIABLE_15M, label: "Currents · 15 m" });
  }
  return options;
}

// Shared state — linked across every panel (contracts.md §6).
const view = { zoom: 1, centerNX: GLOBAL_DEFAULT_CENTER_NX, centerNY: 0.5 };
const DEFAULT_LAYOUT = { controlsWidth: 256, railWidth: 352 };
const savedLayout = JSON.parse(localStorage.getItem("oceanbench.viewer.layout") || "null") || {};
const shared = {
  startIndex: 0,
  leadDay: 1,
  theme: "light",
  layout: 1,
  // "single" = per-start-date fields (the default view); "year" = precomputed
  // whole-year error-geography raster + RMSD-by-start diagnostics.
  scope: "single",
  // Year-scope map metric: "error" = time-mean |obs − model| (sequential), "bias" =
  // time-mean signed model − obs (diverging, centred 0). Single-forecast scope ignores it.
  yearMetric: "error",
  // PSD rectangle tool: { lon, lat, w, h } in degrees (centre + size). Disabled by
  // default so the initial map stays clean; enabling creates a centred default box.
  psdEnabled: false,
  psdBox: null,
  // The size the user asked for, kept apart from the clamped size actually drawn: a
  // second forecast can raise the minimum box size, and the box must shrink back to the
  // requested size once that constraint goes away. { w, h } in degrees; null when unset.
  psdBoxRequest: null,
  overlayMode: "none",
  // Water-column click point { lon, lat } (profile-on-click mode); null when unset. Kept
  // in the hash so a link reproduces the clicked profile.
  columnPoint: null,
  region: "global",
  eddyReference: "glorys",
  showParticles: true,
  particleSpeed: 1,
  railCollapsed: localStorage.getItem("oceanbench.viewer.railCollapsed") === "1",
  controlsCollapsed: localStorage.getItem("oceanbench.viewer.controlsCollapsed") === "1",
  controlsWidth: Number(savedLayout.controlsWidth) || DEFAULT_LAYOUT.controlsWidth,
  railWidth: Number(savedLayout.railWidth) || Number(localStorage.getItem("oceanbench.viewer.railWidth")) || DEFAULT_LAYOUT.railWidth,
  // 2-forecast display: "side" (two panels) or "swipe" (one map, F1 left / F2 right).
  displayMode: "side",
  // Which forecast the rail shows when 2 forecasts carry different variables.
  railForecast: 0,
};

const stores = new Map();
const manifests = new Map();
const coordinatesByLevel = new Map();
let datasetCatalog = [];
let insightIndex = null;
let scoresSummary = [];
// scores-summary.json is a multi-megabyte aggregate that only the context rail reads,
// so it stays off the boot path: the map paints first and the rail pulls it in on its
// first update. Memoised, so later rail updates cost nothing.
let scoresSummaryPromise = null;
function ensureScoresSummary() {
  if (!scoresSummaryPromise) {
    scoresSummaryPromise = loadScoresSummary(insightIndex).then((rows) => {
      scoresSummary = rows;
      return rows;
    });
  }
  return scoresSummaryPromise;
}
const panels = [];
let activePanelIndex = 0;
const elements = {};
const trajectoryWorker = new Worker(new URL("./modules/trajectory-worker.js", import.meta.url), { type: "module" });
let trajectoryState = null;
let trajectoryRequestId = 0;
// Water-column stores: slug -> store handle, or false when none is published (404 on
// .zmetadata) so the profile-on-click mode degrades gracefully to "not offered".
const columnStores = new Map();
// Decoded water column at the last clicked point: all leads/depths held in memory so
// the lead slider re-renders the profile with NO new fetch. Null when nothing is shown.
let columnProfile = null;
let columnRequestId = 0;

// ---- store / manifest / coordinate helpers (shared cache) -------------------

async function ensureStore(slug) {
  if (stores.has(slug)) return stores.get(slug);
  const descriptor = datasetCatalog.find((entry) => entry.slug === slug);
  if (!descriptor) throw new Error(`Unknown dataset ${slug}`);
  const [store, manifest] = await Promise.all([
    loadStore(resolveViewerDataUrl(descriptor.store)),
    loadManifest(resolveViewerDataUrl(descriptor.manifest)),
  ]);
  stores.set(slug, store);
  manifests.set(slug, manifest);
  return store;
}

function manifestFor(slug) {
  return manifests.get(slug);
}

async function loadCoordinates(slug, level) {
  const key = `${slug}/${level}`;
  if (coordinatesByLevel.has(key)) return coordinatesByLevel.get(key);
  const store = stores.get(slug);
  const [latitudes, longitudes] = await Promise.all([
    readCoordinate(store, level, "latitude"),
    readCoordinate(store, level, "longitude"),
  ]);
  const record = { latitudes, longitudes };
  coordinatesByLevel.set(key, record);
  return record;
}

function labelFor(slug) {
  const descriptor = datasetCatalog.find((entry) => entry.slug === slug);
  return descriptor ? descriptor.label : slug;
}

function scoreProductKey(slug) {
  if (slug === "glorys_one_degree") return "glorys";
  if (slug === "glo12_one_degree") return "glo12";
  return slug;
}

function prettyName(standardName) {
  return standardName.replace(/_/g, " ").replace(/\b\w/g, (character) => character.toUpperCase());
}

function variableLabel(manifest, key) {
  const entry = manifest.variables[key];
  return `${prettyName(entry.standard_name)} · ${entry.depth}`;
}

// Geographic world coordinates: nx = (lon+180)/360, ny = (90-lat)/180 (north-up).
function worldEdges(latitudes, longitudes) {
  const lonStep = longitudes.length > 1 ? Math.abs(longitudes[1] - longitudes[0]) : 1;
  const latStep = latitudes.length > 1 ? Math.abs(latitudes[1] - latitudes[0]) : 1;
  const lonMin = Math.min(longitudes[0], longitudes[longitudes.length - 1]);
  const lonMax = Math.max(longitudes[0], longitudes[longitudes.length - 1]);
  const latMin = Math.min(latitudes[0], latitudes[latitudes.length - 1]);
  const latMax = Math.max(latitudes[0], latitudes[latitudes.length - 1]);
  return {
    nx0: (lonMin - lonStep / 2 + 180) / 360,
    nx1: (lonMax + lonStep / 2 + 180) / 360,
    nyTop: (90 - (latMax + latStep / 2)) / 180,
    nyBottom: (90 - (latMin - latStep / 2)) / 180,
  };
}

// ---- panel construction -----------------------------------------------------

// Model + resolution of a catalog entry. The catalog carries no explicit resolution
// field, but every label is "MODEL (resolution)" (e.g. "GLO12 (1/12°)"), so read the
// pair from there and fall back to the slug when a label does not follow that shape.
function datasetIdentity(entry) {
  const match = /^(.*?)\s*\(([^()]*)\)\s*$/.exec(entry.label || "");
  if (match) return { model: match[1].trim().toLowerCase(), resolution: match[2].trim() };
  const model = entry.slug.replace(/_(one|1)_degree$/, "");
  return { model, resolution: model === entry.slug ? "native" : "1°" };
}

// Default partner for a forecast panel. Prefer a DIFFERENT model at the SAME resolution:
// the same model's 1-degree twin compares a model with itself, and the mixed resolutions
// make the shared PSD box degenerate ("Resolutions too different"). Fall back to any
// different model, then to the first different slug.
function otherSlug(slug) {
  const self = datasetCatalog.find((entry) => entry.slug === slug);
  const candidates = self
    ? datasetCatalog.filter((entry) => entry.slug !== slug && datasetIdentity(entry).model !== datasetIdentity(self).model)
    : [];
  const sameResolution = candidates.find(
    (entry) => datasetIdentity(entry).resolution === datasetIdentity(self).resolution,
  );
  return (
    sameResolution?.slug || candidates[0]?.slug || datasetCatalog.find((entry) => entry.slug !== slug)?.slug || slug
  );
}

function defaultPanelState(index) {
  // Forecast 1's dataset when it already exists, so a 1 -> 2 switch pairs against what
  // is actually on screen rather than against the first catalog entry.
  const first = panels[0]?.state?.dataset || datasetCatalog[0].slug;
  const second = otherSlug(first);
  const dataset = index === 1 ? second : first;
  return {
    dataset,
    variable: "sea_surface_height_above_geoid",
    colormap: null,
  };
}

function buildPanel(index) {
  const container = document.createElement("div");
  container.className = "panel";
  container.dataset.index = String(index);
  container.style.setProperty("--forecast-color", forecastColor(index));
  container.innerHTML = `
    <div class="panel-head">
      <span class="panel-forecast-label">Forecast ${index + 1}</span>
      <select class="panel-dataset" aria-label="Panel dataset"></select>
      <select class="panel-variable" aria-label="Panel variable"></select>
      <span class="spacer"></span>
      <span class="panel-badge"></span>
    </div>
    <div class="panel-canvas-wrap">
      <canvas class="panel-field"></canvas>
      <canvas class="panel-particles"></canvas>
      <canvas class="panel-overlay"></canvas>
      <div class="panel-readout" role="status"></div>
      <div class="panel-ghost-cursor" aria-hidden="true" hidden></div>
      <div class="panel-loading" hidden>Loading dataset…</div>
      <div class="panel-swipe-hint" hidden></div>
    </div>`;
  const panel = {
    index,
    container,
    state: defaultPanelState(index),
    els: {
      dataset: container.querySelector(".panel-dataset"),
      variable: container.querySelector(".panel-variable"),
      badge: container.querySelector(".panel-badge"),
      wrap: container.querySelector(".panel-canvas-wrap"),
      field: container.querySelector(".panel-field"),
      particles: container.querySelector(".panel-particles"),
      overlay: container.querySelector(".panel-overlay"),
      readout: container.querySelector(".panel-readout"),
      ghostCursor: container.querySelector(".panel-ghost-cursor"),
      loading: container.querySelector(".panel-loading"),
      swipeHint: container.querySelector(".panel-swipe-hint"),
    },
    offscreenA: null,
    offscreenB: null,
    landStencil: null,
    edgesA: null,
    edgesB: null,
    field: null,
    latitudes: null,
    longitudes: null,
    range: null,
    colormap: null,
    units: "",
    label: "",
    particleHandle: null,
    particleContext: null,
    swipeX: 0.5,
    renderToken: 0,
    renderAbort: null,
    loadingTimer: null,
    dragging: null,
    draggingSwipe: false,
  };
  wirePanel(panel);
  return panel;
}

function populateSelect(select, options, selectedValue) {
  select.innerHTML = "";
  for (const option of options) {
    const element = document.createElement("option");
    element.value = option.value;
    element.textContent = option.label;
    if (String(option.value) === String(selectedValue)) element.selected = true;
    select.appendChild(element);
  }
}

function refreshPanelControls(panel) {
  populateSelect(
    panel.els.dataset,
    datasetCatalog.map((entry) => ({ value: entry.slug, label: entry.label })),
    panel.state.dataset,
  );
  const manifest = manifestFor(panel.state.dataset);
  if (manifest) {
    if (!variableExists(manifest, panel.state.variable)) panel.state.variable = Object.keys(manifest.variables)[0];
    const options = Object.keys(manifest.variables).map((key) => ({ value: key, label: variableLabel(manifest, key) }));
    populateSelect(panel.els.variable, options.concat(currentsVariableOptions(manifest)), panel.state.variable);
  }
}

function wirePanel(panel) {
  panel.container.addEventListener("mousedown", () => setActivePanel(panel.index), true);
    panel.els.dataset.addEventListener("change", async (event) => {
    clearTrajectories();
    panel.state.dataset = event.target.value;
    setPanelLoading(panel, true);
    try {
      await ensureStore(panel.state.dataset);
      const manifest = manifestFor(panel.state.dataset);
      // Preserve the current selection across a dataset switch: variables are keyed by
      // standard name + depth, so the same channel (incl. derived currents) carries over
      // when the new dataset has it. Only fall back — with a brief note — when it does not.
      // Lead day, zoom/pan, purpose mode, region and scope live in shared/view state and
      // are untouched here, so they are preserved automatically.
      let fallbackNote = "";
      if (!variableExists(manifest, panel.state.variable)) {
        const previousVariable = panel.state.variable;
        const fallback = Object.keys(manifest.variables)[0];
        panel.state.variable = fallback;
        const fallbackLabel = manifest.variables[fallback] ? prettyName(manifest.variables[fallback].standard_name) : fallback;
        fallbackNote = `${prettyName(previousVariable)} is not available for ${labelFor(panel.state.dataset)}, showing ${fallbackLabel} instead`;
      }
      updateSharedTimeControls(manifest);
      refreshPanelControls(panel);
      setActivePanel(panel.index);
      await renderPanel(panel);
      if (isDiffView() && panel.index === 1) await renderPanel(panels[0]);
      // Reload overlays through applyOverlayMode so the overlay note is refreshed too:
      // switching to a dataset without published match-ups must flip the note to the
      // quiet "not published" message instead of leaving the previous dataset's note.
      await applyOverlayMode();
      // A different challenger may or may not publish a column store; re-probe, then
      // re-read the clicked column against the new challenger if the mode stays active.
      await updateColumnModeAvailability();
      if (columnModeActive() && shared.columnPoint) await readColumnProfileAt(shared.columnPoint.lon, shared.columnPoint.lat);
      // renderPanel clears the status on success, so surface the fallback note afterwards.
      if (fallbackNote) setStatus(fallbackNote);
      writeHash();
    } catch (error) {
      setStatus(String(error.message || error), true);
      console.error(error);
    } finally {
      setPanelLoading(panel, false);
    }
  });
  panel.els.variable.addEventListener("change", async (event) => {
    // Trajectories are advected from u/v, not from the displayed variable, so they stay.
    panel.state.variable = event.target.value;
    setActivePanel(panel.index);
    refreshPanelControls(panel);
    await renderPanel(panel);
    if (isDiffView() && panel.index === 1) await renderPanel(panels[0]);
    await updateContextRail();
    updateCurrentsControlVisibility();
    // The column variable follows the panel variable (temperature/salinity); re-read.
    if (columnModeActive() && shared.columnPoint) await readColumnProfileAt(shared.columnPoint.lon, shared.columnPoint.lat);
    writeHash();
  });
  const field = panel.els.field;
  field.addEventListener("pointerdown", (event) => beginPanelDrag(panel, event));
  field.addEventListener("wheel", (event) => onPanelWheel(panel, event), { passive: false });
  field.addEventListener("mouseleave", () => {
    panel.els.readout.textContent = "";
    panel.els.field.style.cursor = "";
    for (const other of panels) {
      if (other && other.els.ghostCursor) other.els.ghostCursor.hidden = true;
    }
  });
  // Tap/click also populates the consolidated readout pill (touch has no hover).
  field.addEventListener("click", (event) => updateHover(event));
  // Double-click resets zoom/pan, like the drawer splitters reset their width. The
  // browser fires the click gestures first, so a double-click in trajectories or column
  // mode re-seeds or re-reads the same point before the reset: the same work twice at
  // the same coordinates, which is idempotent, so it is left undebounced.
  field.addEventListener("dblclick", () => resetZoomPan());
}

// ---- rendering a single panel ----------------------------------------------

function currentDepthVariables(panel) {
  const is15m = panel.state.variable.endsWith("_15m");
  return {
    u: is15m ? "eastward_sea_water_velocity_15m" : "eastward_sea_water_velocity",
    v: is15m ? "northward_sea_water_velocity_15m" : "northward_sea_water_velocity",
    depth: is15m ? "15m" : "surface",
  };
}

function selectRenderLevel(manifest) {
  const levels = [...manifest.levels].sort((a, b) => a.cell_size_deg - b.cell_size_deg);
  const finest = levels[0];
  const targetCellSize = finest.cell_size_deg * Math.max(1, 24 / view.zoom);
  return levels.findLast((level) => level.cell_size_deg <= targetCellSize)?.level ?? levels[levels.length - 1].level;
}

function renderLevelForSlug(slug) {
  const manifest = manifestFor(slug);
  if (!manifest) return null;
  return selectRenderLevel(manifest);
}

// Whether a panel could draw this lead day without waiting on the network: every tile
// the render would read is already decoded in the store's chunk cache, either from an
// earlier render or from the neighbour prefetch. The lead slider asks this to decide
// between painting inside the frame and waiting for the fetch debounce.
function panelLeadCached(panel, leadDay) {
  // Year scope reads the memoised insight artifacts, never a lead tile.
  if (shared.scope === "year") return true;
  const manifest = manifestFor(panel.state.dataset);
  const store = stores.get(panel.state.dataset);
  if (!manifest || !store) return false;
  const level = selectRenderLevel(manifest);
  const startIndex = Math.min(shared.startIndex, manifest.start_dates.length - 1);
  const leadIndex = leadDay - 1;
  const currents = isCurrentsVariable(panel.state.variable);
  const variables = currents
    ? [currentDepthVariables(panel).u, currentDepthVariables(panel).v]
    : [panel.state.variable];
  for (const variable of variables) {
    if (!isLayerCached(store, { variable, level, startIndex, leadIndex })) return false;
  }
  // The difference view reads the partner forecast's field on the same slice.
  if (isDiffHost(panel) && !currents) {
    const partner = stores.get(panels[1].state.dataset);
    if (!partner) return false;
    if (!isLayerCached(partner, { variable: panel.state.variable, level, startIndex, leadIndex })) return false;
  }
  return true;
}

function leadFramesCached(leadDay) {
  for (let i = 0; i < shared.layout; i += 1) {
    if (!panelLeadCached(panels[i], leadDay)) return false;
  }
  return true;
}

async function renderPanel(panel) {
  const token = ++panel.renderToken;
  // A superseded render must stop paying for tiles it will never draw: abort the
  // previous render's in-flight chunk fetches before starting this one.
  if (panel.renderAbort) panel.renderAbort.abort();
  panel.renderAbort = new AbortController();
  setPanelLoading(panel, true);
  try {
    await ensureStore(panel.state.dataset);
    const manifest = manifestFor(panel.state.dataset);
    if (!variableExists(manifest, panel.state.variable)) panel.state.variable = Object.keys(manifest.variables)[0];
    // A panel can be shown before its manifest is cached (e.g. switching to 2 forecasts
    // warms only the initially visible panels). refreshPanelControls only fills the
    // variable select when the manifest is present, so re-run it here — now that the
    // store is loaded — to populate the (otherwise empty) select. Idempotent and cheap.
    refreshPanelControls(panel);

    if (shared.scope === "year") {
      await renderYearPanel(panel, token, manifest);
      if (token !== panel.renderToken) return;
      resizePanelCanvases(panel);
      drawPanel(panel);
      updateSharedColorbar();
      setStatus("");
      return;
    }

    const level = selectRenderLevel(manifest);
    panel.renderedLevel = level; // the pyramid level the displayed field came from
    const start = Math.min(shared.startIndex, manifest.start_dates.length - 1);
    const leadIndex = shared.leadDay - 1;

    if (isDiffHost(panel) && !isCurrentsVariable(panel.state.variable)) {
      await renderDifferencePanel(panel, token, manifest, level, start, leadIndex, panels[1].state.dataset);
    } else if (isCurrentsVariable(panel.state.variable)) {
      await renderCurrentsPanel(panel, token, manifest, level, start, leadIndex);
    } else {
      await renderFieldPanel(panel, token, manifest, level, start, leadIndex);
    }
    if (token !== panel.renderToken) return;
    resizePanelCanvases(panel);
    drawPanel(panel);
    drawOverlays(panel);
    drawTrajectoryFans(panel);
    updateSharedColorbar();
    setStatus("");
  } catch (error) {
    if (error && error.name === "AbortError") return;
    if (token === panel.renderToken) setStatus(String(error.message || error), true);
    console.error(error);
  } finally {
    if (token === panel.renderToken) setPanelLoading(panel, false);
  }
}

async function readAlignedField(panel, sourceSlug, variable, level, start, leadIndex, targetLat, targetLon) {
  await ensureStore(sourceSlug);
  const signal = panel.renderAbort ? panel.renderAbort.signal : undefined;
  const layer = await readLayer(stores.get(sourceSlug), { variable, level, startIndex: start, leadIndex, signal });
  const coordinates = await loadCoordinates(sourceSlug, level);
  if (!targetLat) return { field: layer, latitudes: coordinates.latitudes, longitudes: coordinates.longitudes };
  const aligned = resampleOntoGrid(layer, coordinates.latitudes, coordinates.longitudes, targetLat, targetLon);
  return { field: aligned, latitudes: targetLat, longitudes: targetLon, compressedBytes: layer.compressedBytes };
}

// True when more than one panel is shown and every shown panel holds this panel's
// variable — the condition under which same-variable panels share one colour scale (and
// the field colorbar is labelled "(shared)"). Mirrors the colorbar's `sameVariable` test.
function shownPanelsShareVariable(panel) {
  if (shared.layout <= 1) return false;
  for (let i = 0; i < shared.layout; i += 1) {
    if (!panels[i] || panels[i].state.variable !== panel.state.variable) return false;
  }
  return true;
}

// Combine one or more [low, high] variable ranges into the range a field is colorized
// with. A diverging colormap (balance/delta) is centred on physical zero — [-M, +M],
// M = max|bound| — so its neutral colour reads as 0; a sequential map keeps the data
// bounds ([min low, max high]).
function combineFieldRange(ranges, diverging) {
  if (diverging) {
    let magnitude = 0;
    for (const [low, high] of ranges) magnitude = Math.max(magnitude, Math.abs(low), Math.abs(high));
    const bound = magnitude || 1;
    return [-bound, bound];
  }
  let low = Infinity;
  let high = -Infinity;
  for (const [rangeLow, rangeHigh] of ranges) {
    if (rangeLow < low) low = rangeLow;
    if (rangeHigh > high) high = rangeHigh;
  }
  return [Number.isFinite(low) ? low : 0, Number.isFinite(high) ? high : 1];
}

// The value range a field panel is colorized with. Diverging maps are zero-centred (see
// combineFieldRange). When several panels show the same variable, the range spans all of
// them so identical physical values render as identical colours and the "(shared)" label
// is truthful. ensureStore is memoised, so awaiting the sibling manifests is cheap and
// makes the shared range deterministic regardless of which panel renders first.
async function fieldColorRange(panel, colormap, defaultRange) {
  const diverging = DIVERGING.has(colormap);
  const ranges = [defaultRange];
  if (shownPanelsShareVariable(panel)) {
    for (let i = 0; i < shared.layout; i += 1) {
      const candidate = panels[i];
      if (!candidate || candidate === panel) continue;
      await ensureStore(candidate.state.dataset);
      const otherEntry = manifestFor(candidate.state.dataset).variables[candidate.state.variable];
      if (otherEntry && otherEntry.default_range) ranges.push(otherEntry.default_range);
    }
  }
  return combineFieldRange(ranges, diverging);
}

async function renderFieldPanel(panel, token, manifest, level, start, leadIndex) {
  const entry = manifest.variables[panel.state.variable];
  const primary = await readAlignedField(panel, panel.state.dataset, panel.state.variable, level, start, leadIndex);
  if (token !== panel.renderToken) return;
  const colormap = panel.state.colormap || entry.default_colormap;
  const range = await fieldColorRange(panel, colormap, entry.default_range);
  if (token !== panel.renderToken) return;
  panel.field = primary.field;
  clearYearReadoutMetadata(panel);
  panel.latitudes = primary.latitudes;
  panel.longitudes = primary.longitudes;
  panel.edgesA = worldEdges(primary.latitudes, primary.longitudes);
  panel.offscreenA = colorize(primary.field, primary.latitudes, colormap, range);
  panel.offscreenB = null;
  panel.range = range;
  panel.colormap = colormap;
  panel.units = entry.units;
  panel.label = `${labelFor(panel.state.dataset)} · ${prettyName(entry.standard_name)}`;
  stopParticles(panel);
  prefetchNeighbours(panel, level, start, leadIndex);
}

async function renderDifferencePanel(panel, token, manifest, level, start, leadIndex, compareSlug) {
  const entry = manifest.variables[panel.state.variable];
  const primary = await readAlignedField(panel, panel.state.dataset, panel.state.variable, level, start, leadIndex);
  if (token !== panel.renderToken) return;
  await ensureStore(compareSlug);
  const compareLevel = renderLevelForSlug(compareSlug);
  const compare = await readAlignedField(
    panel,
    compareSlug,
    panel.state.variable,
    compareLevel,
    start,
    leadIndex,
    primary.latitudes,
    primary.longitudes,
  );
  if (token !== panel.renderToken) return;
  const difference = differenceField(primary.field, compare.field);
  const range = symmetricRange(difference);
  panel.field = difference;
  clearYearReadoutMetadata(panel);
  panel.latitudes = primary.latitudes;
  panel.longitudes = primary.longitudes;
  panel.edgesA = worldEdges(primary.latitudes, primary.longitudes);
  panel.offscreenA = colorize(difference, primary.latitudes, DIFFERENCE_COLORMAP, range);
  panel.offscreenB = null;
  panel.range = range;
  panel.colormap = DIFFERENCE_COLORMAP;
  panel.units = entry.units;
  panel.label = `${labelFor(panel.state.dataset)} − ${labelFor(compareSlug)} · ${prettyName(entry.standard_name)}`;
  stopParticles(panel);
  prefetchNeighbours(panel, level, start, leadIndex);
}

async function renderCurrentsPanel(panel, token, manifest, level, start, leadIndex) {
  const variables = currentDepthVariables(panel);
  if (!(variables.u in manifest.variables)) {
    setStatus(`${labelFor(panel.state.dataset)} has no velocity fields for currents`, true);
    return;
  }
  const uPrimary = await readAlignedField(panel, panel.state.dataset, variables.u, level, start, leadIndex);
  if (token !== panel.renderToken) return;
  const vPrimary = await readAlignedField(panel, panel.state.dataset, variables.v, level, start, leadIndex);
  if (token !== panel.renderToken) return;
  const speed = speedMagnitudeField(uPrimary.field, vPrimary.field);
  const range = [0, CURRENTS_MAX_SPEED];
  panel.field = speed;
  clearYearReadoutMetadata(panel);
  panel.latitudes = uPrimary.latitudes;
  panel.longitudes = uPrimary.longitudes;
  panel.edgesA = worldEdges(uPrimary.latitudes, uPrimary.longitudes);
  panel.offscreenA = colorize(speed, uPrimary.latitudes, SPEED_COLORMAP, range);
  panel.offscreenB = null;
  panel.landStencil = landStencil(speed, uPrimary.latitudes);
  panel.range = range;
  panel.colormap = SPEED_COLORMAP;
  panel.units = "m/s";
  panel.label = `${labelFor(panel.state.dataset)} · currents (${currentsDepthLabel(panel.state.variable)})`;
  panel.velocity = {
    sampler: makeVelocitySampler(uPrimary.field, vPrimary.field, uPrimary.latitudes, uPrimary.longitudes),
  };
  if (shared.showParticles) startPanelParticles(panel);
  else stopParticles(panel);
  prefetchNeighbours(panel, level, start, leadIndex);
}

function colorize(field, latitudes, colormap, range, transparentNaN = false, landMask = null) {
  const flip = latitudes[0] < latitudes[latitudes.length - 1];
  const image = fieldToImageData(field, colormap, range, {
    flipVertical: flip,
    theme: shared.theme,
    transparentNaN,
    landMask,
  });
  const canvas = new OffscreenCanvas(field.width, field.height);
  canvas.getContext("2d", { willReadFrequently: true }).putImageData(image, 0, 0);
  return canvas;
}

function landStencil(field, latitudes) {
  const flip = latitudes[0] < latitudes[latitudes.length - 1];
  const image = landStencilImageData(field, { flipVertical: flip });
  const canvas = new OffscreenCanvas(field.width, field.height);
  canvas.getContext("2d").putImageData(image, 0, 0);
  return canvas;
}

// ---- entire-year scope: precomputed error geography raster ------------------

function clearYearPanel(panel, note) {
  panel.field = null;
  panel.offscreenA = null;
  panel.offscreenB = null;
  panel.range = null;
  panel.units = "";
  panel.label = "";
  panel.yearMeta = null;
  panel.yearCounts = null;
  panel.yearBiasSE = null;
  panel.yearMissing = note;
}

function clearYearReadoutMetadata(panel) {
  panel.yearMeta = null;
  panel.yearCounts = null;
  panel.yearBiasSE = null;
  panel.yearMetric = null;
}

// Shared [0, max] scale across the visible panels that show the same year variable,
// so two-forecast rasters stay directly comparable.
async function sharedYearRange(shortName, leadDay) {
  let maximum = 0;
  for (let i = 0; i < shared.layout; i += 1) {
    const candidate = panels[i];
    if (!candidate) continue;
    const mapping = yearVariableMapping(candidate.state.variable);
    if (!mapping || mapping.short !== shortName) continue;
    const url = insightsFor(insightIndex, candidate.state.dataset, shared.region).year_error_geography;
    if (!url) continue;
    const geography = await loadYearGeography(url);
    if (!geography) continue;
    maximum = Math.max(maximum, yearGeographyMax(geography, shortName, leadDay));
  }
  return [0, maximum || 1];
}

// Symmetric [-M, +M] scale for the signed-bias raster, shared across the visible panels
// showing the same year variable so the diverging (balance) colormap stays centred on 0
// and directly comparable between two forecasts.
async function sharedYearBiasRange(shortName, leadDay) {
  let maximum = 0;
  for (let i = 0; i < shared.layout; i += 1) {
    const candidate = panels[i];
    if (!candidate) continue;
    const mapping = yearVariableMapping(candidate.state.variable);
    if (!mapping || mapping.short !== shortName) continue;
    const url = insightsFor(insightIndex, candidate.state.dataset, shared.region).year_error_geography;
    if (!url) continue;
    const geography = await loadYearGeography(url);
    if (!geography) continue;
    maximum = Math.max(maximum, yearBiasMax(geography, shortName, leadDay));
  }
  const bound = maximum || 1;
  return [-bound, bound];
}

async function renderYearPanel(panel, token, manifest) {
  stopParticles(panel);
  // The velocity error geography and RMSD-by-start are built from 15 m drifter obs.
  // A surface current selection cannot be honestly mapped onto them.
  if (isSurfaceCurrentVariable(panel.state.variable)) {
    clearYearPanel(panel, "Current observations (drifters) are measured at 15 m depth. Switch to 15 m currents to compare against them.");
    return;
  }
  const urls = insightsFor(insightIndex, panel.state.dataset, shared.region);
  const geoUrl = urls.year_error_geography;
  if (!geoUrl) {
    clearYearPanel(panel, "Year diagnostics not available for this dataset/region.");
    return;
  }
  const geography = await loadYearGeography(geoUrl);
  if (token !== panel.renderToken) return;
  const biasMode = shared.yearMetric === "bias";
  const mapping = geography ? yearVariableMapping(panel.state.variable) : null;
  const built = mapping
    ? biasMode
      ? buildYearBiasField(geography, mapping.short, shared.leadDay)
      : buildYearGeographyField(geography, mapping.short, shared.leadDay)
    : null;
  if (!geography || !mapping || !built) {
    // |error| must still work when bias is absent: the bias-specific note fires only when
    // the geography and variable/lead exist but carry no bias field for this selection.
    const biasAbsent = biasMode && geography && mapping && buildYearGeographyField(geography, mapping.short, shared.leadDay);
    clearYearPanel(
      panel,
      !geography
        ? "Year diagnostics not available for this dataset/region."
        : biasAbsent
          ? "Bias is not available for this dataset. A republish is pending."
          : biasMode
            ? "Bias not available for this variable at this lead."
            : "Year diagnostics not available for this variable at this lead.",
    );
    return;
  }
  const range = biasMode
    ? await sharedYearBiasRange(mapping.short, shared.leadDay)
    : await sharedYearRange(mapping.short, shared.leadDay);
  if (token !== panel.renderToken) return;
  // Separate land from unobserved ocean: both are NaN in the error raster, so derive a
  // land mask from the dataset's own coarsest pyramid level (tiny) and resample it onto
  // the raster grid. Land then renders in the field land colour, ocean-without-obs in a
  // faint tint — consistent with single-forecast rendering. A failed fetch degrades to
  // the previous transparent-NaN behaviour.
  const landMask = await yearLandMask(panel, built.latitudes, built.longitudes);
  if (token !== panel.renderToken) return;
  panel.field = built.field;
  panel.yearCounts = buildYearObservationCounts(geography, mapping.short, shared.leadDay, { bias: biasMode });
  // Per-cell bias standard error for the hover readout (bias mode only; null on old artifacts).
  panel.yearBiasSE = biasMode ? buildYearBiasStandardError(geography, mapping.short, shared.leadDay) : null;
  panel.latitudes = built.latitudes;
  panel.longitudes = built.longitudes;
  panel.edgesA = worldEdges(built.latitudes, built.longitudes);
  const colormap = biasMode ? DIFFERENCE_COLORMAP : YEAR_ERROR_COLORMAP;
  panel.offscreenA = landMask
    ? colorize(built.field, built.latitudes, colormap, range, false, landMask)
    : colorize(built.field, built.latitudes, colormap, range, true);
  panel.offscreenB = null;
  panel.range = range;
  panel.colormap = colormap;
  panel.units = mapping.unit;
  panel.yearMissing = null;
  panel.yearMetric = shared.yearMetric;
  panel.yearMeta = { nStarts: geography.meta && geography.meta.n_starts, component: mapping.component };
  const entry = variableEntry(manifest, panel.state.variable);
  const varLabel = entry ? prettyName(entry.standard_name) : panel.state.variable;
  panel.label =
    `${labelFor(panel.state.dataset)} · ${biasMode ? "bias" : "mean |obs − model|"} · ${varLabel}` +
    (mapping.component ? ` (${mapping.component})` : "");
}

const yearLandMaskCache = new Map();

// The raw variable whose land/ocean pattern defines the mask. Derived current speed
// has no store variable of its own, so borrow its eastward component.
function landMaskVariable(panel) {
  if (panel.state.variable === "current_speed" || panel.state.variable === "current_speed_15m") {
    return currentDepthVariables(panel).u;
  }
  return panel.state.variable;
}

async function yearLandMask(panel, latitudes, longitudes) {
  const slug = panel.state.dataset;
  const variable = landMaskVariable(panel);
  const key = `${slug}/${variable}/${shared.region}/${latitudes.length}x${longitudes.length}`;
  if (yearLandMaskCache.has(key)) return yearLandMaskCache.get(key);
  const promise = (async () => {
    try {
      await ensureStore(slug);
      const manifest = manifestFor(slug);
      if (!manifest || !Array.isArray(manifest.levels) || !manifest.levels.length) return null;
      if (!variableExists(manifest, variable)) return null;
      const levels = [...manifest.levels].sort((a, b) => a.cell_size_deg - b.cell_size_deg);
      const coarsest = levels[levels.length - 1].level;
      const layer = await readLayer(stores.get(slug), { variable, level: coarsest, startIndex: 0, leadIndex: 0 });
      const coordinates = await loadCoordinates(slug, coarsest);
      const resampled = resampleOntoGrid(layer, coordinates.latitudes, coordinates.longitudes, latitudes, longitudes);
      const mask = new Uint8Array(resampled.data.length);
      for (let i = 0; i < mask.length; i += 1) mask[i] = Number.isNaN(resampled.data[i]) ? 1 : 0;
      return mask;
    } catch (error) {
      console.warn("year land mask unavailable", error);
      return null;
    }
  })();
  yearLandMaskCache.set(key, promise);
  return promise;
}

function prefetchNeighbours(panel, level, start, leadIndex) {
  const manifest = manifestFor(panel.state.dataset);
  if (!manifest || !manifest.lead_days) return;
  const maxLead = Math.max(...manifest.lead_days);
  for (const delta of [1, -1]) {
    const lead = shared.leadDay + delta;
    if (lead < 1 || lead > maxLead) continue;
    if (isCurrentsVariable(panel.state.variable)) {
      const variables = currentDepthVariables(panel);
      prefetchLayer(stores.get(panel.state.dataset), { variable: variables.u, level, startIndex: start, leadIndex: lead - 1 });
      prefetchLayer(stores.get(panel.state.dataset), { variable: variables.v, level, startIndex: start, leadIndex: lead - 1 });
    } else {
      prefetchLayer(stores.get(panel.state.dataset), { variable: panel.state.variable, level, startIndex: start, leadIndex: lead - 1 });
    }
  }
}

// ---- projection + drawing ---------------------------------------------------

function projectionFor(panel) {
  const canvas = panel.els.field;
  const width = canvas.width;
  const height = canvas.height;
  const fit = Math.min(height, width / 2);
  const displayHeight = fit * view.zoom;
  const displayWidth = 2 * displayHeight;
  const originX = width / 2 - view.centerNX * displayWidth;
  const originY = height / 2 - view.centerNY * displayHeight;
  return {
    width,
    height,
    displayWidth,
    displayHeight,
    originX,
    originY,
    project: (nx, ny) => ({ x: originX + nx * displayWidth, y: originY + ny * displayHeight }),
    unproject: (x, y) => ({ nx: (x - originX) / displayWidth, ny: (y - originY) / displayHeight }),
  };
}

function drawImageWorld(context, offscreen, edges, projection) {
  const visibleLeft = projection.unproject(0, 0).nx;
  const visibleRight = projection.unproject(projection.width, 0).nx;
  const firstCopy = shared.region === "global" ? Math.floor(visibleLeft - edges.nx1) : 0;
  const lastCopy = shared.region === "global" ? Math.ceil(visibleRight - edges.nx0) : 0;
  // Snap every copy to whole device pixels. Copy k's right edge and copy k+1's left
  // edge are the same world coordinate, so rounding hands them the same integer and
  // the copies abut exactly. Drawn at fractional coordinates instead, each copy gets
  // an antialiased edge, and where the two meet the backdrop shows through as a
  // one-pixel column: the black line at the dateline in the default Pacific-centred
  // view. The blit alone is snapped; projection, hit-testing and overlays are
  // untouched, so nothing moves by more than the sub-pixel the rounding absorbs.
  const top = Math.round(projection.project(edges.nx0, edges.nyTop).y);
  const bottom = Math.round(projection.project(edges.nx0, edges.nyBottom).y);
  for (let copy = firstCopy; copy <= lastCopy; copy += 1) {
    const left = Math.round(projection.project(edges.nx0 + copy, edges.nyTop).x);
    const right = Math.round(projection.project(edges.nx1 + copy, edges.nyTop).x);
    context.drawImage(offscreen, left, top, right - left, bottom - top);
  }
}

function isSwipeHost(panel) {
  return shared.layout === 2 && shared.displayMode === "swipe" && panel.index === 0;
}

// The Difference comparison view exists only with two forecasts; it collapses to a
// single map (hosted by Forecast 1) showing Forecast 1 − Forecast 2.
function isDiffView() {
  return shared.layout === 2 && shared.displayMode === "diff";
}

function isDiffHost(panel) {
  return isDiffView() && panel.index === 0;
}

// The background forecast field is desaturated while a colored purpose overlay is active
// (class-4 obs, eddy census, trajectories) so the overlay colors stay legible. Field mode,
// the difference view, and the year raster keep their full-color scale.
function fieldMutedUnderOverlay() {
  if (shared.scope === "year") return false;
  if (isDiffView()) return false;
  return shared.overlayMode === "class4" || shared.overlayMode === "eddies" || shared.overlayMode === "trajectories";
}

// Theme colors drawn on canvas come from the shared design tokens (tokens.css,
// --ob-viewer-*, themed by data-theme on <html>); the per-theme literal is a
// fallback kept in sync with that file.
function themeToken(name, fallback) {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

function drawPanel(panel) {
  const canvas = panel.els.field;
  const context = canvas.getContext("2d");
  context.fillStyle = themeToken("--ob-viewer-canvas-bg", shared.theme === "light" ? "#eef2f6" : "#080b11");
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (!panel.offscreenA) {
    if (shared.scope === "year" && panel.yearMissing) {
      context.fillStyle = themeToken("--ob-viewer-canvas-note", shared.theme === "light" ? "#5b6675" : "#8b97a6");
      context.font = `${14 * (window.devicePixelRatio || 1)}px system-ui, sans-serif`;
      context.textAlign = "center";
      context.textBaseline = "middle";
      context.fillText(panel.yearMissing, canvas.width / 2, canvas.height / 2);
      panel.els.swipeHint.hidden = true;
    }
    updatePanelMethodNote(panel);
    return;
  }
  const projection = projectionFor(panel);
  context.imageSmoothingEnabled = false;

  // Under a colored purpose overlay (class-4 obs, eddies, trajectories) desaturate the
  // background field so the overlay owns the color channel and its points/contours read
  // clearly. Field mode and the difference view keep full color. Applied as a cheap
  // canvas filter on the field blit only — no re-colorize, no tile refetch; the hover
  // readout still reads the raw field values.
  const fieldFilter = fieldMutedUnderOverlay() ? "grayscale(1) contrast(0.9)" : "none";

  // In swipe display the single host panel (Forecast 1) overlays Forecast 2 on its
  // right side; take Forecast 2's coloured field straight from the second panel.
  if (isSwipeHost(panel)) {
    panel.offscreenB = panels[1] ? panels[1].offscreenA : null;
    panel.edgesB = panels[1] ? panels[1].edgesA : null;
  }

  if (isSwipeHost(panel) && panel.offscreenB) {
    context.filter = fieldFilter;
    drawImageWorld(context, panel.offscreenA, panel.edgesA, projection);
    const dividerX = panel.swipeX * canvas.width;
    context.save();
    context.beginPath();
    context.rect(dividerX, 0, canvas.width - dividerX, canvas.height);
    context.clip();
    drawImageWorld(context, panel.offscreenB, panel.edgesB, projection);
    context.restore();
    context.filter = "none";
    context.strokeStyle = themeToken("--ob-viewer-swipe-divider", shared.theme === "light" ? "#1f6feb" : "#38bdf8");
    context.lineWidth = 2 * (window.devicePixelRatio || 1);
    context.beginPath();
    context.moveTo(dividerX, 0);
    context.lineTo(dividerX, canvas.height);
    context.stroke();
    panel.els.swipeHint.hidden = false;
    panel.els.swipeHint.textContent = `◀ Forecast 1 · ${labelFor(panels[0].state.dataset)}  |  Forecast 2 · ${labelFor(panels[1].state.dataset)} ▶`;
  } else {
    context.filter = fieldFilter;
    drawImageWorld(context, panel.offscreenA, panel.edgesA, projection);
    context.filter = "none";
    panel.els.swipeHint.hidden = true;
  }
  // In year scope outline the raster extent so the map's data limits are legible at
  // zoom 1 — especially the regional (ibi) domain against the empty page outside it.
  if (shared.scope === "year" && panel.edgesA) drawRasterBorder(context, panel.edgesA, projection);
  updateParticleProjection(panel, projection);
  updatePanelBadge(panel);
  updatePanelMethodNote(panel);
}

// Attach the right "?" method note to a panel's header: the difference-view note on the
// diff host, the year error-geography note in year scope, otherwise the field-map note
// (with the panel's dataset label). Cleared and re-attached each draw so it tracks state.
function updatePanelMethodNote(panel) {
  const head = panel.container.querySelector(".panel-head");
  if (!head) return;
  head.querySelectorAll(".method-note-btn").forEach((button) => button.remove());
  if (shared.scope === "year") {
    if (!panel.yearMissing) attachMethodNote(head, "year-geography");
  } else if (isDiffHost(panel)) {
    attachMethodNote(head, "diff-view");
  } else {
    attachMethodNote(head, "field-map", { dataset: labelFor(panel.state.dataset) });
  }
}

function drawRasterBorder(context, edges, projection) {
  const ratio = window.devicePixelRatio || 1;
  const visibleLeft = projection.unproject(0, 0).nx;
  const visibleRight = projection.unproject(projection.width, 0).nx;
  const firstCopy = shared.region === "global" ? Math.floor(visibleLeft - edges.nx1) : 0;
  const lastCopy = shared.region === "global" ? Math.ceil(visibleRight - edges.nx0) : 0;
  context.save();
  context.strokeStyle = themeToken("--ob-viewer-raster-border", shared.theme === "light" ? "rgba(40, 52, 72, 0.5)" : "rgba(184, 200, 224, 0.5)");
  context.lineWidth = 1.5 * ratio;
  for (let copy = firstCopy; copy <= lastCopy; copy += 1) {
    const topLeft = projection.project(edges.nx0 + copy, edges.nyTop);
    const bottomRight = projection.project(edges.nx1 + copy, edges.nyBottom);
    if (shared.region === "global") {
      // Longitude wraps, so vertical edges would draw a spurious seam mid-ocean; only
      // the top/bottom latitude limits are real boundaries of the raster.
      context.beginPath();
      context.moveTo(topLeft.x, topLeft.y);
      context.lineTo(bottomRight.x, topLeft.y);
      context.moveTo(topLeft.x, bottomRight.y);
      context.lineTo(bottomRight.x, bottomRight.y);
      context.stroke();
    } else {
      context.strokeRect(topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
    }
  }
  context.restore();
}

function updatePanelBadge(panel) {
  const mean = panel.field ? areaWeightedMean(panel.field, panel.latitudes) : NaN;
  panel.els.badge.textContent = `area-weighted mean ${formatFixed(mean, 3)} ${panel.units}`;
}

// ---- overlays ---------------------------------------------------------------

let overlayData = { eddiesCensuses: [], eddiesMatch: null, class4: null, class4Error: null, region: null };
let redrawAllPanelsFrame = 0;

async function loadInsightManifest(url) {
  if (!url) return null;
  try {
    const response = await fetch(resolveViewerDataUrl(url), { cache: "no-cache" });
    if (!response.ok) return null;
    return response.json();
  } catch {
    return null;
  }
}

async function loadOverlayData() {
  const slug = panels[activePanelIndex] ? panels[activePanelIndex].state.dataset : datasetCatalog[0].slug;
  const region = shared.region;
  const urls = insightsFor(insightIndex, slug, region);
  overlayData.region = region;
  overlayData.eddiesCensuses = [];
  overlayData.eddiesMatch = null;
  overlayData.eddiesLeadMismatch = false;
  overlayData.class4 = null;
  overlayData.class4Error = null;
  overlayData.class4Unpublished = false;
  // Nothing may prefetch match-ups for an overlay that is no longer showing them.
  if (shared.overlayMode !== "class4") stopClass4Prefetch(null);
  if (shared.overlayMode === "eddies") {
    // Load each visible forecast's own eddy artifact and reduce it to a census. The
    // two forecasts come from the panel pickers; no dataset is a hardcoded truth.
    const eddiesByPanel = await Promise.all(
      panels.slice(0, shared.layout).map((panel) => {
        const panelUrls = insightsFor(insightIndex, panel.state.dataset, region);
        return loadEddies(panelUrls.eddies || null);
      }),
    );
    // Two forecasts are cross-matched only at a lead day BOTH publish: snap the requested
    // lead to the intersection of their available leads and read both censuses there, so a
    // wave challenger with a different lead set can never silently match different leads.
    // No shared lead → suppress the match and report both leads (see renderEddyLegend).
    if (shared.layout === 2 && eddiesByPanel[0] && eddiesByPanel[1]) {
      const aligned = await alignedEddyCensuses(eddiesByPanel[0], eddiesByPanel[1], shared.leadDay);
      overlayData.eddiesCensuses = aligned.censuses;
      overlayData.eddiesLeadMismatch = aligned.mismatch;
      overlayData.eddiesMatch =
        !aligned.mismatch && aligned.censuses[0] && aligned.censuses[1]
          ? matchCensuses(aligned.censuses[0].detections, aligned.censuses[1].detections)
          : null;
    } else {
      overlayData.eddiesCensuses = await Promise.all(eddiesByPanel.map((eddies) => eddyCensus(eddies, shared.leadDay)));
      overlayData.eddiesMatch = null;
    }
  } else if (shared.overlayMode === "class4") {
    const class4Url = urls.class4_matchups;
    // Reference datasets (and any dataset without published match-ups) carry an
    // explicit null class4_matchups in insights.json. That is a legitimate absence,
    // not a failure — show the honest "not published" empty state and paint no
    // points. We never substitute another model's obs: a fallback to some other
    // dataset's match-ups would paint provenance-mismatched points under this
    // dataset's name while the rail truthfully says it has none.
    if (!class4Url) {
      overlayData.class4Unpublished = true;
      return;
    }
    const manifest = await loadInsightManifest(urls.manifest);
    const class4Manifest = manifest && manifest["class4-matchups"];
    const request = {
      url: class4Url,
      byteLength: class4ByteLengthHint(class4Url, class4Manifest),
      startDate: currentStartDate(slug),
      leadDay: shared.leadDay,
      variables: class4RequestVariables(),
    };
    // The user is waiting now: stop any prefetch of another lead from holding connections
    // this read needs. A prefetch of this very pair is kept, it is the read we want.
    stopClass4Prefetch(request);
    try {
      overlayData.class4 = await loadClass4(request.url, { ...request, onProgress: showClass4Progress });
    } catch (error) {
      overlayData.class4 = null;
      overlayData.class4Error = error instanceof Error ? error.message : String(error);
    }
    if (overlayData.class4) scheduleClass4Prefetch(request, slug);
  }
}

// Scrubbing the lead slider is the common move, and each lead is its own download. Once the
// lead the user asked for has landed, read the neighbouring leads into the same worker cache
// while the connection is idle, so the next step of the scrub is already there. Two leads
// only: the pipe is shared with the map tiles and the origins throttle per connection.
const CLASS4_PREFETCH_LEADS = 2;
let class4PrefetchGeneration = 0;

function stopClass4Prefetch(keep) {
  class4PrefetchGeneration += 1;
  cancelClass4Prefetches(keep || null);
}

function scheduleClass4Prefetch(request, slug) {
  const leads = neighbouringLeads(request.leadDay, slug).slice(0, CLASS4_PREFETCH_LEADS);
  if (!leads.length) return;
  const generation = class4PrefetchGeneration;
  const stale = () => generation !== class4PrefetchGeneration || shared.overlayMode !== "class4";
  whenIdle(async () => {
    for (const leadDay of leads) {
      if (stale()) return;
      await prefetchClass4(request.url, { ...request, leadDay });
    }
  });
}

// Next lead first: a scrub goes forward far more often than back.
function neighbouringLeads(leadDay, slug) {
  const manifest = manifestFor(slug);
  if (!manifest || !Array.isArray(manifest.lead_days) || !manifest.lead_days.length) return [];
  const minimum = Math.min(...manifest.lead_days);
  const maximum = Math.max(...manifest.lead_days);
  return [leadDay + 1, leadDay - 1].filter((lead) => lead >= minimum && lead <= maximum);
}

function whenIdle(run) {
  if (typeof self.requestIdleCallback === "function") self.requestIdleCallback(() => run(), { timeout: 2000 });
  else setTimeout(run, 400);
}

// Bucket latency on the match-up parquet ranges from a second to well over a minute, so the
// wait carries its own byte counter and bar. Both ride inside the note box the layout already
// reserves (the bar is positioned out of flow), so a load moves nothing on screen. The label
// is whatever the caller last announced, since a first load and a lead change say different
// things. Any later `note.textContent =` clears the bar with the text it replaces.
let class4ProgressLabel = "Loading Class-4 match-ups…";

function showClass4Progress(progress) {
  const note = elements["overlay-note"];
  if (!note) return;
  const done = megabytes(progress.bytesDone);
  const total = progress.bytesTotal > 0 ? megabytes(progress.bytesTotal) : null;
  note.textContent = total ? `${class4ProgressLabel} ${done} of ${total} MB` : `${class4ProgressLabel} ${done} MB`;
  const bar = document.createElement("span");
  bar.className = "note-progress";
  const fill = document.createElement("span");
  fill.className = "note-progress-fill";
  if (total) fill.style.width = `${Math.min(100, (progress.bytesDone / progress.bytesTotal) * 100).toFixed(1)}%`;
  else bar.classList.add("indeterminate");
  bar.appendChild(fill);
  note.appendChild(bar);
}

function megabytes(bytes) {
  return (Number(bytes) / (1024 * 1024)).toFixed(1);
}

// The parquet variable name(s) needed to draw the class-4 overlay for every visible
// panel. Derived current speed needs both velocity components. Used to skip row groups
// that provably hold no requested variable (rows sorted start,lead,variable).
function class4RequestVariables() {
  const set = new Set();
  for (let i = 0; i < shared.layout; i += 1) {
    const variable = panels[i] && panels[i].state.variable;
    if (!variable) continue;
    if (isCurrentsVariable(variable)) {
      set.add("eastward_sea_water_velocity");
      set.add("northward_sea_water_velocity");
    } else {
      set.add(class4ParquetVariable(variable));
    }
  }
  return [...set];
}

function class4ByteLengthHint(class4Url, class4Manifest) {
  const byteLength = Number(class4Manifest && class4Manifest.bytes);
  if (!Number.isFinite(byteLength) || byteLength <= 0) return undefined;
  const manifestUrl = class4Manifest.url || class4Manifest.href || class4Manifest.path;
  if (!manifestUrl) return byteLength;
  try {
    const expected = new URL(resolveViewerDataUrl(class4Url), window.location.href).href;
    const hinted = new URL(resolveViewerDataUrl(manifestUrl), window.location.href).href;
    return expected === hinted ? byteLength : undefined;
  } catch {
    return undefined;
  }
}

// The forecast start date currently selected, as the YYYY-MM-DD string the
// match-up parquet's start_date column and row-group statistics use.
function currentStartDate(slug) {
  const manifest = manifestFor(slug) || (panels[0] && manifestFor(panels[0].state.dataset));
  if (!manifest || !Array.isArray(manifest.start_dates) || !manifest.start_dates.length) return null;
  return manifest.start_dates[Math.min(shared.startIndex, manifest.start_dates.length - 1)];
}

function drawOverlays(panel) {
  const canvas = panel.els.overlay;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (shared.scope === "year") return;
  const projection = projectionFor(panel);
  // The PSD rectangle is a spectrum tool, not a purpose overlay: it draws in every
  // single-forecast display (incl. swipe and difference — one shared box, both spectra).
  if (psdBoxVisible()) drawPsdBox(panel, context, projection);
  // Class-4 points and eddies are per-forecast; the difference view has no single
  // forecast to attach them to, so it stays overlay-free (see the quiet note).
  if (isDiffView()) return;
  if (shared.overlayMode === "none") return;
  const ratio = window.devicePixelRatio || 1;
  // Points/contours belong to the periodic world too: draw them on every visible
  // wrapped copy so they stay on the field when panning across the dateline.
  const copyOffsets = visibleCopyOffsets(projection, canvas);
  const projectOnCopy = (offset) => (nx, ny) => projection.project(nx + offset, ny);

  if (shared.overlayMode === "column") {
    if (shared.columnPoint) drawColumnMarker(panel, context, projection, copyOffsets);
    return;
  }
  if (shared.overlayMode === "eddies") {
    const index = panel.index;
    const match = overlayData.eddiesMatch;
    if (shared.layout === 2 && match) {
      const own = index === 0 ? match.matched.map((pair) => pair.a) : match.matched.map((pair) => pair.b);
      const only = index === 0 ? match.onlyA : match.onlyB;
      for (const offset of copyOffsets) {
        drawEddyDetections(context, projectOnCopy(offset), own, EDDY_MATCHED_COLOR, { devicePixelRatio: ratio });
        drawEddyDetections(context, projectOnCopy(offset), only, forecastColor(index), { devicePixelRatio: ratio });
      }
    } else {
      const census = overlayData.eddiesCensuses[index];
      if (census) {
        for (const offset of copyOffsets) {
          drawEddyDetections(context, projectOnCopy(offset), census.detections, forecastColor(index), {
            devicePixelRatio: ratio,
          });
        }
      }
    }
  } else if (shared.overlayMode === "class4" && overlayData.class4 && !isSurfaceCurrentVariable(panel.state.variable)) {
    const manifest = manifestFor(panel.state.dataset);
    const entry = manifest && manifest.variables[panel.state.variable];
    const depthBin = class4DepthBin(entry);
    const startDate = manifest ? manifest.start_dates[Math.min(shared.startIndex, manifest.start_dates.length - 1)] : null;
    const rows = overlayData.class4.rows || [];
    const targeted = overlayData.class4.targeted;
    const selector = {
      variable: panel.state.variable,
      depthBin,
      leadDay: shared.leadDay,
      startDate,
    };
    // Row-scan (filtering + error scale + match count) depends only on the selector and
    // the loaded rows, not the viewport — cache it so pan/zoom rAF redraws only reproject
    // the already-filtered points instead of rescanning every row again (perf).
    const cacheKey = `${panel.state.variable}|${depthBin || ""}|${shared.leadDay}|${startDate || ""}|${rows.length}|${overlayData.class4.targeted}`;
    let prepared = panel.class4Prepared;
    if (!prepared || prepared.key !== cacheKey) {
      const preparedPoints = class4Points(rows, selector);
      prepared = {
        key: cacheKey,
        points: preparedPoints,
        matchedTotal: countClass4Matches(rows, selector),
        scale: class4ErrorScale(preparedPoints),
        // Parallel typed arrays + coarse lon/lat bucket grid, built once per point set so
        // pan/zoom frames only project the buckets overlapping the viewport (perf).
        ...buildClass4Index(preparedPoints),
      };
      panel.class4Prepared = prepared;
    }
    const matchedTotal = prepared.matchedTotal;
    const scale = prepared.scale;
    // Larger points at high zoom so individual obs are distinguishable from a line.
    const radius = 2.2 + 2.6 * Math.min(1, (view.zoom - 1) / 20);
    const display = drawClass4Frame(context, projection, prepared, copyOffsets, canvas, {
      devicePixelRatio: ratio,
      errorScale: scale,
      radius,
    });
    panel.class4Scale = scale;
    panel.class4Count = display.drawnVisible;
    panel.class4VisibleTotal = display.visibleTotal;
    panel.class4Stride = display.stride;
    panel.class4Thinned = display.thinned;
    panel.class4Matched = matchedTotal;
    panel.class4Targeted = targeted;
    panel.class4HitPoints = prepared.points;
    panel.class4HitSelected = display.selectedMask;
    panel.class4PointRadius = radius;
  } else {
    panel.class4HitPoints = null;
    panel.class4HitSelected = null;
    panel.class4Thinned = false;
    panel.class4Count = 0;
    panel.class4VisibleTotal = 0;
    panel.class4Matched = 0;
    panel.class4Scale = 0;
  }
}

// Crosshair marker at the clicked water-column point, drawn on every visible world copy so
// it stays on the field when panning across the dateline.
function drawColumnMarker(panel, context, projection, copyOffsets) {
  const ratio = window.devicePixelRatio || 1;
  const nx = (shared.columnPoint.lon + 180) / 360;
  const ny = (90 - shared.columnPoint.lat) / 180;
  context.save();
  context.strokeStyle = themeToken("--ob-viewer-canvas-note", shared.theme === "light" ? "#1f2937" : "#e5edf5");
  context.lineWidth = 1.6 * ratio;
  const arm = 9 * ratio;
  for (const offset of copyOffsets) {
    const point = projection.project(nx + offset, ny);
    context.beginPath();
    context.arc(point.x, point.y, 6 * ratio, 0, Math.PI * 2);
    context.moveTo(point.x - arm, point.y);
    context.lineTo(point.x - 3 * ratio, point.y);
    context.moveTo(point.x + 3 * ratio, point.y);
    context.lineTo(point.x + arm, point.y);
    context.moveTo(point.x, point.y - arm);
    context.lineTo(point.x, point.y - 3 * ratio);
    context.moveTo(point.x, point.y + 3 * ratio);
    context.lineTo(point.x, point.y + arm);
    context.stroke();
  }
  context.restore();
}

function drawTrajectoryFans(panel) {
  if (isDiffView()) return;
  if (!trajectoryState || !trajectoryState.trajectories) return;
  const context = panel.els.overlay.getContext("2d");
  const projection = projectionFor(panel);
  const ratio = window.devicePixelRatio || 1;
  const shownLead = Math.min(shared.leadDay, trajectoryState.maximumLead);
  const copyOffsets = visibleCopyOffsets(projection, panel.els.overlay);
  context.save();
  context.lineCap = "round";
  context.lineJoin = "round";
  trajectoryState.trajectories.forEach((fan, forecastIndex) => {
    for (const trajectory of fan) {
      const points = trajectory.slice(0, shownLead + 1);
      for (let segment = 1; segment < points.length; segment += 1) {
        const alpha = 0.2 + 0.65 * segment / Math.max(1, points.length - 1);
        context.strokeStyle = `${TRAJECTORY_COLORS[forecastIndex]}${Math.round(alpha * 255).toString(16).padStart(2, "0")}`;
        context.lineWidth = (forecastIndex === 0 ? 1.7 : 1.4) * ratio;
        for (const offset of copyOffsets) {
          const from = projection.project((points[segment - 1].longitude + 180) / 360 + offset, (90 - points[segment - 1].latitude) / 180);
          const to = projection.project((points[segment].longitude + 180) / 360 + offset, (90 - points[segment].latitude) / 180);
          context.beginPath();
          context.moveTo(from.x, from.y);
          context.lineTo(to.x, to.y);
          context.stroke();
        }
      }
    }
  });
  context.restore();
}

function trajectoryModeActive() {
  return shared.overlayMode === "trajectories";
}

// In trajectory purpose-mode, any visible panel is eligible whatever variable it
// renders (SSH, salinity, currents…): the worker fetches that forecast's u/v current
// fields directly, so eligibility only needs the panel drawn and its manifest to carry
// velocity components.
function trajectoryEligiblePanels() {
  if (!trajectoryModeActive()) return [];
  return panels.slice(0, shared.layout).filter((panel) => {
    if (!panel || !panel.longitudes) return false;
    const manifest = manifestFor(panel.state.dataset);
    return manifest && currentDepthVariables(panel).u in manifest.variables;
  });
}

function makeSeedCluster(longitude, latitude, radiusDegrees) {
  const seeds = [];
  const count = 20;
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));
  for (let index = 0; index < count; index += 1) {
    const radius = radiusDegrees * Math.sqrt((index + 0.5) / count);
    const angle = index * goldenAngle;
    seeds.push({
      longitude: longitude + radius * Math.cos(angle) / Math.max(0.2, Math.cos(latitude * Math.PI / 180)),
      latitude: latitude + radius * Math.sin(angle),
    });
  }
  return seeds;
}

// Trajectory advection always runs on the model's FINEST published pyramid level,
// whatever the display zoom — coarse levels smooth the currents and change the physics
// of the fan. A whole-domain finest read (10 leads × u,v at 1/12°) would be far too
// heavy, so only the tiles covering the seed cluster plus a generous drift margin are
// fetched (readLayerWindow). Particles that outrun the margin stop, exactly as they
// would at a domain edge; the margin scales with the lead range so that stays rare.
const TRAJECTORY_MARGIN_BASE_DEG = 8;
const TRAJECTORY_MARGIN_PER_LEAD_DEG = 0.5;

function finestLevel(manifest) {
  const levels = [...manifest.levels].sort((a, b) => a.cell_size_deg - b.cell_size_deg);
  return levels[0].level;
}

async function loadTrajectoryFields(panel, maximumLead, seedCentre) {
  const manifest = manifestFor(panel.state.dataset);
  const level = finestLevel(manifest);
  const coordinates = await loadCoordinates(panel.state.dataset, level);
  const variables = currentDepthVariables(panel);
  const startIndex = Math.min(shared.startIndex, manifest.start_dates.length - 1);
  const margin = TRAJECTORY_MARGIN_BASE_DEG + TRAJECTORY_MARGIN_PER_LEAD_DEG * maximumLead;
  const box = {
    latMin: seedCentre.latitude - margin,
    latMax: seedCentre.latitude + margin,
    lonMin: seedCentre.longitude - margin,
    lonMax: seedCentre.longitude + margin,
  };
  const store = stores.get(panel.state.dataset);
  const fields = [];
  for (let leadIndex = 0; leadIndex < maximumLead; leadIndex += 1) {
    const [u, v] = await Promise.all([
      readLayerWindow(store, { variable: variables.u, level, startIndex, leadIndex }, coordinates.latitudes, coordinates.longitudes, box),
      readLayerWindow(store, { variable: variables.v, level, startIndex, leadIndex }, coordinates.latitudes, coordinates.longitudes, box),
    ]);
    if (!u || !v) throw new Error("Seed point is outside this forecast's domain");
    fields.push({
      u: u.data,
      v: v.data,
      width: u.width,
      height: u.height,
      lon0: u.lon0,
      lat0: u.lat0,
      lonStep: u.lonStep,
      latStep: u.latStep,
      // The window is a regional cut-out with a continuous longitude axis: never
      // periodic, even on a global grid (the margin absorbs dateline crossings).
      periodic: false,
    });
  }
  return { fields };
}

async function seedTrajectories(panel, event) {
  if (!trajectoryModeActive()) return;
  const eligible = trajectoryEligiblePanels();
  if (!eligible.length || (shared.layout === 2 && eligible.length !== 2)) return;
  const rectangle = panel.els.field.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const world = projectionFor(panel).unproject((event.clientX - rectangle.left) * ratio, (event.clientY - rectangle.top) * ratio);
  let longitude = world.nx * 360 - 180;
  if (shared.region === "global") longitude = ((((longitude + 180) % 360) + 360) % 360) - 180;
  const latitude = 90 - world.ny * 180;
  // Seed-cluster radius from the model's FINEST grid (the grid the particles advect
  // on), not the zoom-dependent display grid — so the same click yields the same fan
  // at any zoom. Falls back to the display spacing if the manifest lacks levels.
  const finestCellDeg =
    finestCellDegFor(panel.state.dataset) ??
    Math.max(Math.abs(panel.longitudes[1] - panel.longitudes[0]), Math.abs(panel.latitudes[1] - panel.latitudes[0]));
  const seeds = makeSeedCluster(longitude, latitude, Math.max(0.75, finestCellDeg * 1.5));
  const maximumLead = Math.min(...eligible.map((candidate) => Math.max(...manifestFor(candidate.state.dataset).lead_days)));
  const requestId = ++trajectoryRequestId;
  trajectoryState = { requestId, maximumLead, loading: true };
  renderTrajectoryRail();
  setStatus("Computing illustrative trajectories…");
  try {
    const forecasts = await Promise.all(
      eligible.map((candidate) => loadTrajectoryFields(candidate, maximumLead, { longitude, latitude })),
    );
    if (requestId !== trajectoryRequestId) return;
    trajectoryWorker.postMessage({ requestId, seeds, forecasts, maximumLead });
  } catch (error) {
    if (requestId === trajectoryRequestId) {
      clearTrajectories();
      setStatus(`Trajectory computation failed: ${error.message}`, true);
    }
  }
}

trajectoryWorker.addEventListener("message", ({ data }) => {
  if (!trajectoryState || data.requestId !== trajectoryState.requestId) return;
  Object.assign(trajectoryState, data, { loading: false });
  // Debug/verification surface (like __oceanbenchViewerDataBaseUrl): lets tooling assert
  // that the same seed yields identical trajectories regardless of display zoom.
  window.__oceanbenchTrajectories = trajectoryState;
  setStatus("");
  redrawOverlaysAll();
  renderTrajectoryRail();
});

function clearTrajectories() {
  trajectoryRequestId += 1;
  trajectoryState = null;
  // renderTrajectoryRail only rewrites the caption while a fan exists, so blank it here
  // rather than leave the previous fan's sentence under a hidden section.
  if (elements["rail-trajectory-note"]) elements["rail-trajectory-note"].textContent = "";
  if (elements["rail-trajectory-section"]) renderTrajectoryRail();
  if (panels.length) redrawOverlaysAll();
}

function renderTrajectoryRail() {
  const section = elements["rail-trajectory-section"];
  if (!section) return;
  section.hidden = !trajectoryState;
  if (!trajectoryState) return;
  elements["rail-trajectory-chart"].innerHTML = trajectoryState.separation
    ? trajectorySeparationSVG(trajectoryState.separation, shared.leadDay, chartTypeScale(elements["rail-trajectory-chart"]))
    : "";
  elements["rail-trajectory-note"].textContent = trajectoryState.loading
    ? "Loading current fields and advecting 20 shared seeds…"
    : trajectoryState.separation.length
      ? "Mean separation between corresponding Forecast 1 and Forecast 2 particles."
      : "Single-forecast trajectory fan.";
  wireCursorTooltip(elements["rail-trajectory-chart"]);
}

// ---- water-column profile on click ------------------------------------------

const COLUMN_STORE_VARIABLES = ["sea_water_potential_temperature", "sea_water_salinity"];

// The one sentence for "this dataset has no column store", mirroring class4EmptyMessage:
// the carried datasets publish no <slug>.columns.zarr, and a click that reads nothing must
// say so rather than look like a dead map.
const COLUMN_UNPUBLISHED_MESSAGE = "No water-column data is published for this dataset.";
// True while column mode is on and no visible forecast has a store to read.
let columnUnpublished = false;

// Whether any visible forecast publishes a column store; also refreshes `columnUnpublished`
// so the rail and the overlay note state the same thing.
async function anyVisibleColumnStore() {
  const visible = panels.slice(0, shared.layout).filter(Boolean);
  const stores = await Promise.all(visible.map((panel) => ensureColumnStore(panel.state.dataset)));
  const available = stores.some(Boolean);
  columnUnpublished = !available;
  return available;
}

function columnModeActive() {
  return shared.overlayMode === "column" && !isDiffView();
}

// Resolve (once, cached) whether a challenger publishes a water-column store. The manifest
// says so outright with `has_columns`; manifests written before that key existed do not, and
// for those a missing store 404s on its .zmetadata, loadStore rejects and we remember `false`
// so the feature is simply not offered for that challenger. No retries, no console noise.
async function ensureColumnStore(slug) {
  if (columnStores.has(slug)) return columnStores.get(slug);
  const manifest = manifestFor(slug);
  if (manifest && manifest.has_columns === false) {
    columnStores.set(slug, false);
    return false;
  }
  let store = false;
  try {
    store = await loadStore(resolveColumnStoreUrl(slug));
  } catch {
    store = false;
  }
  columnStores.set(slug, store);
  return store;
}

// Show/enable the "Water column (click)" overlay option only when at least one visible
// challenger has a column store, so a mode with nothing behind it is not offered. The one
// exception is a mode already active: switching to a challenger without a store must not
// silently drop the user out of the mode, it must keep the option and say why the map
// reads nothing (COLUMN_UNPUBLISHED_MESSAGE, rendered by applyOverlayMode and the rail).
async function updateColumnModeAvailability() {
  const option = elements["overlay-mode"] && elements["overlay-mode"].querySelector('option[value="column"]');
  if (!option) return;
  const available = await anyVisibleColumnStore();
  const active = shared.overlayMode === "column";
  option.hidden = !available && !active;
  option.disabled = option.hidden;
  if (!available && active) {
    clearColumnProfile();
    const note = elements["overlay-note"];
    if (note) note.textContent = COLUMN_UNPUBLISHED_MESSAGE;
  }
}

function nearestCoordinateIndex(coordinates, target) {
  let bestIndex = 0;
  let bestDistance = Infinity;
  for (let i = 0; i < coordinates.length; i += 1) {
    const distance = Math.abs(coordinates[i] - target);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestIndex = i;
    }
  }
  return bestIndex;
}

// Which column-store variable a panel maps to: its own variable when that is temperature
// or salinity, otherwise temperature (the sensible default for a non-T/S panel).
function columnVariableFor(panel) {
  const entry = variableEntry(manifestFor(panel.state.dataset), panel.state.variable);
  const standardName = entry && entry.standard_name;
  if (standardName === "sea_water_salinity") return { name: standardName, unit: entry.units || "", label: "Salinity" };
  if (standardName === "sea_water_potential_temperature") return { name: standardName, unit: entry.units || "°C", label: "Temperature" };
  return { name: "sea_water_potential_temperature", unit: "°C", label: "Temperature" };
}

function clearColumnProfile() {
  columnRequestId += 1;
  columnProfile = null;
  shared.columnPoint = null;
  window.__oceanbenchColumnProfile = null;
  renderColumnProfileRail();
  if (panels.length) redrawOverlaysAll();
}

// Read the full water column (all leads + all depths) at a clicked lon/lat for every
// visible challenger that has a store. One chunk per challenger — held in memory so the
// lead slider re-renders with no refetch (renderColumnProfileRail slices in-memory).
async function readColumnProfileAt(longitude, latitude) {
  const eligiblePanels = [];
  for (const panel of panels.slice(0, shared.layout).filter(Boolean)) {
    const store = await ensureColumnStore(panel.state.dataset);
    if (store) eligiblePanels.push({ panel, store });
  }
  // No store to read: say so once, in the note and the rail, instead of swallowing the click.
  if (!eligiblePanels.length) {
    columnUnpublished = true;
    const emptyNote = elements["overlay-note"];
    if (emptyNote) emptyNote.textContent = COLUMN_UNPUBLISHED_MESSAGE;
    renderColumnProfileRail();
    return;
  }
  columnUnpublished = false;
  shared.columnPoint = { lon: longitude, lat: latitude };
  const requestId = ++columnRequestId;
  const forecasts = [];
  for (const { panel, store } of eligiblePanels) {
    const choice = columnVariableFor(panel);
    const arrayMeta = store.metadata[`${choice.name}/.zarray`];
    if (!arrayMeta) continue;
    // The column store may hold fewer starts than the pyramid it sits beside; clamp so a
    // high start index never requests a chunk past the store's start axis.
    const startIndex = Math.min(shared.startIndex, arrayMeta.shape[0] - 1);
    const [depths, latitudes, longitudes] = await Promise.all([
      readRootCoordinate(store, "depth"),
      readRootCoordinate(store, "latitude"),
      readRootCoordinate(store, "longitude"),
    ]);
    const latIndex = nearestCoordinateIndex(latitudes, latitude);
    const lonIndex = nearestCoordinateIndex(longitudes, longitude);
    const column = await readColumn(store, { variable: choice.name, startIndex, latIndex, lonIndex });
    if (requestId !== columnRequestId) return;
    forecasts.push({
      panelIndex: panel.index,
      slug: panel.state.dataset,
      variable: choice.name,
      startIndex,
      latIndex,
      lonIndex,
      color: forecastColor(panel.index),
      datasetLabel: labelFor(panel.state.dataset),
      variableLabel: choice.label,
      unit: choice.unit,
      depths: Array.from(depths),
      values: column.values,
      leads: column.leads,
      depthSize: column.depths,
      allNaN: !column.values.some((value) => Number.isFinite(value)),
    });
  }
  if (requestId !== columnRequestId) return;
  columnProfile = { lon: longitude, lat: latitude, forecasts };
  // Deterministic hook for tests: the decoded values at the clicked point.
  window.__oceanbenchColumnProfile = columnProfile;
  const note = elements["overlay-note"];
  if (note) note.textContent = "Click the map to read the model water column at another point.";
  renderColumnProfileRail();
  if (panels.length) redrawOverlaysAll();
  writeHash();
}

function formatLatLon(lon, lat) {
  const latText = `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"}`;
  const lonText = `${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "W"}`;
  return `${latText}, ${lonText}`;
}

// Render the profile chart for the CURRENT lead day from the in-memory column (no fetch).
// Both forecasts are drawn in their panel accent colours; a point over land/missing (all
// fill values) yields the honest "no water column" note.
function renderColumnProfileRail() {
  const section = elements["rail-column-section"];
  if (!section) return;
  if (!columnProfile || !columnModeActive()) {
    // Nothing published for the visible forecasts is a fact worth showing: keep the section
    // up with the empty-state sentence instead of hiding the feature and looking broken.
    const unpublished = columnModeActive() && columnUnpublished;
    section.hidden = !unpublished;
    elements["rail-column-chart"].innerHTML = "";
    elements["rail-column-point"].textContent = "";
    elements["rail-column-note"].textContent = unpublished ? COLUMN_UNPUBLISHED_MESSAGE : "";
    return;
  }
  section.hidden = false;
  elements["rail-column-point"].textContent = formatLatLon(columnProfile.lon, columnProfile.lat);
  const comparison = columnProfile.forecasts.length === 2;
  const leadIndex = Math.max(0, shared.leadDay - 1);
  const lines = [];
  let unit = "";
  for (const forecast of columnProfile.forecasts) {
    if (forecast.allNaN) continue;
    const lead = Math.min(leadIndex, forecast.leads - 1);
    const points = [];
    for (let depth = 0; depth < forecast.depthSize; depth += 1) {
      const value = forecast.values[lead * forecast.depthSize + depth];
      if (Number.isFinite(value)) points.push({ depth: forecast.depths[depth], value });
    }
    if (!points.length) continue;
    unit = unit || forecast.unit;
    lines.push({
      label: comparison ? `F${forecast.panelIndex + 1} · ${forecast.variableLabel}` : forecast.variableLabel,
      color: forecast.color,
      points,
    });
  }
  const variableLabel = columnProfile.forecasts.find((forecast) => !forecast.allNaN)?.variableLabel || "Value";
  elements["rail-column-chart"].innerHTML = columnProfileSVG(lines, {
    title: comparison ? "Water column (both forecasts)" : "Water column",
    unit,
    xLabel: variableLabel,
  });
  if (!lines.length) {
    elements["rail-column-note"].textContent = "No water column at this point.";
  } else {
    elements["rail-column-note"].textContent = `Model ${variableLabel.toLowerCase()} profile at the selected start and lead day ${shared.leadDay}. Move the lead slider to re-read from the same download.`;
  }
  const heading = section.querySelector("h3");
  if (heading) attachMethodNote(heading, "column-profile");
  wireCursorTooltip(elements["rail-column-chart"]);
}

const CLASS4_INDEX_COLS = 360;
const CLASS4_INDEX_ROWS = 180;

// Build parallel typed arrays (normalized lon/lat + abs error) and a coarse lon/lat
// bucket grid (CSR layout: `start` offsets + `order` point ids) once per point set.
// The grid lets pan/zoom frames project only the buckets overlapping the viewport and
// lets the hover hit-test scan the cursor's bucket neighbourhood instead of every point.
function buildClass4Index(points) {
  const count = points.length;
  const normalizedLongitude = new Float32Array(count);
  const normalizedLatitude = new Float32Array(count);
  const absoluteError = new Float32Array(count);
  const cols = CLASS4_INDEX_COLS;
  const rows = CLASS4_INDEX_ROWS;
  const bucketCounts = new Int32Array(cols * rows);
  const bucketOfPoint = new Int32Array(count);
  for (let i = 0; i < count; i += 1) {
    const point = points[i];
    const nx = (point.longitude + 180) / 360;
    const ny = (90 - point.latitude) / 180;
    normalizedLongitude[i] = nx;
    normalizedLatitude[i] = ny;
    absoluteError[i] = class4AbsoluteError(point);
    let column = Math.floor(nx * cols);
    if (column < 0) column = 0;
    else if (column >= cols) column = cols - 1;
    let row = Math.floor(ny * rows);
    if (row < 0) row = 0;
    else if (row >= rows) row = rows - 1;
    const bucket = row * cols + column;
    bucketOfPoint[i] = bucket;
    bucketCounts[bucket] += 1;
  }
  const start = new Int32Array(cols * rows + 1);
  for (let bucket = 0; bucket < cols * rows; bucket += 1) start[bucket + 1] = start[bucket] + bucketCounts[bucket];
  const order = new Int32Array(count);
  const cursor = Int32Array.from(start.subarray(0, cols * rows));
  for (let i = 0; i < count; i += 1) {
    const bucket = bucketOfPoint[i];
    order[cursor[bucket]] = i;
    cursor[bucket] += 1;
  }
  return {
    normalizedLongitude,
    normalizedLatitude,
    absoluteError,
    spatialIndex: { cols, rows, start, order },
    // Frame-stamp visibility marker: avoids clearing an N-length mask every frame.
    visibilityStamp: new Int32Array(count),
    frameStamp: 0,
    candidateScratch: new Int32Array(count),
    columnScratch: new Uint8Array(cols),
  };
}

// Project the viewport-overlapping buckets once, decide stride-thinning, and draw — no
// second projection pass. Returns the exact same `visibleTotal` / `drawnVisible` / `stride`
// / `thinned` semantics as the previous per-point scan, plus the display selection mask the
// hover hit-test reuses. `selectedMask` is null when nothing is thinned (whole set drawn).
function drawClass4Frame(context, projection, prepared, copyOffsets, canvas, options) {
  const normalizedLongitude = prepared.normalizedLongitude;
  const normalizedLatitude = prepared.normalizedLatitude;
  const absoluteError = prepared.absoluteError;
  const count = normalizedLongitude.length;
  if (!count) return { visibleTotal: 0, drawnVisible: 0, stride: 1, thinned: false, selectedMask: null };
  const { cols, rows, start, order } = prepared.spatialIndex;
  const { originX, originY, displayWidth, displayHeight } = projection;
  const width = canvas.width;
  const height = canvas.height;
  // Viewport bounds in base normalized coordinates (before world-copy offset).
  const nyTop = (0 - originY) / displayHeight;
  const nyBottom = (height - originY) / displayHeight;
  const nxLeft = (0 - originX) / displayWidth;
  const nxRight = (width - originX) / displayWidth;
  let rowLow = Math.floor(Math.min(nyTop, nyBottom) * rows) - 1;
  let rowHigh = Math.floor(Math.max(nyTop, nyBottom) * rows) + 1;
  rowLow = Math.max(0, rowLow);
  rowHigh = Math.min(rows - 1, rowHigh);
  const columnMask = prepared.columnScratch;
  columnMask.fill(0);
  for (const offset of copyOffsets) {
    let low = Math.min(nxLeft - offset, nxRight - offset);
    let high = Math.max(nxLeft - offset, nxRight - offset);
    low = Math.max(0, low);
    high = Math.min(0.9999999, high);
    if (low > high) continue;
    let columnLow = Math.floor(low * cols) - 1;
    let columnHigh = Math.floor(high * cols) + 1;
    columnLow = Math.max(0, columnLow);
    columnHigh = Math.min(cols - 1, columnHigh);
    for (let column = columnLow; column <= columnHigh; column += 1) columnMask[column] = 1;
  }
  // Gather candidate point ids from the overlapping buckets.
  const candidates = prepared.candidateScratch;
  let candidateCount = 0;
  if (rowLow <= rowHigh) {
    for (let row = rowLow; row <= rowHigh; row += 1) {
      const rowBase = row * cols;
      for (let column = 0; column < cols; column += 1) {
        if (!columnMask[column]) continue;
        const bucket = rowBase + column;
        for (let s = start[bucket]; s < start[bucket + 1]; s += 1) candidates[candidateCount++] = order[s];
      }
    }
  }
  const stamp = prepared.visibilityStamp;
  const frameStamp = (prepared.frameStamp += 1);
  let visibleTotal = 0;
  // Per world-copy: project each candidate once, cull off-canvas, keep the device-pixel
  // coordinates for the draw pass (single projection). Mark visibility across copies.
  const perCopy = [];
  for (const offset of copyOffsets) {
    const screenX = new Float32Array(candidateCount);
    const screenY = new Float32Array(candidateCount);
    const pointIds = new Int32Array(candidateCount);
    let onCanvas = 0;
    for (let t = 0; t < candidateCount; t += 1) {
      const id = candidates[t];
      const x = originX + (normalizedLongitude[id] + offset) * displayWidth;
      if (x < 0 || x > width) continue;
      const y = originY + normalizedLatitude[id] * displayHeight;
      if (y < 0 || y > height) continue;
      screenX[onCanvas] = x;
      screenY[onCanvas] = y;
      pointIds[onCanvas] = id;
      onCanvas += 1;
      if (stamp[id] !== frameStamp) {
        stamp[id] = frameStamp;
        visibleTotal += 1;
      }
    }
    perCopy.push({ screenX, screenY, pointIds, onCanvas });
  }
  let thinned = false;
  let stride = 1;
  let drawnVisible = visibleTotal;
  let selectedMask = null;
  if (!(view.zoom >= CLASS4_FULL_DENSITY_ZOOM) && visibleTotal > CLASS4_DISPLAY_POINT_BUDGET) {
    thinned = true;
    stride = Math.ceil(visibleTotal / CLASS4_DISPLAY_POINT_BUDGET);
    selectedMask = new Uint8Array(count);
    // Walk point ids in ascending order (identical to the old point-order scan) so the
    // stride selection picks exactly the same points, then count the drawn subset.
    let visiblePosition = 0;
    let selectedCount = 0;
    for (let id = 0; id < count; id += 1) {
      if (stamp[id] !== frameStamp) continue;
      if (visiblePosition % stride === 0) {
        selectedMask[id] = 1;
        selectedCount += 1;
      }
      visiblePosition += 1;
    }
    drawnVisible = selectedCount;
  }
  for (const copy of perCopy) {
    drawClass4Screen(context, copy.screenX, copy.screenY, copy.pointIds, copy.onCanvas, absoluteError, selectedMask, options);
  }
  return { visibleTotal, drawnVisible, stride, thinned, selectedMask };
}

// Integer world-copy offsets whose earth copy is currently visible (periodic wrap).
function visibleCopyOffsets(projection, canvas) {
  if (shared.region !== "global") return [0];
  const left = projection.unproject(0, 0).nx;
  const right = projection.unproject(canvas.width, 0).nx;
  const offsets = [];
  for (let k = Math.floor(Math.min(left, right)); k <= Math.ceil(Math.max(left, right)); k += 1) offsets.push(k);
  return offsets.length ? offsets : [0];
}

// Number of Class-4 rows matching the active selector before spatial thinning — the
// "of M sampled" denominator the legend reports so low counts read as weak (item 5).
function countClass4Matches(rows, { variable, depthBin, leadDay, startDate }) {
  if (isCurrentsVariable(variable)) return class4Points(rows, { variable, depthBin, leadDay, startDate }).length;
  if (!rows) return 0;
  const parquetVariable = class4ParquetVariable(variable);
  const requestedLead = leadDay == null ? null : Number(leadDay);
  let total = 0;
  for (const row of rows) {
    if (row.variable !== parquetVariable) continue;
    if (depthBin && row.depth_bin !== depthBin) continue;
    if (requestedLead !== null && Number(row.lead_day) !== requestedLead) continue;
    if (startDate && String(row.start_date).slice(0, 10) !== startDate) continue;
    // Match the drawn set: masked-model rows (non-finite error) are not drawn, so they
    // must not inflate the "N obs" / "of Y" denominator either.
    if (!Number.isFinite(class4AbsoluteError(row))) continue;
    total += 1;
  }
  return total;
}

function class4DepthBin(entry) {
  if (!entry) return null;
  if (entry.depth === "15m") return "15m";
  if (entry.standard_name.includes("velocity")) return "15m";
  if (entry.standard_name === "sea_surface_height_above_geoid") return "surface";
  return "0-5m"; // temperature / salinity near-surface bin matching the surface viewer field
}

function class4DepthLabel(entry, depthBin) {
  if (!entry) return depthBin || "selected depth";
  if (entry.depth && entry.depth !== "surface") return entry.depth;
  return depthBin || entry.depth || "selected depth";
}

// ---- particles --------------------------------------------------------------

function startPanelParticles(panel) {
  stopParticles(panel);
  resizePanelCanvases(panel);
  const projection = projectionFor(panel);
  panel.particleContext = {
    sampleVelocity: (nx, ny) => panel.velocity.sampler(nx, ny),
    project: projection.project,
    viewport: visibleViewport(projection, panel.els.particles),
    magnitudeScale: PARTICLE_MAGNITUDE_SCALE,
    theme: shared.theme,
    muted: fieldMutedUnderOverlay(),
    speed: shared.particleSpeed,
    devicePixelRatio: window.devicePixelRatio || 1,
    playing: true,
    punchLand: makeLandPunch(panel, projection),
  };
  panel.particleHandle = startParticleField(panel.els.particles, panel.particleContext);
}

function stopParticles(panel) {
  if (panel.particleHandle) {
    panel.particleHandle.stop();
    panel.particleHandle = null;
    panel.particleContext = null;
  }
}

// Start or stop the particle overlay on every visible panel to match the checkbox.
function applyParticleVisibility() {
  for (let i = 0; i < shared.layout; i += 1) {
    const panel = panels[i];
    if (isCurrentsVariable(panel.state.variable) && shared.showParticles && panel.velocity) startPanelParticles(panel);
    else stopParticles(panel);
  }
}

function updateParticleProjection(panel, projection) {
  if (!panel.particleContext) return;
  panel.particleContext.project = projection.project;
  panel.particleContext.viewport = visibleViewport(projection, panel.els.particles);
  panel.particleContext.theme = shared.theme;
  panel.particleContext.muted = fieldMutedUnderOverlay();
  panel.particleContext.speed = shared.particleSpeed;
  panel.particleContext.devicePixelRatio = window.devicePixelRatio || 1;
  panel.particleContext.punchLand = makeLandPunch(panel, projection);
}

// Erasing the particle layer wherever the field blit drew land. The velocity sampler
// already refuses to advect into a land cell, but it works on a 1 degree grid whose
// cells are many screen pixels wide, so a legal segment can still overhang the coastline
// it is drawn beside. The stencil is the same raster, blitted through the same snapped
// world path with the same nearest-neighbour scaling, so the particle layer cannot
// disagree with the map about where the coast is.
function makeLandPunch(panel, projection) {
  const stencil = panel.landStencil;
  if (!stencil) return null;
  return (target) => {
    target.save();
    target.imageSmoothingEnabled = false;
    target.globalCompositeOperation = "destination-out";
    drawImageWorld(target, stencil, panel.edgesA, projection);
    target.restore();
  };
}

function visibleViewport(projection, canvas) {
  const topLeft = projection.unproject(0, 0);
  const bottomRight = projection.unproject(canvas.width, canvas.height);
  return {
    minX: Math.min(topLeft.nx, bottomRight.nx),
    maxX: Math.max(topLeft.nx, bottomRight.nx),
    minY: Math.max(0, Math.min(topLeft.ny, bottomRight.ny)),
    maxY: Math.min(1, Math.max(topLeft.ny, bottomRight.ny)),
  };
}

// ---- pointer interaction (pan / zoom shared, hover per panel) ---------------

function beginPanelDrag(panel, event) {
  if (event.button !== 0) return;
  const projection = projectionFor(panel);
  // PSD rectangle grabs take priority over map panning: a pointerdown on the box
  // interior moves it, on a handle/edge resizes it.
  if (psdBoxVisible()) {
    const ratio = window.devicePixelRatio || 1;
    const rectangle = panel.els.field.getBoundingClientRect();
    const hit = psdBoxHitTest(
      panel,
      (event.clientX - rectangle.left) * ratio,
      (event.clientY - rectangle.top) * ratio,
      projection,
    );
    if (hit) {
      panel.draggingPsd = { hit, startBox: { ...shared.psdBox }, x: event.clientX, y: event.clientY, projection };
      panel.els.field.setPointerCapture(event.pointerId);
      panel.els.field.addEventListener("pointermove", onPanelPointerMove);
      panel.els.field.addEventListener("pointerup", endPanelDrag, { once: true });
      panel.els.field.addEventListener("pointercancel", endPanelDrag, { once: true });
      event.preventDefault();
      return;
    }
  }
  if (isSwipeHost(panel) && panel.offscreenB) {
    const ratio = window.devicePixelRatio || 1;
    const rectangle = panel.els.field.getBoundingClientRect();
    const localX = (event.clientX - rectangle.left) * ratio;
    if (Math.abs(localX - panel.swipeX * panel.els.field.width) < 12 * ratio) {
      panel.draggingSwipe = true;
      panel.els.field.setPointerCapture(event.pointerId);
      panel.els.field.addEventListener("pointermove", onPanelPointerMove);
      panel.els.field.addEventListener("pointerup", endPanelDrag, { once: true });
      panel.els.field.addEventListener("pointercancel", endPanelDrag, { once: true });
      return;
    }
  }
  panel.dragging = { x: event.clientX, y: event.clientY, centerNX: view.centerNX, centerNY: view.centerNY, projection };
  panel.els.field.style.cursor = "grabbing";
  panel.els.field.setPointerCapture(event.pointerId);
  panel.els.field.addEventListener("pointermove", onPanelPointerMove);
  panel.els.field.addEventListener("pointerup", endPanelDrag, { once: true });
  panel.els.field.addEventListener("pointercancel", endPanelDrag, { once: true });
  event.preventDefault();
}

function onPanelPointerMove(event) {
  const panel = panels.find((candidate) => candidate.els.field === event.currentTarget);
  if (panel.draggingPsd) {
    const ratio = window.devicePixelRatio || 1;
    applyPsdBoxDrag(
      panel.draggingPsd,
      (event.clientX - panel.draggingPsd.x) * ratio,
      (event.clientY - panel.draggingPsd.y) * ratio,
    );
    scheduleRedrawAllPanels();
    scheduleRailUpdate();
    scheduleHashWrite();
  } else if (panel.draggingSwipe) {
    const ratio = window.devicePixelRatio || 1;
    const rectangle = panel.els.field.getBoundingClientRect();
    panel.swipeX = Math.min(0.98, Math.max(0.02, ((event.clientX - rectangle.left) * ratio) / panel.els.field.width));
    drawPanel(panel);
  } else if (panel.dragging) {
    if (Math.hypot(event.clientX - panel.dragging.x, event.clientY - panel.dragging.y) > 4) panel.dragging.moved = true;
    const ratio = window.devicePixelRatio || 1;
    view.centerNX = panel.dragging.centerNX - ((event.clientX - panel.dragging.x) * ratio) / panel.dragging.projection.displayWidth;
    view.centerNY = panel.dragging.centerNY - ((event.clientY - panel.dragging.y) * ratio) / panel.dragging.projection.displayHeight;
    clampView();
    scheduleRedrawAllPanels();
    scheduleHashWrite();
    scheduleRailUpdate();
    // The world point under the pointer changes as the view pans, so the mirrored
    // companion cursor and both bottom-left readouts must keep tracking during the
    // drag — not only on plain mouse moves. updateHover would clear the "grabbing"
    // pointer style, so restore it afterwards.
    updateHover(event);
    panel.els.field.style.cursor = "grabbing";
  } else {
    updateHover(event);
  }
}

function endPanelDrag(event) {
  const panel = panels.find((candidate) => candidate.els.field === event.currentTarget);
  const shouldSeed = panel.dragging && !panel.dragging.moved;
  panel.els.field.removeEventListener("pointermove", onPanelPointerMove);
  if (shared.region === "global") view.centerNX = ((view.centerNX % 1) + 1) % 1;
  panel.dragging = null;
  panel.draggingSwipe = false;
  panel.draggingPsd = null;
  panel.els.field.style.cursor = "";
  if (shouldSeed) seedTrajectories(panel, event);
  if (shouldSeed && columnModeActive()) readColumnProfileAt(...pointerLonLat(panel, event));
  writeHash();
}

// Geographic lon/lat under a pointer event on a panel (longitude wrapped to ±180 on the
// periodic global grid), shared by the trajectory and water-column click handlers.
function pointerLonLat(panel, event) {
  const rectangle = panel.els.field.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const world = projectionFor(panel).unproject((event.clientX - rectangle.left) * ratio, (event.clientY - rectangle.top) * ratio);
  let longitude = world.nx * 360 - 180;
  if (shared.region === "global") longitude = ((((longitude + 180) % 360) + 360) % 360) - 180;
  return [longitude, 90 - world.ny * 180];
}

// A trackpad emits wheel events far faster than the display refreshes, and each one
// used to drive a full projection + redraw pass. Accumulate the deltas instead and
// apply them once per frame, so a fast pinch costs one render per frame, not ten.
let wheelFrame = 0;
let pendingWheel = null;

function onPanelWheel(panel, event) {
  event.preventDefault();
  if (pendingWheel && pendingWheel.panel === panel) {
    pendingWheel.deltaY += event.deltaY;
    pendingWheel.event = event;
  } else {
    pendingWheel = { panel, deltaY: event.deltaY, event };
  }
  if (wheelFrame) return;
  wheelFrame = requestAnimationFrame(() => {
    wheelFrame = 0;
    const pending = pendingWheel;
    pendingWheel = null;
    if (pending) applyWheelZoom(pending.panel, pending.event, pending.deltaY);
  });
}

function applyWheelZoom(panel, event, deltaY) {
  const projection = projectionFor(panel);
  const previousLevels = panels.slice(0, shared.layout).map((candidate) => {
    const manifest = manifests.get(candidate.state.dataset);
    return manifest ? selectRenderLevel(manifest) : null;
  });
  const ratio = window.devicePixelRatio || 1;
  const rectangle = panel.els.field.getBoundingClientRect();
  const cursorX = (event.clientX - rectangle.left) * ratio;
  const cursorY = (event.clientY - rectangle.top) * ratio;
  const before = projection.unproject(cursorX, cursorY);
  const factor = Math.exp(-deltaY * 0.0015);
  const previousZoom = view.zoom;
  view.zoom = Math.min(60, Math.max(minimumZoomFor(panel), view.zoom * factor));
  if (view.zoom === previousZoom) return;
  // Keep the world point under the cursor fixed.
  const after = projectionFor(panel).unproject(cursorX, cursorY);
  view.centerNX += before.nx - after.nx;
  view.centerNY += before.ny - after.ny;
  clampView();
  const needsRerender = panels.slice(0, shared.layout).some((candidate, index) => {
    const manifest = manifests.get(candidate.state.dataset);
    return manifest && previousLevels[index] !== selectRenderLevel(manifest);
  });
  if (needsRerender) {
    renderAllPanels().then(() => {
      redrawOverlaysAll();
      updateContextRail();
    });
  } else {
    scheduleRedrawAllPanels();
  }
  scheduleHashWrite();
  scheduleRailUpdate();
  // Zooming moves the world point under a stationary pointer, so refresh the
  // companion cursor and readouts for the new projection while hovering.
  updateHover(event);
}

function minimumZoomFor(panel) {
  const bounds = REGION_BOUNDS[shared.region];
  if (!bounds || !panel) return 1;
  const width = panel.els.field.width;
  const height = panel.els.field.height;
  const fit = Math.min(height, width / 2);
  const longitudeSpan = (bounds.east - bounds.west) / 360;
  const latitudeSpan = (bounds.north - bounds.south) / 180;
  const horizontalZoom = width / (2 * fit * longitudeSpan);
  const verticalZoom = height / (fit * latitudeSpan);
  return Math.min(60, Math.max(1, Math.min(horizontalZoom, verticalZoom)));
}

function clampView() {
  // Backstop for any non-finite view value that slipped past readHash (a stale
  // localStorage entry, a manual assignment, arithmetic underflow): a NaN zoom or
  // centre otherwise propagates into projection maths and blanks the map.
  if (!Number.isFinite(view.zoom)) view.zoom = 1;
  if (!Number.isFinite(view.centerNX)) view.centerNX = GLOBAL_DEFAULT_CENTER_NX;
  if (!Number.isFinite(view.centerNY)) view.centerNY = 0.5;
  const panel = panels.find((candidate) => candidate && candidate.els && candidate.els.field.width > 0);
  if (!panel) {
    view.centerNY = Math.min(1, Math.max(0, view.centerNY));
    return;
  }
  view.zoom = Math.min(60, Math.max(minimumZoomFor(panel), view.zoom));
  const projection = projectionFor(panel);
  const bounds = REGION_BOUNDS[shared.region];
  if (bounds) {
    const minimumNX = (bounds.west + 180) / 360;
    const maximumNX = (bounds.east + 180) / 360;
    const minimumNY = (90 - bounds.north) / 180;
    const maximumNY = (90 - bounds.south) / 180;
    const halfWidth = projection.width / (2 * projection.displayWidth);
    const halfHeight = projection.height / (2 * projection.displayHeight);
    view.centerNX =
      halfWidth * 2 >= maximumNX - minimumNX
        ? (minimumNX + maximumNX) / 2
        : Math.min(maximumNX - halfWidth, Math.max(minimumNX + halfWidth, view.centerNX));
    view.centerNY =
      halfHeight * 2 >= maximumNY - minimumNY
        ? (minimumNY + maximumNY) / 2
        : Math.min(maximumNY - halfHeight, Math.max(minimumNY + halfHeight, view.centerNY));
    return;
  }
  if (projection.displayHeight <= projection.height) {
    view.centerNY = 0.5;
    return;
  }
  const halfViewport = projection.height / (2 * projection.displayHeight);
  view.centerNY = Math.min(1 - halfViewport, Math.max(halfViewport, view.centerNY));
}

function fitRegionView() {
  const bounds = REGION_BOUNDS[shared.region];
  if (!bounds) {
    view.zoom = 1;
    view.centerNX = GLOBAL_DEFAULT_CENTER_NX;
    view.centerNY = 0.5;
    return;
  }
  const panel = panels.find((candidate) => candidate && candidate.els && candidate.els.field.width > 0);
  view.centerNX = ((bounds.west + bounds.east) / 2 + 180) / 360;
  view.centerNY = (90 - (bounds.south + bounds.north) / 2) / 180;
  if (!panel) return;
  view.zoom = minimumZoomFor(panel);
  clampView();
}

// Back to the region's default zoom/pan, redrawing (or re-rendering, when the pyramid
// level changes with the zoom). Shared by the "Reset zoom / pan" button and the
// double-click on a map.
function resetZoomPan() {
  const previousLevels = panels.slice(0, shared.layout).map((panel) => selectRenderLevel(manifestFor(panel.state.dataset)));
  fitRegionView();
  const needsRerender = panels
    .slice(0, shared.layout)
    .some((panel, index) => previousLevels[index] !== selectRenderLevel(manifestFor(panel.state.dataset)));
  if (needsRerender) renderAllPanels().then(() => redrawOverlaysAll());
  else redrawAllPanels();
  writeHash();
}

function updateHover(event) {
  let hoverPanel = null;
  let hoverRectangle = null;
  for (const panel of panels.slice(0, shared.layout)) {
    if (panel.els.ghostCursor) panel.els.ghostCursor.hidden = true;
  }
  for (const panel of panels.slice(0, shared.layout)) {
    const rectangle = panel.els.field.getBoundingClientRect();
    if (
      event.clientX < rectangle.left ||
      event.clientX > rectangle.right ||
      event.clientY < rectangle.top ||
      event.clientY > rectangle.bottom
    ) {
      panel.els.readout.textContent = "";
      continue;
    }
    if (!panel.field) continue;
    hoverPanel = panel;
    hoverRectangle = rectangle;
    break;
  }
  if (!hoverPanel || !hoverRectangle) return;
  const ratio = window.devicePixelRatio || 1;
  const projection = projectionFor(hoverPanel);
  // Cursor affordance for the PSD rectangle (move over the interior, resize on edges).
  if (psdBoxVisible()) {
    const hit = psdBoxHitTest(
      hoverPanel,
      (event.clientX - hoverRectangle.left) * ratio,
      (event.clientY - hoverRectangle.top) * ratio,
      projection,
    );
    hoverPanel.els.field.style.cursor = psdCursorFor(hit);
  } else if (columnModeActive()) {
    hoverPanel.els.field.style.cursor = "crosshair";
  } else if (hoverPanel.els.field.style.cursor) {
    hoverPanel.els.field.style.cursor = "";
  }
  const point = projection.unproject((event.clientX - hoverRectangle.left) * ratio, (event.clientY - hoverRectangle.top) * ratio);
  const wrappedNX = ((point.nx % 1) + 1) % 1;
  const lon = wrappedNX * 360 - 180;
  const lat = 90 - point.ny * 180;
  const readoutPanels = shared.layout === 2 ? panels.slice(0, 2) : [hoverPanel];
  for (const panel of panels.slice(0, shared.layout)) {
    if (!readoutPanels.includes(panel)) panel.els.readout.textContent = "";
  }
  // All hover information consolidates into the fixed bottom-left pill: when the cursor
  // is on a Class-4 obs point, its details are appended to the hovered panel's readout
  // instead of floating a tooltip with the cursor.
  const obsRecord = nearestClass4Record(hoverPanel, event, hoverRectangle);
  for (const panel of readoutPanels) {
    if (!panel.field) {
      panel.els.readout.textContent = "";
      continue;
    }
    updatePanelReadout(panel, lat, lon, panel === hoverPanel && obsRecord ? class4ReadoutSuffix(obsRecord, panel.units) : "");
  }
  updateGhostCursor(hoverPanel, wrappedNX, point.ny);
}

// Companion cursor: in side-by-side 2-forecast mode, mark the mirrored geographic
// position on the panel the user is NOT hovering with the exact same custom crosshair
// as the real cursor (at reduced opacity), so the same spot is visible on both forecasts.
function updateGhostCursor(hoverPanel, wrappedNX, ny) {
  if (shared.layout !== 2 || shared.displayMode !== "side") return;
  const other = panels[1 - hoverPanel.index];
  const ghost = other && other.els.ghostCursor;
  if (!ghost || !other.field) return;
  const projection = projectionFor(other);
  const ratio = window.devicePixelRatio || 1;
  // Pick the world copy of the longitude closest to the visible view.
  let best = null;
  for (const offset of visibleCopyOffsets(projection, other.els.field)) {
    const screen = projection.project(wrappedNX + offset, ny);
    if (screen.x >= 0 && screen.x <= projection.width && screen.y >= 0 && screen.y <= projection.height) {
      best = screen;
      break;
    }
    if (!best) best = screen;
  }
  if (!best) return;
  const x = best.x / ratio;
  const y = best.y / ratio;
  const wrap = other.els.wrap.getBoundingClientRect();
  const field = other.els.field.getBoundingClientRect();
  if (x < 0 || x > field.width || y < 0 || y > field.height) return;
  ghost.style.transform = `translate(${(x + field.left - wrap.left).toFixed(1)}px, ${(y + field.top - wrap.top).toFixed(1)}px)`;
  ghost.hidden = false;
}

function updatePanelReadout(panel, lat, lon, suffix = "") {
    const column = nearestIndex(panel.longitudes, lon);
    const row = nearestIndex(panel.latitudes, lat);
    if (column < 0 || row < 0) {
      panel.els.readout.textContent = suffix ? `${formatFixed(lat, 2)}°, ${formatFixed(lon, 2)}°${suffix}` : "";
      return;
    }
    const value = panel.field.data[row * panel.field.width + column];
    const count = panel.yearCounts && panel.yearCounts.width === panel.field.width ? panel.yearCounts.data[row * panel.yearCounts.width + column] : null;
    const standardError =
      panel.yearBiasSE && panel.yearBiasSE.width === panel.field.width
        ? panel.yearBiasSE.data[row * panel.yearBiasSE.width + column]
        : null;
    const valueText = fieldReadoutValue(panel, value, count, standardError);
    panel.els.readout.textContent = `${formatFixed(lat, 2)}°, ${formatFixed(lon, 2)}° · ${valueText}${suffix}`;
}

function fieldReadoutValue(panel, value, count, standardError) {
  if (shared.scope === "year") {
    if (!Number.isFinite(value) || count === 0) return "no observations";
    const biasMode = panel.yearMetric === "bias";
    // Non-bias year cells are the time-mean |obs − model| (MAE), matching the panel title
    // and method note — not RMSD. Label it "|error|" so the hover does not misname it.
    const metric = biasMode ? "bias" : "|error|";
    const sign = biasMode && value > 0 ? "+" : "";
    // Bias cells carry a ±1 standard error (std(model − obs)/sqrt(n)); absent on old artifacts.
    const errorText = biasMode && Number.isFinite(standardError) ? ` ± ${formatFixed(standardError, 3)}` : "";
    const countText = Number.isFinite(count) && count > 0 ? ` · n = ${count.toLocaleString("en-US")}` : "";
    return `${metric} ${sign}${formatFixed(value, 3)}${errorText} ${panel.units}${countText}`;
  }
  return Number.isNaN(value) ? "land / no data" : `${formatFixed(value, 3)} ${panel.units}`;
}

// Nearest-point hit-test over the Class-4 obs drawn on this panel (canvas points
// have no DOM nodes, so we project the small filtered set and pick the closest to
// the cursor). The details render inside the panel's fixed readout pill — nothing
// floats with the cursor.
function nearestClass4Record(panel, event, rectangle) {
  if (shared.overlayMode !== "class4" || !panel.class4HitPoints || !panel.class4HitPoints.length) return null;
  const prepared = panel.class4Prepared;
  if (!prepared || !prepared.spatialIndex) return null;
  const selectedMask = panel.class4HitSelected; // null → whole display set is hittable
  const ratio = window.devicePixelRatio || 1;
  const cursorX = (event.clientX - rectangle.left) * ratio;
  const cursorY = (event.clientY - rectangle.top) * ratio;
  const projection = projectionFor(panel);
  const copyOffsets = visibleCopyOffsets(projection, panel.els.overlay);
  const threshold = Math.max((panel.class4PointRadius || 2.2) * ratio + 6 * ratio, 11 * ratio);
  const { cols, rows, start, order } = prepared.spatialIndex;
  const normalizedLongitude = prepared.normalizedLongitude;
  const normalizedLatitude = prepared.normalizedLatitude;
  const { originX, originY, displayWidth, displayHeight } = projection;
  // Threshold radius expressed in normalized coordinates → cursor bucket neighbourhood.
  const deltaColumns = threshold / displayWidth;
  const deltaRows = threshold / displayHeight;
  let nearest = null;
  let nearestDistance = threshold;
  const cursorNy = (cursorY - originY) / displayHeight;
  for (const offset of copyOffsets) {
    const cursorNx = (cursorX - originX) / displayWidth - offset;
    let columnLow = Math.floor((cursorNx - deltaColumns) * cols) - 1;
    let columnHigh = Math.floor((cursorNx + deltaColumns) * cols) + 1;
    let rowLow = Math.floor((cursorNy - deltaRows) * rows) - 1;
    let rowHigh = Math.floor((cursorNy + deltaRows) * rows) + 1;
    columnLow = Math.max(0, columnLow);
    columnHigh = Math.min(cols - 1, columnHigh);
    rowLow = Math.max(0, rowLow);
    rowHigh = Math.min(rows - 1, rowHigh);
    if (columnLow > columnHigh || rowLow > rowHigh) continue;
    for (let row = rowLow; row <= rowHigh; row += 1) {
      const rowBase = row * cols;
      for (let column = columnLow; column <= columnHigh; column += 1) {
        const bucket = rowBase + column;
        for (let s = start[bucket]; s < start[bucket + 1]; s += 1) {
          const id = order[s];
          if (selectedMask && !selectedMask[id]) continue;
          const x = originX + (normalizedLongitude[id] + offset) * displayWidth;
          const y = originY + normalizedLatitude[id] * displayHeight;
          const distance = Math.hypot(x - cursorX, y - cursorY);
          if (distance < nearestDistance) {
            nearestDistance = distance;
            nearest = prepared.points[id];
          }
        }
      }
    }
  }
  return nearest;
}

const CLASS4_PLATFORM_KEYS = ["platform", "satellite", "platform_id", "source", "wmo_platform_code", "sensor"];

// Compact one-line obs details appended to the readout pill, e.g.
// " · obs 0.402 · fcst 0.451 · err 0.049 · argo".
function class4ReadoutSuffix(record, units) {
  const obs = numericOrNaN(record.observation_value);
  const model = numericOrNaN(record.model_value);
  const hasObs = Number.isFinite(obs);
  const hasModel = Number.isFinite(model);
  const error = Number.isFinite(numericOrNaN(record.abs_error))
    ? numericOrNaN(record.abs_error)
    : hasObs && hasModel
      ? Math.abs(model - obs)
      : NaN;
  const parts = [
    `obs ${formatFixed(obs, 3)}`,
    `fcst ${formatFixed(model, 3)}`,
    `err ${formatFixed(error, 3)}${units ? ` ${units}` : ""}`,
  ];
  const platformKey = CLASS4_PLATFORM_KEYS.find((key) => record[key] != null && record[key] !== "");
  if (platformKey) parts.push(String(record[platformKey]));
  else if (record.depth_bin) parts.push(String(record.depth_bin));
  return ` · ${parts.join(" · ")}`;
}

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character];
  });
}

function nearestIndex(coordinates, value) {
  const step = coordinates.length > 1 ? coordinates[1] - coordinates[0] : 1;
  const index = Math.round((value - coordinates[0]) / step);
  if (index < 0 || index >= coordinates.length) return -1;
  if (Math.abs(coordinates[index] - value) > Math.abs(step)) return -1;
  return index;
}

// ---- layout + redraw --------------------------------------------------------

function syncPanelGrid() {
  const grid = elements["panel-grid"];
  // Both swipe and difference collapse to a single shared map hosted by Forecast 1;
  // CSS lays it out as one column and reduces Forecast 2's panel to its picker strip.
  const single = shared.layout === 2 && (shared.displayMode === "swipe" || shared.displayMode === "diff");
  grid.dataset.layout = String(single ? 1 : shared.layout);
  grid.dataset.display = shared.displayMode;
  while (panels.length < shared.layout) {
    const panel = buildPanel(panels.length);
    panels.push(panel);
  }
  grid.innerHTML = "";
  for (let i = 0; i < shared.layout; i += 1) {
    panels[i].container.classList.toggle("head-only", single && i === 1);
    grid.appendChild(panels[i].container);
    refreshPanelControls(panels[i]);
  }
  for (let i = shared.layout; i < panels.length; i += 1) stopParticles(panels[i]);
  if (activePanelIndex >= shared.layout) activePanelIndex = 0;
  markActivePanel();
}

function markActivePanel() {
  panels.forEach((panel, index) => panel.container.classList.toggle("active", index === activePanelIndex && shared.layout > 1));
}

function setActivePanel(index) {
  if (index >= shared.layout) return;
  activePanelIndex = index;
  markActivePanel();
  updateContextRail();
  updateSharedColorbar();
}

// A cached render completes in tens of milliseconds, so showing the scrim straight
// away only produces a flash. Arm a short timer instead and let a fast render cancel
// it, so the scrim appears solely when the wait is long enough to read as a freeze.
// The dataset/variable selects are never disabled: the user must always be able to
// change their mind mid-load.
const LOADING_SCRIM_DELAY_MILLISECONDS = 180;

function setPanelLoading(panel, loading) {
  if (panel.loadingTimer) {
    clearTimeout(panel.loadingTimer);
    panel.loadingTimer = null;
  }
  if (!loading) {
    panel.container.classList.remove("loading");
    panel.els.loading.hidden = true;
    return;
  }
  panel.loadingTimer = setTimeout(() => {
    panel.loadingTimer = null;
    panel.container.classList.add("loading");
    panel.els.loading.hidden = false;
  }, LOADING_SCRIM_DELAY_MILLISECONDS);
}

function resizePanelCanvases(panel) {
  const ratio = window.devicePixelRatio || 1;
  const rectangle = panel.els.wrap.getBoundingClientRect();
  for (const key of ["field", "particles", "overlay"]) {
    const canvas = panel.els[key];
    const width = Math.max(1, Math.round(rectangle.width * ratio));
    const height = Math.max(1, Math.round(rectangle.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    canvas.style.width = `${rectangle.width}px`;
    canvas.style.height = `${rectangle.height}px`;
  }
}

function redrawAllPanels() {
  redrawAllPanelsFrame = 0;
  for (let i = 0; i < shared.layout; i += 1) {
    resizePanelCanvases(panels[i]);
    drawPanel(panels[i]);
    drawOverlays(panels[i]);
    drawTrajectoryFans(panels[i]);
  }
}

function scheduleRedrawAllPanels() {
  if (redrawAllPanelsFrame) return;
  redrawAllPanelsFrame = requestAnimationFrame(redrawAllPanels);
}

async function renderAllPanels() {
  const jobs = [];
  for (let i = 0; i < shared.layout; i += 1) jobs.push(renderPanel(panels[i]));
  await Promise.all(jobs);
  // Panels render concurrently, so the swipe host may have drawn before Forecast 2's
  // field existed; redraw it now that both are ready so the divider composite appears.
  if (isSwipeHost(panels[0])) drawPanel(panels[0]);
}

// ---- shared colorbar --------------------------------------------------------

function rgbCss(rgb) {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}

function updateYearLegend(visible) {
  const legend = elements["year-legend"];
  if (!legend) return;
  legend.hidden = !visible;
  if (!visible) return;
  legend.innerHTML =
    `<span class="item"><span class="swatch" style="background:${rgbCss(landColor(shared.theme))}"></span>land</span>` +
    `<span class="item"><span class="swatch" style="background:${rgbCss(noObsColor(shared.theme))}"></span>ocean, no observations</span>`;
  // Colorbar-adjacent method note for the year raster (the colorbar itself is a canvas).
  attachMethodNote(legend, "year-geography");
}

// The single bottom legend strip always describes the PRIMARY layer actually drawn:
// the field colorbar with no overlay, or the overlay's own key (obs-error ramp,
// eddy/trajectory swatches) while a purpose overlay mutes the field. The right rail
// carries no legend block any more — all legend information lives here.
function updateSharedColorbar() {
  const panel = isDiffView() ? panels[0] : panels[activePanelIndex];
  updateYearLegend(shared.scope === "year");
  const colorbar = elements.colorbar;
  const legend = elements["map-legend"];
  const textColor = themeToken("--ob-viewer-colorbar-text", shared.theme === "light" ? "#14181d" : "#e6edf3");

  // Year scope keeps its own colorbar untouched (design: don't touch year legend).
  if (shared.scope === "year") {
    colorbar.hidden = false;
    hideMapLegend(legend);
    if (!panel || !panel.colormap || !panel.range) {
      elements["layer-info"].textContent = `entire year · zoom ${view.zoom.toFixed(1)}×`;
      return;
    }
    const nStarts = panel.yearMeta && panel.yearMeta.nStarts;
    const biasMode = panel.yearMetric === "bias";
    drawColorbar(colorbar, panel.colormap, panel.range, {
      label: `${biasMode ? "mean (model − obs)" : "mean |obs − model|"} over ${nStarts || "?"} start dates · ${panel.label} (${panel.units})`,
      textColor,
    });
    // The start-date count is already on the colorbar caption above; do not repeat it.
    elements["layer-info"].textContent = `entire year · zoom ${view.zoom.toFixed(1)}×`;
    return;
  }
  if (!panel || !panel.colormap || !panel.range) return;

  const mode = isDiffView() ? "none" : shared.overlayMode;
  if (mode === "class4") {
    // Nothing drawn means an obs-error ramp would be a scale over no data: drop the
    // colorbar and the legend strip and let the overlay note carry the one sentence.
    if (class4EmptyMessage()) {
      colorbar.hidden = true;
      hideMapLegend(legend);
      setLayerInfo(panel);
      return;
    }
    const scale = Math.max(...panels.slice(0, shared.layout).map((candidate) => candidate.class4Scale || 0), 0);
    colorbar.hidden = false;
    drawColorbar(colorbar, CLASS4_COLORMAP, [0, scale || 1], {
      label: `|obs − model| (${panel.units})`,
      textColor,
    });
    renderClass4Legend(legend, panel, scale);
  } else if (mode === "eddies") {
    colorbar.hidden = true;
    renderEddyLegend(legend);
  } else if (mode === "trajectories") {
    colorbar.hidden = true;
    renderTrajectoryLegend(legend);
  } else {
    colorbar.hidden = false;
    hideMapLegend(legend);
    const sameVariable = panels
      .slice(0, shared.layout)
      .every((candidate) => candidate.state.variable === panel.state.variable);
    // Which panel the scale describes leads the caption rather than trailing it: a
    // long variable name is ellipsized on the right, and that qualifier is the part a
    // reader with two panels in front of them cannot do without.
    const prefix = isDiffView() ? "" : shared.layout > 1 ? (sameVariable ? "shared · " : `panel ${activePanelIndex + 1} · `) : "";
    drawColorbar(colorbar, panel.colormap, panel.range, {
      label: `${prefix}${panel.label} (${panel.units})`,
      textColor,
    });
  }
  setLayerInfo(panel);
}

function setLayerInfo(panel) {
  const manifest = manifestFor(panel.state.dataset);
  if (!manifest || !Array.isArray(manifest.start_dates) || !manifest.start_dates.length) {
    elements["layer-info"].textContent = `zoom ${view.zoom.toFixed(1)}× · loading metadata`;
    return;
  }
  // The start date is the drawer's own field and the lead day is spelled out beside
  // the scrubber a few centimetres to the left, so the readout carries neither.
  elements["layer-info"].textContent = `zoom ${view.zoom.toFixed(1)}×`;
}

function hideMapLegend(legend) {
  legend.hidden = true;
  legend.innerHTML = "";
}

function legendHelpAnchor() {
  return `<span class="legend-help"></span>`;
}

function legendSwatch(color, label, count) {
  const countText = count == null ? "" : ` · <strong>${formatCount(count)}</strong>`;
  return `<span class="legend-item"><span class="legend-dot" style="background:${color}"></span>${escapeHtml(label)}${countText}</span>`;
}

function legendLine(color, label) {
  return `<span class="legend-item"><span class="legend-line" style="background:${color}"></span>${escapeHtml(label)}</span>`;
}

// Class-4 obs mode: the obs-error ramp is the primary key (drawn on the colorbar
// canvas). This strip carries the density note (moved out of the right rail), the
// muted-background note and the "?" helper.
function renderClass4Legend(legend, panel, scale) {
  const hostPanel = panels[0];
  const shown = hostPanel ? hostPanel.class4Count || 0 : 0;
  const visibleTotal = hostPanel ? hostPanel.class4VisibleTotal || shown : shown;
  const matched = hostPanel ? hostPanel.class4Matched || 0 : 0;
  const thinned = Boolean(hostPanel && hostPanel.class4Thinned);
  const weak = matched > 0 && matched < 30 ? " · low count, statistic is weak" : "";
  const fullDensityNote = thinned ? " · zoom in for full density" : "";
  const countText = thinned
    ? `<strong>showing ${formatCount(shown)} of ${formatCount(visibleTotal)} obs</strong>${fullDensityNote}`
    : `<strong>${formatCount(matched)} obs</strong>`;
  legend.hidden = false;
  legend.innerHTML =
    `<span class="legend-note">${countText} · scale ≈ ${scale ? scale.toFixed(3) : "n/a"} ${escapeHtml(panel.units)} · region ${escapeHtml(shared.region)}${weak}${legendHelpAnchor()}</span>`;
  attachMethodNote(legend.querySelector(".legend-help"), "class4-legend");
}

// Eddies mode: categorical swatch row faithful to what is drawn — matched pairs (the
// intercomparison neutral colour) and each forecast's only-eddies, or a single census.
function renderEddyLegend(legend) {
  const censuses = overlayData.eddiesCensuses || [];
  const match = overlayData.eddiesMatch;
  const mismatch = Boolean(overlayData.eddiesLeadMismatch);
  let swatches;
  let caption;
  if (shared.layout === 2 && match) {
    const forecast1 = labelFor(panels[0].state.dataset);
    const forecast2 = labelFor(panels[1].state.dataset);
    const lead = (censuses.find(Boolean) || {}).leadDay;
    const meanText = Number.isFinite(match.meanDisplacementKm) ? `${formatFixed(match.meanDisplacementKm, 0)} km` : "n/a";
    swatches =
      legendSwatch(EDDY_MATCHED_COLOR, "Matched pairs", match.matched.length) +
      legendSwatch(forecastColor(0), `Only in ${forecast1}`, match.onlyA.length) +
      legendSwatch(forecastColor(1), `Only in ${forecast2}`, match.onlyB.length);
    caption = `mean centre displacement of matched pairs ${meanText} · lead ${lead ?? "n/a"} (nearest shared)`;
  } else if (shared.layout === 2 && mismatch && censuses[0] && censuses[1]) {
    // No lead day in common between the two forecasts: never cross-match different leads —
    // show each forecast's own census at its own nearest lead, and say so.
    const forecast1 = labelFor(panels[0].state.dataset);
    const forecast2 = labelFor(panels[1].state.dataset);
    swatches =
      legendSwatch(forecastColor(0), `${forecast1} eddies · lead ${censuses[0].leadDay ?? "n/a"}`, censuses[0].detections.length) +
      legendSwatch(forecastColor(1), `${forecast2} eddies · lead ${censuses[1].leadDay ?? "n/a"}`, censuses[1].detections.length);
    caption = "no lead day in common, so each forecast is shown at its own nearest lead and never cross-matched";
  } else if (censuses[0]) {
    swatches = legendSwatch(forecastColor(0), `${labelFor(panels[0].state.dataset)} eddies`, censuses[0].detections.length);
    caption = `single forecast census · lead ${censuses[0].leadDay ?? "n/a"} (nearest available)`;
  } else {
    swatches = `<span class="legend-note">No eddy detections for this selection.</span>`;
    caption = "";
  }
  legend.hidden = false;
  legend.innerHTML =
    `<span class="legend-row">${swatches}</span>` +
    `<span class="legend-note dim">${escapeHtml(caption)}${legendHelpAnchor()}</span>`;
  const census = censuses.find(Boolean);
  attachEddyMethodNote(legend.querySelector(".legend-help"), census && census.parameters);
}

// Trajectories mode: the two forecast colours the fans are drawn in, plus the seed note.
function renderTrajectoryLegend(legend) {
  const forecast1 = labelFor(panels[0].state.dataset);
  let swatches = legendLine(TRAJECTORY_COLORS[0], `${forecast1} drift`);
  if (shared.layout === 2) swatches += legendLine(TRAJECTORY_COLORS[1], `${labelFor(panels[1].state.dataset)} drift`);
  swatches += `<span class="legend-item"><span class="legend-seed">◦</span>seed cluster</span>`;
  const seeded = trajectoryState && trajectoryState.trajectories;
  const caption = seeded
    ? "20 shared seeds advected through each forecast's currents"
    : "click the map to seed a 20-particle cluster";
  legend.hidden = false;
  legend.innerHTML =
    `<span class="legend-row">${swatches}</span>` +
    `<span class="legend-note dim">${escapeHtml(caption)}${legendHelpAnchor()}</span>`;
  attachMethodNote(legend.querySelector(".legend-help"), "trajectories");
}

// ---- context rail -----------------------------------------------------------

// The rail is derived entirely from the Forecast 1 / Forecast 2 pickers (the panels),
// never from its own selectors (item 2). One forecast → its diagnostics. Two with the
// SAME variable → an overlaid comparison (Forecast 2 is the reference for the error
// spectrum). Two with DIFFERENT variables → a small F1/F2 toggle, one forecast at a time.
async function updateContextRail() {
  const forecasts = railForecasts();
  if (!forecasts.length) return;
  await ensureScoresSummary();
  const comparison = forecasts.length === 2 && sameForecastVariable(forecasts[0], forecasts[1]);
  const toggleForecasts = forecasts.length === 2 && !comparison;

  const toggle = elements["rail-forecast-toggle"];
  toggle.hidden = !toggleForecasts;
  if (toggleForecasts) {
    if (shared.railForecast > 1) shared.railForecast = 0;
    for (const button of toggle.querySelectorAll("button")) {
      const index = Number(button.dataset.forecast);
      const selected = index === shared.railForecast;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-pressed", String(selected));
      button.textContent = `F${index + 1}`;
      button.title = `Forecast ${index + 1} · ${labelFor(forecasts[index].state.dataset)}`;
    }
  }

  const shown = comparison ? forecasts : [forecasts[toggleForecasts ? shared.railForecast : 0]];
  elements["rail-subtitle"].textContent = comparison
    ? `${shown.map((p) => labelFor(p.state.dataset)).join(" vs ")} · ${prettyVariable(shown[0])} · ${shared.region}`
    : `${shown[0].label} · ${shared.region}`;

  updateCurrentDepthGateNote(shown);
  renderRailSkill(shown, comparison);
  await renderRailDepthProfile(shown, comparison);
  const yearSection = elements["rail-year-rmsd-section"];
  if (shared.scope === "year") {
    if (yearSection) yearSection.hidden = false;
    renderRailYearRmsd(shown);
    return;
  }
  if (yearSection) yearSection.hidden = true;
  renderRailPsd(shown, comparison);
  renderTrajectoryRail();
  renderRailProvenance(shown);
}

// Data-provenance line(s) at the foot of the rail: which oceanbench pipeline version
// produced the shown forecast(s)' artifacts and when. Manifests published before this
// block existed carry no `provenance`, so nothing is shown for them (the line stays
// hidden). In 2-forecast mode a shared stamp collapses to one line; distinct stamps
// show one line each.
function formatProvenanceLine(provenance) {
  const version = provenance.oceanbench_version || "?";
  const generatedDate = String(provenance.generated_at || "").slice(0, 10);
  const suffix = generatedDate ? ` · generated ${generatedDate}` : "";
  return `data: oceanbench ${version}${suffix}`;
}

function renderRailProvenance(shown) {
  const element = elements["rail-provenance"];
  if (!element) return;
  const lines = [];
  for (const panel of shown) {
    const manifest = manifestFor(panel.state.dataset);
    const provenance = manifest && manifest.provenance;
    if (!provenance) continue;
    const line = formatProvenanceLine(provenance);
    if (!lines.includes(line)) lines.push(line);
  }
  element.textContent = "";
  if (!lines.length) {
    element.hidden = true;
    return;
  }
  lines.forEach((line, index) => {
    if (index > 0) element.appendChild(document.createElement("br"));
    const span = document.createElement("span");
    span.textContent = line;
    element.appendChild(span);
  });
  element.hidden = false;
  attachMethodNote(element, "data-provenance");
}

// RMSD vertical profile (RMSD vs depth) for the shown forecast(s) at the selected lead
// day, so the rail answers "where in the water column is the model wrong at this lead".
// Only 3D variables (temperature, salinity) carry a depth profile: the artifact keys its
// `variables` by the observation standard name, so surface-only channels (SSH, currents)
// simply find no entry and the section stays hidden. The artifact may be absent while the
// backfill is in flight — a missing file resolves to null and hides the section too.
async function renderRailDepthProfile(shown, comparison) {
  const section = elements["rail-depth-profile-section"];
  const slot = elements["rail-depth-profile"];
  const note = elements["rail-depth-profile-note"];
  if (!section || !slot) return;
  const lines = [];
  let unit = "";
  let lead = null;
  for (const panel of shown) {
    const entry = variableEntry(manifestFor(panel.state.dataset), panel.state.variable);
    if (!entry || isVelocityFamilyVariable(panel.state.variable)) continue;
    const url = insightsFor(insightIndex, panel.state.dataset, shared.region).rmsd_by_depth;
    const data = url ? await loadRmsdByDepth(url) : null;
    const profile = data ? rmsdDepthProfile(data, entry.standard_name, shared.leadDay) : null;
    if (!profile) continue;
    lead = lead == null ? profile.lead : lead;
    unit = unit || entry.units || "";
    lines.push({
      label: comparison ? `Forecast ${panel.index + 1} · ${labelFor(panel.state.dataset)}` : "RMSD vs depth",
      color: forecastColor(panel.index),
      bins: profile.bins,
    });
  }
  if (!lines.length) {
    section.hidden = true;
    slot.innerHTML = "";
    if (note) note.textContent = "";
    return;
  }
  section.hidden = false;
  slot.innerHTML = rmsdByDepthSVG(lines, {
    title: comparison ? "RMSD vs depth (both forecasts)" : "RMSD vs depth",
    unit,
  });
  if (note) {
    note.textContent = `Class-4 RMSD per depth bin at lead day ${lead ?? shared.leadDay}, pooled over all match-ups of the year (same method as the official scores).`;
  }
  wireCursorTooltip(slot);
}

// RMSD by start date, one line per visible forecast at the selected lead day. Clicking
// a point drills down into single-forecast scope with that start date selected.
async function renderRailYearRmsd(shown) {
  const slot = elements["rail-year-rmsd"];
  const note = elements["rail-year-rmsd-note"];
  const biasMode = shared.yearMetric === "bias";
  // Keep the section title in step with the metric (the "?" button is a later child,
  // so only the leading text node is retitled).
  const yearHeading = elements["rail-year-rmsd-section"].querySelector("h3");
  if (yearHeading && yearHeading.firstChild) {
    yearHeading.firstChild.textContent = biasMode ? "Bias by start date" : "RMSD by start date";
  }
  const lines = [];
  let unit = "";
  let missing = 0;
  // Lead-independent y-bound (max across ALL leads, over the shown datasets): the axis
  // stays fixed while the lead slider scrubs, so the curve visibly grows/shifts within a
  // constant frame. yearRmsdSeriesMax is a pure function of the loaded artifact, so this
  // effectively only changes with the dataset/variable/region/metric selection.
  let yBound = 0;
  for (const panel of shown) {
    if (isSurfaceCurrentVariable(panel.state.variable)) continue; // 15 m obs only; covered by switch note
    const url = insightsFor(insightIndex, panel.state.dataset, shared.region).year_rmsd_by_start;
    const mapping = url ? yearVariableMapping(panel.state.variable) : null;
    const rmsd = url ? await loadYearRmsd(url) : null;
    const entry = rmsd && mapping ? yearRmsdSeries(rmsd, mapping.short, shared.leadDay) : null;
    // In bias mode a series without a parallel bias array degrades gracefully (skipped,
    // counted as missing) — the |error| path is unaffected.
    if (!entry || (biasMode && !entry.bias)) {
      missing += 1;
      continue;
    }
    yBound = Math.max(yBound, yearRmsdSeriesMax(rmsd, mapping.short, { signed: biasMode }));
    unit = unit || (mapping.unit ? mapping.unit : "");
    lines.push({
      label: `${labelFor(panel.state.dataset)}${mapping.component ? " · u" : ""}`,
      color: forecastColor(panel.index),
      panelIndex: panel.index,
      dates: entry.dates,
      rmsd: biasMode ? entry.bias : entry.rmsd,
      // Parallel 95% CI edges for the active metric; null on old artifacts (chart draws no band).
      ciLow: biasMode ? entry.biasCiLow : entry.ciLow,
      ciHigh: biasMode ? entry.biasCiHigh : entry.ciHigh,
    });
  }
  slot.innerHTML = rmsdByStartSVG(lines, {
    title: biasMode ? "Bias by start date" : "RMSD by start date",
    unit,
    signed: biasMode,
    yBound,
  });
  note.textContent = lines.length
    ? biasMode
      ? "Pooled mean(model − obs) per start date, same method as the official scores. Click a point to open that start date."
      : "Class-4 RMSD per start date, same method as the official scores (pooled over all match-ups for that start). Click a point to open that start date."
    : biasMode
      ? "Bias by start not available for this dataset/region."
      : "Year RMSD-by-start not available for this dataset/region.";
  wireCursorTooltip(slot);
  wireYearRmsdDrilldown(slot, lines);
}

function wireYearRmsdDrilldown(slot, lines) {
  const svg = slot.querySelector("svg");
  if (!svg) return;
  svg.querySelectorAll(".year-point").forEach((point) => {
    point.style.cursor = "pointer";
    point.addEventListener("click", () => {
      const date = point.getAttribute("data-date");
      if (date) drillDownToStartDate(date);
    });
  });
}

// Switch from year scope to single-forecast scope, selecting the clicked start date
// (matched against the primary forecast's manifest start_dates; nearest if inexact).
function drillDownToStartDate(date) {
  const manifest = manifestFor(panels[0].state.dataset);
  const dates = (manifest && manifest.start_dates) || [];
  let index = dates.findIndex((candidate) => String(candidate).slice(0, 10) === date);
  if (index < 0 && dates.length) {
    const target = Date.parse(date);
    let best = 0;
    let bestDelta = Infinity;
    dates.forEach((candidate, i) => {
      const delta = Math.abs(Date.parse(String(candidate).slice(0, 10)) - target);
      if (delta < bestDelta) {
        bestDelta = delta;
        best = i;
      }
    });
    index = best;
  }
  if (index >= 0) shared.startIndex = index;
  setScope("single");
}

// When a surface current variable is selected in an obs-based context (Class-4 overlay
// or year scope), there are no honest surface observations to compare against: the
// drifter obs sit at 15 m. Explain this and offer a one-click switch to the 15 m field.
function updateCurrentDepthGateNote(shown) {
  const note = elements["rail-current-depth-note"];
  if (!note) return;
  const obsContext = shared.overlayMode === "class4" || shared.scope === "year";
  const gated = obsContext && shown.filter((panel) => isSurfaceCurrentVariable(panel.state.variable));
  if (!obsContext || !gated.length) {
    note.hidden = true;
    note.innerHTML = "";
    return;
  }
  note.hidden = false;
  note.innerHTML =
    `<p>Current observations (drifters) are measured at 15&nbsp;m depth. Switch to 15&nbsp;m currents to compare against them.</p>` +
    `<button type="button" class="ghost-button" id="rail-switch-15m-currents">Switch to 15&nbsp;m currents</button>`;
  const button = note.querySelector("#rail-switch-15m-currents");
  if (button) button.addEventListener("click", () => switchShownPanelsTo15mCurrents(gated));
}

async function switchShownPanelsTo15mCurrents(gatedPanels) {
  const targets = gatedPanels && gatedPanels.length ? gatedPanels : panels.slice(0, shared.layout);
  for (const panel of targets) {
    if (!isSurfaceCurrentVariable(panel.state.variable)) continue;
    panel.state.variable = matching15mCurrentVariable(panel.state.variable);
    refreshPanelControls(panel);
    await renderPanel(panel);
  }
  await updateContextRail();
  updateCurrentsControlVisibility();
  writeHash();
}

function railForecasts() {
  const scope = shared.layout === 2 ? panels.slice(0, 2) : panels.slice(0, 1);
  return scope.filter(Boolean);
}

function sameForecastVariable(a, b) {
  return a.state.variable === b.state.variable;
}

function prettyVariable(panel) {
  const manifest = manifestFor(panel.state.dataset);
  const entry = manifest && variableEntry(manifest, panel.state.variable);
  if (isCurrentsVariable(panel.state.variable)) return `currents (${currentsDepthLabel(panel.state.variable)})`;
  return entry ? `${prettyName(entry.standard_name)} · ${entry.depth}` : panel.state.variable;
}

// Obs-based skill (Class-4 RMSD vs observations) for a forecast's selected variable.
// Returns { rows, unit, n } or null when no observation-based metric exists (item 4).
function obsSkillSeries(panel) {
  // Surface currents have no 15 m drifter obs to compare against — the switch note
  // handles this case, so emit no skill curve for them.
  if (isSurfaceCurrentVariable(panel.state.variable)) return null;
  const manifest = manifestFor(panel.state.dataset);
  const entry = manifest && variableEntry(manifest, panel.state.variable);
  if (!entry) return null;
  const depthKeys = scoreDepthKeys(entry);
  const scoreVariables = isCurrentsVariable(panel.state.variable)
    ? ["eastward_sea_water_velocity", "northward_sea_water_velocity"]
    : [entry.standard_name];
  const challenger = scoreProductKey(panel.state.dataset);
  const rowsByVariable = new Map(scoreVariables.map((variable) => [variable, []]));
  let unit = "";
  let starts = 0;
  for (const row of scoresSummary) {
    if (row.metric !== "class4_rmsd") continue;
    if (!rowsByVariable.has(row.variable)) continue;
    if (row.reference !== "observations") continue; // observation-based only, no gridded fallback
    if (depthKeys.length && !depthKeys.includes(row.depth)) continue;
    if (row.challenger !== challenger) continue;
    if (shared.region && row.region && row.region !== shared.region) continue;
    unit = row.unit || unit;
    starts = Math.max(starts, Number(row.n_starts) || 0);
    rowsByVariable.get(row.variable).push(row);
  }
  if (![...rowsByVariable.values()].some((rows) => rows.length)) return null;
  return { rowsByVariable, unit, n: starts };
}

function renderRailSkill(shown, comparison) {
  const series = new Map();
  const labels = new Map();
  const colors = new Map();
  let unit = "";
  const notes = [];
  try {
    for (const panel of shown) {
      const skill = obsSkillSeries(panel);
      const key = scoreProductKey(panel.state.dataset);
      if (!skill) {
        if (isSurfaceCurrentVariable(panel.state.variable)) continue; // covered by the 15 m switch note
        const suffix = isCurrentsVariable(panel.state.variable)
          ? `currents at ${currentsDepthLabel(panel.state.variable)}`
          : "this variable";
        notes.push(`${labelFor(panel.state.dataset)}: no scored observations for ${suffix}`);
        continue;
      }
      unit = skill.unit || unit;
      for (const [variable, rows] of skill.rowsByVariable) {
        if (!rows.length) continue;
        const component = variable === "eastward_sea_water_velocity" ? "eastward" : "northward";
        const seriesKey = isCurrentsVariable(panel.state.variable)
          ? `forecast-${panel.index}:${key}:${component}`
          : `forecast-${panel.index}:${key}`;
        const aggregated = aggregateLeadSeries(new Map([[seriesKey, rows]]));
        if (aggregated.has(seriesKey)) series.set(seriesKey, aggregated.get(seriesKey));
        colors.set(seriesKey, forecastColor(panel.index));
        labels.set(
          seriesKey,
          isCurrentsVariable(panel.state.variable)
            ? `${comparison ? `Forecast ${panel.index + 1} · ${labelFor(panel.state.dataset)} · ` : ""}${
                component === "eastward" ? "u (eastward)" : "v (northward)"
              }`
            : comparison
              ? `Forecast ${panel.index + 1} · ${labelFor(panel.state.dataset)}`
              : "RMSD vs observations",
        );
      }
      // n_starts is available from the summary, so report the real number of start dates
      // behind the aggregate (item 5). TODO(pipeline): expose per-lead matchup counts too.
      notes.push(
        `Forecast ${panel.index + 1} · ${labelFor(panel.state.dataset)}: n = ${skill.n} start dates${
          skill.n < 10 ? " (low, weak statistic)" : ""
        }`,
      );
      if (isCurrentsVariable(panel.state.variable)) notes.push("Current speed map points are paired u/v speeds; curves show u/v component RMSD.");
    }
  } catch (error) {
    console.error("Cannot render observation-based skill", error);
    notes.length = 0;
    notes.push("No scored observations for this variable/product");
  }
  // With nothing to plot the caption below already says so, and it says which
  // product it is saying it about. Drawing an empty chart whose only content is a
  // shorter version of that same sentence stated the fact twice.
  elements["rail-lead-curve"].innerHTML = series.size || !notes.length
    ? leadCurveSVG(series, {
        unit,
        labels,
        colors,
        title: comparison ? "RMSD vs lead day (both forecasts)" : "RMSD vs lead day",
        emptyMessage: "no scored observations for this variable/product",
      })
    : "";
  elements["rail-skill-note"].textContent = notes.join(" · ");
  wireCursorTooltip(elements["rail-lead-curve"]);
}

// ---- live PSD: explicit size-capped rectangle at the model's native grid -----
//
// The spectrum is computed over an explicit rectangle drawn on the map — draggable,
// resizable, shared between both forecasts in compare mode. Its size is HARD-CAPPED at
// what the finest (native) pyramid grid honestly resolves with the 256-cell FFT budget
// (≈ 256 × finest cell size per axis), so the PSD is ALWAYS computed at native
// resolution from a windowed tile-cropped read — downsampled spectra never exist.

const PSD_FFT_CELLS = 256; // matches psd.js MAX_SIDE
const PSD_MIN_CELLS = 32; // minimum native cells across for a meaningful FFT
const PSD_DEFAULT_WIDTH_DEG = 10;
const PSD_FLASH_MILLISECONDS = 700;

// Current cap/min (degrees) for the visible forecasts; refreshed by ensurePsdBox so
// resize gestures clamp against the latest model pair. A shared box must be valid for
// BOTH models at their native grids, so:
//   sharedMax = min over models of (256 × native cell size)  — the FINER model's max,
//               so the fine model is never decimated;
//   sharedMin = max over models of (32 × native cell size)   — the COARSER model's min,
//               so the coarse model still has ≥32 of its own native cells.
// For very disparate pairs sharedMin > sharedMax: no box is fair to both. Then the box
// is degenerate (psdBoxLimits.degenerate) and PSD-compare is disabled with a note.
let psdBoxLimits = { capDeg: PSD_FFT_CELLS, minDeg: 0.5, degenerate: false, resolutionLabels: [] };
let psdBoxFlashUntil = 0;

function finestCellDegFor(slug) {
  const manifest = manifestFor(slug);
  if (!manifest || !Array.isArray(manifest.levels) || !manifest.levels.length) return null;
  return Math.min(...manifest.levels.map((level) => level.cell_size_deg));
}

// Human label for a grid cell size: 1/12° for fractional-degree grids, 0.5° otherwise.
function cellDegreesLabel(cellDeg) {
  const inverse = 1 / cellDeg;
  if (inverse > 1.01 && Math.abs(inverse - Math.round(inverse)) < 0.05) return `1/${Math.round(inverse)}°`;
  return `${Number(cellDeg.toFixed(2))}°`;
}

// Create the box if absent (centred in the viewport) and clamp it to the current cap —
// also handles switching to a coarser/finer model pair: the box persists, only its
// limits move. Returns the box.
function ensurePsdBox(shown) {
  const cells = (shown || panels.slice(0, shared.layout))
    .map((panel) => finestCellDegFor(panel.state.dataset))
    .filter((value) => value != null);
  if (cells.length) {
    const capDeg = PSD_FFT_CELLS * Math.min(...cells); // finer model's max (smaller in degrees)
    const minDeg = PSD_MIN_CELLS * Math.max(...cells); // coarser model's min (larger in degrees)
    const degenerate = minDeg > capDeg + 1e-9;
    const resolutionLabels = [...new Set(cells.map((cell) => cellDegreesLabel(cell)))];
    psdBoxLimits = { capDeg, minDeg, degenerate, resolutionLabels };
  }
  if (!shared.psdBox) {
    const viewport = currentViewport();
    const lon = ((viewport.minX + viewport.maxX) / 2) * 360 - 180;
    const lat = 90 - ((viewport.minY + viewport.maxY) / 2) * 180;
    const width = Math.min(PSD_DEFAULT_WIDTH_DEG, psdBoxLimits.capDeg);
    shared.psdBox = { lon, lat, w: width, h: width };
    shared.psdBoxRequest = { w: PSD_DEFAULT_WIDTH_DEG, h: PSD_DEFAULT_WIDTH_DEG };
  }
  clampPsdBox(false);
  return shared.psdBox;
}

// Clamp size to [min, cap] and keep the box on the globe. Returns true when the SIZE
// was reduced by the cap (used to trigger the "max size" flash during a resize).
function clampPsdBox(fromResize) {
  const box = shared.psdBox;
  if (!box) return false;
  // A resize IS the request; anything else re-applies the last request first, so a box
  // pushed up by a stricter minimum returns to its requested size when that minimum drops.
  if (fromResize || !shared.psdBoxRequest) shared.psdBoxRequest = { w: box.w, h: box.h };
  else {
    box.w = shared.psdBoxRequest.w;
    box.h = shared.psdBoxRequest.h;
  }
  const capped = box.w > psdBoxLimits.capDeg + 1e-9 || box.h > psdBoxLimits.capDeg + 1e-9;
  box.w = Math.min(psdBoxLimits.capDeg, Math.max(psdBoxLimits.minDeg, box.w));
  box.h = Math.min(psdBoxLimits.capDeg, Math.max(psdBoxLimits.minDeg, box.h));
  box.lon = ((box.lon + 180) % 360 + 360) % 360 - 180;
  box.lat = Math.min(90 - box.h / 2, Math.max(-90 + box.h / 2, box.lat));
  if (capped && fromResize) psdBoxFlashUntil = performance.now() + PSD_FLASH_MILLISECONDS;
  return capped;
}

function psdBoxWorldRect() {
  const box = shared.psdBox;
  if (!box) return null;
  return {
    nx0: (box.lon - box.w / 2 + 180) / 360,
    nx1: (box.lon + box.w / 2 + 180) / 360,
    nyTop: (90 - (box.lat + box.h / 2)) / 180,
    nyBottom: (90 - (box.lat - box.h / 2)) / 180,
  };
}

// The rectangle is a spectrum tool, not an overlay mode: it shows on every panel in
// single-forecast scope (incl. swipe/difference — one shared box drives both spectra).
function psdBoxVisible() {
  return shared.psdEnabled && shared.scope !== "year" && Boolean(shared.psdBox) && !psdBoxLimits.degenerate;
}

function drawPsdBox(panel, context, projection) {
  const rect = psdBoxWorldRect();
  if (!rect) return;
  const ratio = window.devicePixelRatio || 1;
  const accent = shared.theme === "light" ? "#046293" : "#38bdf8";
  const flashing = performance.now() < psdBoxFlashUntil;
  const border = flashing ? "#ff6b6b" : accent;
  const copyOffsets = visibleCopyOffsets(projection, panel.els.overlay);
  context.save();
  for (const offset of copyOffsets) {
    const topLeft = projection.project(rect.nx0 + offset, rect.nyTop);
    const bottomRight = projection.project(rect.nx1 + offset, rect.nyBottom);
    const width = bottomRight.x - topLeft.x;
    const height = bottomRight.y - topLeft.y;
    context.fillStyle = shared.theme === "light" ? "rgba(4, 98, 147, 0.07)" : "rgba(56, 189, 248, 0.08)";
    context.fillRect(topLeft.x, topLeft.y, width, height);
    context.strokeStyle = border;
    context.lineWidth = (flashing ? 2.4 : 1.5) * ratio;
    context.setLineDash(flashing ? [] : [6 * ratio, 4 * ratio]);
    context.strokeRect(topLeft.x, topLeft.y, width, height);
    // Handles: corners + edge midpoints.
    context.setLineDash([]);
    const half = 3.5 * ratio;
    context.fillStyle = shared.theme === "light" ? "#ffffff" : "#10151f";
    for (const [hx, hy] of psdHandlePoints(topLeft, bottomRight)) {
      context.fillRect(hx - half, hy - half, 2 * half, 2 * half);
      context.strokeRect(hx - half, hy - half, 2 * half, 2 * half);
    }
    if (flashing) {
      context.font = `${11 * ratio}px system-ui, sans-serif`;
      context.fillStyle = border;
      context.textAlign = "center";
      context.textBaseline = "bottom";
      context.fillText("max size for native-resolution spectrum", (topLeft.x + bottomRight.x) / 2, topLeft.y - 6 * ratio);
    }
  }
  context.restore();
  if (flashing) setTimeout(() => scheduleRedrawAllPanels(), PSD_FLASH_MILLISECONDS + 30);
}

function psdHandlePoints(topLeft, bottomRight) {
  const midX = (topLeft.x + bottomRight.x) / 2;
  const midY = (topLeft.y + bottomRight.y) / 2;
  return [
    [topLeft.x, topLeft.y], [bottomRight.x, topLeft.y], [topLeft.x, bottomRight.y], [bottomRight.x, bottomRight.y],
    [midX, topLeft.y], [midX, bottomRight.y], [topLeft.x, midY], [bottomRight.x, midY],
  ];
}

// Hit test in canvas pixels: corner/edge handles (resize) first, then interior (move).
// Returns { type: "resize", left, right, top, bottom } | { type: "move" } | null.
function psdBoxHitTest(panel, cursorX, cursorY, projection) {
  if (!psdBoxVisible()) return null;
  const rect = psdBoxWorldRect();
  const ratio = window.devicePixelRatio || 1;
  const grab = 9 * ratio;
  const copyOffsets = visibleCopyOffsets(projection, panel.els.overlay);
  for (const offset of copyOffsets) {
    const topLeft = projection.project(rect.nx0 + offset, rect.nyTop);
    const bottomRight = projection.project(rect.nx1 + offset, rect.nyBottom);
    const nearLeft = Math.abs(cursorX - topLeft.x) <= grab;
    const nearRight = Math.abs(cursorX - bottomRight.x) <= grab;
    const nearTop = Math.abs(cursorY - topLeft.y) <= grab;
    const nearBottom = Math.abs(cursorY - bottomRight.y) <= grab;
    const withinX = cursorX >= topLeft.x - grab && cursorX <= bottomRight.x + grab;
    const withinY = cursorY >= topLeft.y - grab && cursorY <= bottomRight.y + grab;
    if (withinX && withinY && (nearLeft || nearRight || nearTop || nearBottom)) {
      return { type: "resize", left: nearLeft, right: nearRight && !nearLeft, top: nearTop, bottom: nearBottom && !nearTop };
    }
    if (cursorX > topLeft.x && cursorX < bottomRight.x && cursorY > topLeft.y && cursorY < bottomRight.y) {
      return { type: "move" };
    }
  }
  return null;
}

function psdCursorFor(hit) {
  if (!hit) return "";
  if (hit.type === "move") return "move";
  const horizontal = hit.left || hit.right;
  const vertical = hit.top || hit.bottom;
  if (horizontal && vertical) return (hit.left && hit.top) || (hit.right && hit.bottom) ? "nwse-resize" : "nesw-resize";
  return horizontal ? "ew-resize" : "ns-resize";
}

// Apply a drag delta (canvas px) to the box for the gesture captured at pointerdown.
function applyPsdBoxDrag(drag, deltaX, deltaY) {
  const dLon = (deltaX / drag.projection.displayWidth) * 360;
  const dLat = -(deltaY / drag.projection.displayHeight) * 180;
  const start = drag.startBox;
  const box = shared.psdBox;
  if (drag.hit.type === "move") {
    box.lon = start.lon + dLon;
    box.lat = start.lat + dLat;
    clampPsdBox(false);
    return;
  }
  // Resize: the grabbed edges follow the cursor; opposite edges stay fixed.
  let west = start.lon - start.w / 2;
  let east = start.lon + start.w / 2;
  let south = start.lat - start.h / 2;
  let north = start.lat + start.h / 2;
  if (drag.hit.left) west += dLon;
  if (drag.hit.right) east += dLon;
  if (drag.hit.top) north += dLat;
  if (drag.hit.bottom) south += dLat;
  if (east - west < psdBoxLimits.minDeg) {
    if (drag.hit.left) west = east - psdBoxLimits.minDeg;
    else east = west + psdBoxLimits.minDeg;
  }
  if (north - south < psdBoxLimits.minDeg) {
    if (drag.hit.bottom) south = north - psdBoxLimits.minDeg;
    else north = south + psdBoxLimits.minDeg;
  }
  // Cap: clamp the grabbed edge so the fixed edge stays put, and flash the hint.
  if (east - west > psdBoxLimits.capDeg) {
    if (drag.hit.left) west = east - psdBoxLimits.capDeg;
    else east = west + psdBoxLimits.capDeg;
    psdBoxFlashUntil = performance.now() + PSD_FLASH_MILLISECONDS;
  }
  if (north - south > psdBoxLimits.capDeg) {
    if (drag.hit.bottom) south = north - psdBoxLimits.capDeg;
    else north = south + psdBoxLimits.capDeg;
    psdBoxFlashUntil = performance.now() + PSD_FLASH_MILLISECONDS;
  }
  box.w = east - west;
  box.h = north - south;
  box.lon = (east + west) / 2;
  box.lat = (north + south) / 2;
  clampPsdBox(true);
}

// Windowed finest-level reads for the PSD rectangle (shares readLayerWindow with the
// trajectory regional fetch). Memoised with a small LRU keyed by dataset/variable/
// lead/box; the underlying compressed tiles are cached by the zarr store anyway.
const psdWindowCache = new Map();
const PSD_WINDOW_CACHE_LIMIT = 8;

function psdWindowKey(slug, variable, level, start, leadIndex, boxRange) {
  const rounded = [boxRange.lonMin, boxRange.lonMax, boxRange.latMin, boxRange.latMax]
    .map((value) => value.toFixed(2))
    .join(",");
  return `${slug}|${variable}|${level}|${start}|${leadIndex}|${rounded}`;
}

async function psdWindowRead(slug, variable, level, start, leadIndex, boxRange) {
  const key = psdWindowKey(slug, variable, level, start, leadIndex, boxRange);
  if (psdWindowCache.has(key)) {
    const hit = psdWindowCache.get(key);
    psdWindowCache.delete(key);
    psdWindowCache.set(key, hit);
    return hit;
  }
  const promise = (async () => {
    const coordinates = await loadCoordinates(slug, level);
    const window = await readLayerWindow(
      stores.get(slug),
      { variable, level, startIndex: start, leadIndex },
      coordinates.latitudes,
      coordinates.longitudes,
      boxRange,
    );
    if (!window) return null;
    const latitudes = Array.from({ length: window.height }, (_, i) => window.lat0 + i * window.latStep);
    const longitudes = Array.from({ length: window.width }, (_, j) => window.lon0 + j * window.lonStep);
    return { field: { data: window.data, width: window.width, height: window.height }, latitudes, longitudes };
  })();
  psdWindowCache.set(key, promise);
  while (psdWindowCache.size > PSD_WINDOW_CACHE_LIMIT) psdWindowCache.delete(psdWindowCache.keys().next().value);
  return promise;
}

// Native-grid field for one panel over the PSD rectangle (speed magnitude for the
// derived currents variables). Returns { field, latitudes, longitudes, cellDeg } | null.
async function psdSourceFor(panel, boxRange) {
  const manifest = manifestFor(panel.state.dataset);
  if (!manifest) return null;
  const cellDeg = finestCellDegFor(panel.state.dataset);
  const levels = [...manifest.levels].sort((a, b) => a.cell_size_deg - b.cell_size_deg);
  const level = levels[0].level;
  const start = Math.min(shared.startIndex, manifest.start_dates.length - 1);
  const leadIndex = shared.leadDay - 1;
  try {
    if (isCurrentsVariable(panel.state.variable)) {
      const components = currentDepthVariables(panel);
      if (!(components.u in manifest.variables)) return null;
      const [u, v] = await Promise.all([
        psdWindowRead(panel.state.dataset, components.u, level, start, leadIndex, boxRange),
        psdWindowRead(panel.state.dataset, components.v, level, start, leadIndex, boxRange),
      ]);
      if (!u || !v) return null;
      return { field: speedMagnitudeField(u.field, v.field), latitudes: u.latitudes, longitudes: u.longitudes, cellDeg };
    }
    if (!variableExists(manifest, panel.state.variable)) return null;
    const record = await psdWindowRead(panel.state.dataset, panel.state.variable, level, start, leadIndex, boxRange);
    if (!record) return null;
    return { ...record, cellDeg };
  } catch {
    return null;
  }
}

// Drop the error-spectrum points below the coarser model's resolution limit:
// wavelengths shorter than 2× the coarser native cell size (km at the box centre
// latitude, zonal/meridional mean — the same convention as the spectrum's cellKm).
function truncateToCommonScales(spectrum, cellDegrees, centreLatitudeDegrees) {
  if (!spectrum) return null;
  const coarsest = Math.max(...cellDegrees.filter(Number.isFinite));
  if (!Number.isFinite(coarsest) || coarsest <= 0) return spectrum;
  const kmPerDegree = 111.32 * (1 + Math.cos((centreLatitudeDegrees * Math.PI) / 180)) / 2;
  const cutoffMetres = 2 * coarsest * kmPerDegree * 1000;
  const wavelength = [];
  const power = [];
  for (let i = 0; i < spectrum.wavelength.length; i += 1) {
    if (spectrum.wavelength[i] >= cutoffMetres) {
      wavelength.push(spectrum.wavelength[i]);
      power.push(spectrum.power[i]);
    }
  }
  if (!wavelength.length) return null;
  return { ...spectrum, wavelength, power };
}

let psdRenderToken = 0;

async function renderRailPsd(shown, comparison) {
  const token = ++psdRenderToken;
  updatePsdToggle();
  if (!shared.psdEnabled || shared.scope === "year") {
    elements["rail-spectra"].innerHTML = "";
    elements["rail-psd-note"].textContent = "";
    return;
  }
  const box = ensurePsdBox(shown);
  if (psdBoxLimits.degenerate) {
    // No box size is fair to both models simultaneously (e.g. 1° vs 1/12°): the coarse
    // model's 32-cell minimum is larger than the fine model's native-resolution cap.
    scheduleRedrawAllPanels(); // drop the now-hidden box from the map (psdBoxVisible is false)
    elements["rail-spectra"].innerHTML = "";
    const pair = psdBoxLimits.resolutionLabels.join(" vs ");
    elements["rail-psd-note"].textContent =
      `Resolutions too different for a shared-box spectrum${pair ? ` (${pair})` : ""}. ` +
      "Switch to a single forecast to inspect each.";
    return;
  }
  const boxRange = {
    latMin: box.lat - box.h / 2,
    latMax: box.lat + box.h / 2,
    lonMin: box.lon - box.w / 2,
    lonMax: box.lon + box.w / 2,
  };
  // The box expressed as the normalized-world viewport psd.js already consumes; its
  // longitude frame matches the continuous axis readLayerWindow returns.
  const boxViewport = {
    minX: (boxRange.lonMin + 180) / 360,
    maxX: (boxRange.lonMax + 180) / 360,
    minY: (90 - boxRange.latMax) / 180,
    maxY: (90 - boxRange.latMin) / 180,
  };
  const curves = [];
  const sources = [];
  for (const panel of shown) {
    const source = await psdSourceFor(panel, boxRange);
    if (token !== psdRenderToken) return; // stale (box moved / view changed again)
    if (!source) continue;
    const spectrum = boxPowerSpectrum(source.field, source.latitudes, source.longitudes, boxViewport);
    sources.push({ panel, spectrum, source });
    if (spectrum) {
      curves.push({
        label: comparison ? `Forecast ${panel.index + 1} · ${labelFor(panel.state.dataset)}` : labelFor(panel.state.dataset),
        color: forecastColor(panel.index),
        ...spectrum,
      });
    }
  }
  if (comparison && sources.length === 2 && sources[0].source.field && sources[1].source.field) {
    const [a, b] = sources.map((entry) => entry.source);
    const alignedB = resampleOntoGrid(b.field, b.latitudes, b.longitudes, a.latitudes, a.longitudes);
    let errorSpectrum = differenceBoxSpectrum(a.field, a.latitudes, a.longitudes, alignedB, boxViewport, true);
    // A difference spectrum is only meaningful over the commonly-resolved scales: below
    // 2× the COARSER model's native cell size the "difference" is interpolation artifact,
    // not model disagreement, so the error curve is truncated there. The two individual
    // model curves stay full-range (each to its own native limit).
    errorSpectrum = truncateToCommonScales(errorSpectrum, [a.cellDeg, b.cellDeg], box.lat);
    if (errorSpectrum) {
      curves.push({ label: `error (F1−F2)`, color: SERIES_COLORS.error, dashed: true, ...errorSpectrum });
    }
  }
  if (token !== psdRenderToken) return;
  elements["rail-spectra"].innerHTML = psdSpectraSVG(curves, {
    title: comparison ? "Live power spectrum (both forecasts)" : "Live power spectrum",
  });
  // Caption: box dimensions + native grid spacing + resolved wavelength range.
  const gridLabels = [...new Set(sources.filter((entry) => entry.spectrum).map((entry) => cellDegreesLabel(entry.source.cellDeg)))];
  let wavelengthMin = Infinity;
  let wavelengthMax = 0;
  for (const curve of curves) {
    for (const metres of curve.wavelength) {
      if (metres < wavelengthMin) wavelengthMin = metres;
      if (metres > wavelengthMax) wavelengthMax = metres;
    }
  }
  const kmRange =
    Number.isFinite(wavelengthMin) && wavelengthMax > 0
      ? `resolves ≈ ${Math.round(wavelengthMin / 1000)} to ${Math.round(wavelengthMax / 1000)} km`
      : "";
  const oceanFractions = sources
    .filter((entry) => entry.spectrum && Number.isFinite(entry.spectrum.oceanFraction))
    .map((entry) => entry.spectrum.oceanFraction);
  const oceanFraction = oceanFractions.length ? Math.min(...oceanFractions) : NaN;
  const oceanLabel = Number.isFinite(oceanFraction) ? `${Math.round(oceanFraction * 100)}% ocean · ` : "";
  const landFraction = Number.isFinite(oceanFraction) ? 1 - oceanFraction : 0;
  const landWarning =
    landFraction > 0.25
      ? `⚠ ${Math.round(landFraction * 100)}% land, so the spectrum is damped by mean-fill. Prefer an open-ocean box. `
      : "";
  elements["rail-psd-note"].textContent = curves.length
    ? `${landWarning}box ${box.w.toFixed(1)}° × ${box.h.toFixed(1)}° · ${oceanLabel}native ${gridLabels.join(" & ")} grid${kmRange ? " · " + kmRange : ""} · drag the box on the map, resize by its handles`
    : "Move the box over ocean to compute a spectrum (boxed area is mostly land).";
  wireCursorTooltip(elements["rail-spectra"]);
}

function updatePsdToggle() {
  const section = document.getElementById("rail-spectra-section");
  const button = elements["psd-toggle"];
  if (!section || !button) return;
  const enabled = shared.psdEnabled && shared.scope !== "year";
  section.classList.toggle("psd-disabled", !enabled);
  button.setAttribute("aria-pressed", enabled ? "true" : "false");
  button.setAttribute("aria-label", enabled ? "Hide live power spectrum" : "Show live power spectrum");
  button.title = enabled ? "Hide live power spectrum" : "Show live power spectrum";
  button.textContent = enabled ? "Hide" : "Show";
}

function setPsdEnabled(enabled) {
  if (shared.psdEnabled === enabled) return;
  shared.psdEnabled = enabled;
  if (enabled) ensurePsdBox(panels.slice(0, shared.layout));
  updatePsdToggle();
  redrawOverlaysAll();
  updateContextRail();
  writeHash();
}

function currentViewport() {
  const host = panels[0];
  const projection = projectionFor(host);
  return visibleViewport(projection, host.els.field);
}

// Cursor-following tooltip: snap the crosshair to the nearest data x, list the value
// for each series under the cursor, and place the tooltip right next to the pointer —
// no delay, no fixed corner (rail chart interaction requirement).
function wireCursorTooltip(container) {
  const svg = container.querySelector("svg");
  if (!svg) return;
  const crosshair = svg.querySelector(".chart-crosshair");
  const tooltip = svg.querySelector(".chart-tooltip");
  const rect = tooltip ? tooltip.querySelector("rect") : null;
  const points = [...svg.querySelectorAll(".chart-point")];
  if (!crosshair || !tooltip || !rect || !points.length) return;
  const bySeries = new Map();
  for (const point of points) {
    const line = point.dataset.line || "";
    if (!bySeries.has(line)) bySeries.set(line, []);
    bySeries.get(line).push(point);
  }
  const setText = (lines) => {
    for (const old of [...tooltip.querySelectorAll("text")]) old.remove();
    // The label text no longer scales with the slot, so the box around it cannot be
    // laid out in bare user units either: every measurement here is in units of the
    // text's own size, inflated by the same factor the stylesheet applies.
    const scale = chartTypeScale(svg);
    lines.forEach((text, index) => {
      const node = document.createElementNS("http://www.w3.org/2000/svg", "text");
      node.setAttribute("x", String(6 * scale));
      node.setAttribute("y", String((13 + index * 12) * scale));
      node.textContent = text;
      tooltip.appendChild(node);
    });
    rect.setAttribute("height", String((8 + lines.length * 12) * scale));
    rect.setAttribute("width", String(Math.max(96, 7 * Math.max(...lines.map((line) => line.length))) * scale));
  };
  const move = (event) => {
    const svgPoint = svg.createSVGPoint();
    svgPoint.x = event.clientX;
    svgPoint.y = event.clientY;
    const local = svgPoint.matrixTransform(svg.getScreenCTM().inverse());
    let nearestX = null;
    let nearestDistance = Infinity;
    const lines = [];
    for (const [, seriesPoints] of bySeries) {
      let best = null;
      let bestDistance = Infinity;
      for (const point of seriesPoints) {
        const distance = Math.abs(Number(point.getAttribute("cx")) - local.x);
        if (distance < bestDistance) {
          bestDistance = distance;
          best = point;
        }
      }
      if (!best) continue;
      lines.push(`${best.dataset.line}: ${best.dataset.yLabel} @ ${best.dataset.xLabel}`);
      if (bestDistance < nearestDistance) {
        nearestDistance = bestDistance;
        nearestX = Number(best.getAttribute("cx"));
      }
    }
    if (nearestX === null) return;
    crosshair.setAttribute("x1", String(nearestX));
    crosshair.setAttribute("x2", String(nearestX));
    crosshair.removeAttribute("hidden");
    setText(lines);
    // Anchored readout: the tooltip sits in a fixed top corner (opposite the crosshair
    // so it never covers the hovered points) instead of floating with the cursor.
    const width = Number(rect.getAttribute("width"));
    const x = nearestX < 180 ? 360 - width - 4 : 44;
    tooltip.setAttribute("transform", `translate(${x.toFixed(1)} 16)`);
    tooltip.removeAttribute("hidden");
  };
  svg.addEventListener("mousemove", move);
  svg.addEventListener("mouseleave", () => {
    crosshair.setAttribute("hidden", "");
    tooltip.setAttribute("hidden", "");
  });
}

function aggregateLeadSeries(grouped) {
  const series = new Map();
  for (const [key, rows] of grouped) {
    const byLead = new Map();
    for (const row of rows) {
      const leadDay = Number(row.lead_day);
      const value = scoreValue(row);
      if (!Number.isFinite(leadDay) || !Number.isFinite(value)) continue;
      if (!byLead.has(leadDay)) byLead.set(leadDay, []);
      byLead.get(leadDay).push({ row, value });
    }
    const aggregated = [];
    for (const [leadDay, values] of byLead) {
      const mean = values.reduce((total, item) => total + item.value, 0) / values.length;
      let ciLow = mean;
      let ciHigh = mean;
      if (values.length === 1) {
        const row = values[0].row;
        ciLow = Number.isFinite(row.ci_low) ? row.ci_low : mean;
        ciHigh = Number.isFinite(row.ci_high) ? row.ci_high : mean;
      } else {
        const variance = values.reduce((total, item) => total + (item.value - mean) ** 2, 0) / (values.length - 1);
        const error = 1.96 * Math.sqrt(variance / values.length);
        ciLow = mean - error;
        ciHigh = mean + error;
      }
      aggregated.push({ lead_day: leadDay, mean, ci_low: ciLow, ci_high: ciHigh });
    }
    if (aggregated.length) series.set(key, aggregated.sort((a, b) => a.lead_day - b.lead_day));
  }
  return series;
}

function scoreValue(row) {
  for (const key of ["mean", "value", "rmse", "rmsd", "score"]) {
    const value = Number(row[key]);
    if (Number.isFinite(value)) return value;
  }
  return NaN;
}

function mapDepthToScoreDepth(entry) {
  if (entry.standard_name.includes("velocity") && entry.depth === "15m") return "15m";
  if (entry.depth === "surface") return "surface";
  return entry.depth;
}

function scoreDepthKeys(entry) {
  const keys = [];
  const class4Bin = class4DepthBin(entry);
  if (class4Bin) keys.push(class4Bin);
  const legacyDepth = mapDepthToScoreDepth(entry);
  if (legacyDepth && !keys.includes(legacyDepth)) keys.push(legacyDepth);
  return keys;
}

// The single sentence for every "nothing to draw" Class-4 state. One source, so the
// sidebar note, the legend strip and the colorbar never state the same absence three
// different ways. Empty string means there are match-ups to show.
function class4EmptyMessage() {
  if (overlayData.class4Unpublished) return "No Class-4 match-ups are published for this dataset.";
  if (overlayData.class4Error) return `Class-4 match-ups failed to load (${overlayData.class4Error}).`;
  if (!overlayData.class4) return "No Class-4 match-ups are available for this dataset and region.";
  if ((overlayData.class4.rows || []).length === 0) return class4SelectionNote();
  return "";
}

function class4SelectionNote() {
  const panel = panels[0];
  if (!panel) return "No Class-4 match-ups for this selection.";
  const manifest = manifestFor(panel.state.dataset);
  const entry = manifest && variableEntry(manifest, panel.state.variable);
  const variable = entry ? prettyName(entry.standard_name) : panel.state.variable;
  return `No ${variable} match-ups at ${class4DepthLabel(entry, class4DepthBin(entry))} for this start/lead.`;
}

function formatCount(value) {
  return Math.round(Number(value) || 0).toLocaleString("en-US");
}

// ---- global controls --------------------------------------------------------

function updateCurrentsControlVisibility() {
  const anyCurrents = panels.slice(0, shared.layout).some((panel) => isCurrentsVariable(panel.state.variable));
  elements["currents-group"].hidden = !anyCurrents;
  const eligible = trajectoryEligiblePanels();
  const ready = eligible.length > 0 && (shared.layout === 1 || eligible.length === 2);
  for (let index = 0; index < shared.layout; index += 1) {
    panels[index].els.wrap.classList.toggle("trajectory-ready", ready);
  }
}

function updateSharedTimeControls(manifest) {
  if (!manifest || !Array.isArray(manifest.start_dates) || !manifest.start_dates.length) return;
  shared.startIndex = Math.min(shared.startIndex, manifest.start_dates.length - 1);
  populateSelect(
    elements["start-date"],
    manifest.start_dates.map((date, index) => ({ value: index, label: date })),
    shared.startIndex,
  );
  if (!Array.isArray(manifest.lead_days) || !manifest.lead_days.length) return;
  const minimumLead = Math.min(...manifest.lead_days);
  const maximumLead = Math.max(...manifest.lead_days);
  shared.leadDay = Math.min(Math.max(shared.leadDay, minimumLead), maximumLead);
  elements["lead-day"].min = String(minimumLead);
  elements["lead-day"].max = String(maximumLead);
  elements["lead-day"].value = String(shared.leadDay);
  elements["lead-value"].textContent = `Lead day ${shared.leadDay}`;
  fillLeadTicks(minimumLead, maximumLead);
}

// One tick per available lead day. The marks are drawn in CSS across the thumb's
// travel, so all the stylesheet needs is how many gaps to divide that travel into.
function fillLeadTicks(minimumLead, maximumLead) {
  const ticks = elements["lead-ticks"];
  if (!ticks) return;
  ticks.style.setProperty("--lead-steps", String(Math.max(1, maximumLead - minimumLead)));
}

async function applyOverlayMode() {
  const region = shared.region;
  // The two forecasts are chosen from the panel pickers; there is no separate truth
  // selector any more, so the legacy eddy-reference control stays hidden.
  elements["eddy-reference-field"].hidden = true;
  const note = elements["overlay-note"];
  // Overlays are per-forecast; the difference view has no forecast to host them, so
  // leave the map clean and nudge the user back to a per-forecast view.
  if (isDiffView()) {
    clearTrajectories();
    note.textContent = shared.overlayMode === "none" ? "" : "Switch to side-by-side to see overlays.";
    for (let i = 0; i < shared.layout; i += 1) {
      drawPanel(panels[i]);
      drawOverlays(panels[i]);
    }
    updateCurrentsControlVisibility();
    updateContextRail();
    return;
  }
  if (shared.overlayMode !== "trajectories") clearTrajectories();
  if (shared.overlayMode !== "column") clearColumnProfile();
  if (shared.overlayMode === "column") {
    // Probe up front so the note promises a click only when there is a store behind it.
    const columnAvailable = await anyVisibleColumnStore();
    if (!columnAvailable) note.textContent = COLUMN_UNPUBLISHED_MESSAGE;
    else
      note.textContent = columnProfile
        ? "Click the map to read the model water column at another point."
        : "Click the map to read the model temperature/salinity water column at that point.";
    for (let i = 0; i < shared.layout; i += 1) {
      drawPanel(panels[i]);
      drawOverlays(panels[i]);
    }
    renderColumnProfileRail();
    updateCurrentsControlVisibility();
    updateContextRail();
    return;
  }
  if (shared.overlayMode === "trajectories") {
    note.textContent = "Click the map to seed trajectories advected through both forecasts' currents.";
  } else if (shared.overlayMode === "class4") {
    class4ProgressLabel = "Loading Class-4 match-ups…";
    note.textContent = class4ProgressLabel;
  } else if (shared.overlayMode === "eddies") {
    note.textContent = "Loading eddy census…";
  } else {
    note.textContent = "";
  }
  await loadOverlayData();
  if (shared.overlayMode === "eddies") {
    const censuses = overlayData.eddiesCensuses || [];
    if (!censuses.some(Boolean)) {
      note.textContent = "No eddy detections are available for this selection.";
    } else if (shared.layout === 2 && overlayData.eddiesMatch) {
      note.textContent =
        "Eddy intercomparison between the two selected forecasts. This shows agreement, not ground truth.";
    } else {
      note.textContent = "Showing this forecast's own eddy census.";
    }
  }
  if (shared.overlayMode === "class4") {
    note.textContent = class4EmptyMessage() || "Class-4 match-ups for the selected start and lead. Hover a point for details.";
  }
  for (let i = 0; i < shared.layout; i += 1) {
    drawPanel(panels[i]);
    drawOverlays(panels[i]);
  }
  updateSharedColorbar();
  updateCurrentsControlVisibility();
  updateContextRail();
}

// When the parquet is targeted (one pair per row group), a start/lead change means
// a different set of obs must be fetched — reload just the overlay and redraw. Small
// (~1-2MB) row groups keep this sub-second; a short debounce avoids a fetch storm
// while the lead slider is dragged. Legacy files hold all pairs, so this is a no-op.
let class4ReloadTimer = null;
function scheduleClass4Reload() {
  if (shared.overlayMode !== "class4") return;
  if (!overlayData.class4 || !overlayData.class4.targeted) return;
  const note = elements["overlay-note"];
  class4ProgressLabel = `Loading obs for lead ${shared.leadDay}…`;
  if (note) note.textContent = class4ProgressLabel;
  if (class4ReloadTimer) clearTimeout(class4ReloadTimer);
  class4ReloadTimer = setTimeout(async () => {
    class4ReloadTimer = null;
    await loadOverlayData();
    if (note) {
      note.textContent =
        class4EmptyMessage() ||
        (overlayData.class4.targeted
          ? "Class-4 match-ups for the selected start and lead. Hover a point for details."
          : "Class-4 match-ups loaded. Hover a point for details.");
    }
    redrawOverlaysAll();
    updateContextRail();
  }, 120);
}

function applyTheme() {
  document.documentElement.dataset.theme = shared.theme;
  elements["theme-toggle"].textContent = shared.theme === "light" ? "Dark theme" : "Light theme";
}

// Single entry point for a theme change, shared by the in-app toggle and, when the
// viewer is embedded in the Quarto site, the parent page's theme switch (postMessage).
function setViewerTheme(theme) {
  if (theme !== "light" && theme !== "dark") return;
  if (theme === shared.theme) return;
  shared.theme = theme;
  applyTheme();
  renderAllPanels().then(() => redrawOverlaysAll());
  writeHash();
}

// The viewer runs standalone (its own Dark theme button) and embedded inside the
// Quarto site (?embed=1): there the button is hidden and the host page drives the
// theme so the site's single toggle controls everything.
const VIEWER_EMBEDDED = new URLSearchParams(location.search).has("embed");

function wireEmbeddedTheme() {
  if (!VIEWER_EMBEDDED) return;
  document.body.classList.add("viewer-embedded");
  elements["theme-toggle"].hidden = true;
  window.addEventListener("message", (event) => {
    const data = event.data;
    if (data && data.type === "oceanbench-theme") setViewerTheme(data.theme);
  });
  if (window.parent !== window) window.parent.postMessage({ type: "oceanbench-viewer-ready" }, "*");
}

function markScopeButtons() {
  for (const button of document.querySelectorAll(".scope-switch [data-scope]")) {
    const selected = button.dataset.scope === shared.scope;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
}

function applyScope() {
  document.documentElement.dataset.scope = shared.scope;
  markScopeButtons();
}

function setScope(scope) {
  if (shared.scope === scope) return;
  shared.scope = scope;
  // The year raster is a per-forecast view; the difference comparison has no meaning
  // there, so entering year scope while in difference falls back to side-by-side.
  if (scope === "year" && isDiffView()) {
    shared.displayMode = "side";
    markDisplayButtons();
    syncPanelGrid();
  }
  clearTrajectories();
  applyScope();
  renderAllPanels().then(() => {
    redrawOverlaysAll();
    updateSharedColorbar();
    updateContextRail();
    updateCurrentsControlVisibility();
  });
  writeHash();
}

function markMetricButtons() {
  for (const button of document.querySelectorAll(".metric-switch [data-metric]")) {
    const selected = button.dataset.metric === shared.yearMetric;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
}

// One-time "?" method notes on the fixed rail-chart headings and the currents control,
// whose anchor elements persist for the app lifetime (unlike the per-render legends).
function wireStaticMethodNotes() {
  const skillHeading = elements["rail-lead-curve"] && elements["rail-lead-curve"].closest(".rail-section")?.querySelector("h3");
  if (skillHeading) attachMethodNote(skillHeading, "lead-curve");
  const depthHeading = elements["rail-depth-profile-section"] && elements["rail-depth-profile-section"].querySelector("h3");
  if (depthHeading) attachMethodNote(depthHeading, "depth-profile");
  const yearHeading = elements["rail-year-rmsd-section"] && elements["rail-year-rmsd-section"].querySelector("h3");
  if (yearHeading) attachMethodNote(yearHeading, "year-rmsd");
  const psdHeading = document.getElementById("rail-spectra-section")?.querySelector("h3");
  if (psdHeading) attachMethodNote(psdHeading, "psd");
  const trajectoryHeading = elements["rail-trajectory-section"] && elements["rail-trajectory-section"].querySelector("h3");
  if (trajectoryHeading) attachMethodNote(trajectoryHeading, "trajectories");
  const currentsLegend = elements["currents-group"] && elements["currents-group"].querySelector("legend");
  if (currentsLegend) attachMethodNote(currentsLegend, "currents");
}

// Switch the year-scope map + rail between |error| and signed bias. Re-renders the
// panels (which pick the field/colormap/range per metric), the rail, and the colorbar.
function setYearMetric(metric) {
  if (metric !== "error" && metric !== "bias") return;
  if (shared.yearMetric === metric) return;
  shared.yearMetric = metric;
  markMetricButtons();
  renderAllPanels().then(() => {
    redrawOverlaysAll();
    updateSharedColorbar();
    updateContextRail();
  });
  writeHash();
}

// Scrubbing the lead slider is how the sequence is watched, so paint must track the
// handle: a lead every shown panel has already drawn is redrawn on the next animation
// frame, which plays the leads the slider crosses one frame at a time. Only a lead
// that still needs tiles off the network waits, so a fast drag over cold leads issues
// one set of fetches instead of one per step.
const LEAD_FETCH_DEBOUNCE_MILLISECONDS = 150;
let leadFetchTimer = null;
let leadRenderFrame = 0;

function scheduleLeadRender() {
  if (leadRenderFrame) return;
  leadRenderFrame = requestAnimationFrame(() => {
    leadRenderFrame = 0;
    if (leadFramesCached(shared.leadDay)) {
      cancelLeadFetchTimer();
      runLeadRender();
      return;
    }
    cancelLeadFetchTimer();
    leadFetchTimer = setTimeout(() => {
      leadFetchTimer = null;
      runLeadRender();
    }, LEAD_FETCH_DEBOUNCE_MILLISECONDS);
  });
}

function cancelLeadFetchTimer() {
  if (!leadFetchTimer) return;
  clearTimeout(leadFetchTimer);
  leadFetchTimer = null;
}

function runLeadRender() {
  renderAllPanels().then(() => {
    redrawOverlaysAll();
    updateContextRail();
  });
  // Re-render the water-column profile from the in-memory column. No refetch: the
  // clicked chunk already holds every lead day (this is the design win).
  renderColumnProfileRail();
}

function wireGlobalControls() {
  for (const button of document.querySelectorAll(".scope-switch [data-scope]")) {
    button.addEventListener("click", () => setScope(button.dataset.scope));
  }
  for (const button of document.querySelectorAll(".metric-switch [data-metric]")) {
    button.addEventListener("click", () => setYearMetric(button.dataset.metric));
  }
  for (const button of document.querySelectorAll(".layout-switch [data-layout]")) {
    button.addEventListener("click", () => {
      const previousLayout = shared.layout;
      shared.layout = Number(button.dataset.layout);
      // Opening Forecast 2 starts it on the variable Forecast 1 is showing (the key
      // carries the depth variant, so 15m currents carry over too), so the comparison
      // begins like with like. A default only: the picker changes it right afterwards,
      // and renderPanel falls back if the second dataset lacks that variable.
      if (shared.layout === 2 && previousLayout === 1) {
        if (!panels[1]) panels[1] = buildPanel(1);
        panels[1].state.variable = panels[0].state.variable;
      }
      markLayoutButtons();
      syncPanelGrid();
      renderAllPanels().then(() => {
        updateSharedColorbar();
        updateContextRail();
        updateCurrentsControlVisibility();
        // A second forecast may add/remove a column store; re-probe and re-read the point.
        updateColumnModeAvailability().then(() => {
          if (columnModeActive() && shared.columnPoint) readColumnProfileAt(shared.columnPoint.lon, shared.columnPoint.lat);
        });
      });
      writeHash();
    });
  }
  elements["start-date"].addEventListener("change", async (event) => {
    clearTrajectories();
    shared.startIndex = Number(event.target.value);
    await renderAllPanels();
    await loadOverlayData();
    redrawOverlaysAll();
    await updateContextRail();
    // The clicked column is start-specific; re-read it at the same point for the new start.
    if (columnModeActive() && shared.columnPoint) await readColumnProfileAt(shared.columnPoint.lon, shared.columnPoint.lat);
    writeHash();
  });
  elements["lead-day"].addEventListener("input", (event) => {
    shared.leadDay = Number(event.target.value);
    elements["lead-value"].textContent = `Lead day ${shared.leadDay}`;
    scheduleLeadRender();
    scheduleClass4Reload();
    scheduleHashWrite();
  });
  elements["overlay-mode"].addEventListener("change", (event) => {
    shared.overlayMode = event.target.value;
    applyOverlayMode();
    writeHash();
  });
  elements["overlay-region"].addEventListener("change", (event) => {
    clearTrajectories();
    clearColumnProfile();
    shared.region = event.target.value;
    fitRegionView();
    renderAllPanels().then(() => {
      redrawOverlaysAll();
      updateContextRail();
    });
    applyOverlayMode();
    writeHash();
  });
  elements["eddy-reference"].addEventListener("change", (event) => {
    shared.eddyReference = event.target.value;
    redrawOverlaysAll();
    updateContextRail();
    writeHash();
  });
  elements["particles-play"].addEventListener("change", (event) => {
    shared.showParticles = event.target.checked;
    applyParticleVisibility();
    writeHash();
  });
  elements["particle-speed"].addEventListener("input", (event) => {
    shared.particleSpeed = Number(event.target.value);
    elements["speed-value"].textContent = `${shared.particleSpeed.toFixed(1)}×`;
    for (const panel of panels) if (panel.particleContext) panel.particleContext.speed = shared.particleSpeed;
    scheduleHashWrite();
  });
  elements["theme-toggle"].addEventListener("click", () => {
    setViewerTheme(shared.theme === "light" ? "dark" : "light");
  });
  elements["psd-toggle"].addEventListener("click", () => {
    setPsdEnabled(!shared.psdEnabled);
  });
  elements["about-toggle"].addEventListener("click", () => {
    const dialog = elements["about-dialog"];
    if (typeof dialog.showModal === "function") dialog.showModal();
    else dialog.setAttribute("open", "");
  });
  elements["about-close"].addEventListener("click", () => elements["about-dialog"].close());
  elements["about-dialog"].addEventListener("click", (event) => {
    if (event.target === elements["about-dialog"]) elements["about-dialog"].close();
  });
  elements["rail-collapse"].addEventListener("click", () => {
    shared.railCollapsed = !shared.railCollapsed;
    if (!drawersOverlaying()) localStorage.setItem("oceanbench.viewer.railCollapsed", shared.railCollapsed ? "1" : "0");
    applyRailCollapsed();
    redrawAllPanels();
    if (!shared.railCollapsed) updateContextRail();
    writeHash();
  });
  elements["left-collapse"].addEventListener("click", () => {
    shared.controlsCollapsed = !shared.controlsCollapsed;
    if (!drawersOverlaying()) localStorage.setItem("oceanbench.viewer.controlsCollapsed", shared.controlsCollapsed ? "1" : "0");
    applyRailCollapsed();
    redrawAllPanels();
    writeHash();
  });
  wireLayoutSplitters();
  for (const button of elements["rail-forecast-toggle"].querySelectorAll("button")) {
    button.addEventListener("click", () => {
      shared.railForecast = Number(button.dataset.forecast);
      updateContextRail();
      writeHash();
    });
  }
  for (const button of document.querySelectorAll(".display-switch [data-display]")) {
    button.addEventListener("click", () => {
      shared.displayMode = button.dataset.display;
      markDisplayButtons();
      syncPanelGrid();
      renderAllPanels().then(() => {
        redrawOverlaysAll();
        updateSharedColorbar();
        updateContextRail();
        if (shared.overlayMode !== "none") applyOverlayMode();
      });
      writeHash();
    });
  }
  elements["reset-view"].addEventListener("click", resetZoomPan);
  elements["trajectory-clear"].addEventListener("click", clearTrajectories);
  elements["column-clear"].addEventListener("click", () => {
    clearColumnProfile();
    const note = elements["overlay-note"];
    if (columnModeActive() && note) note.textContent = "Click the map to read the model temperature/salinity water column at that point.";
    writeHash();
  });
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      clearTrajectories();
      clearColumnProfile();
      // The overlay picker must not keep reading "trajectories"/"column" once the thing
      // it names is gone, so drop it back to none along with its note.
      if (shared.overlayMode === "trajectories" || shared.overlayMode === "column") {
        shared.overlayMode = "none";
        elements["overlay-mode"].value = "none";
        applyOverlayMode();
        writeHash();
      }
    }
  });

  window.addEventListener("pointermove", (event) => {
    if (!panels.some((panel) => panel.dragging || panel.draggingSwipe)) updateHover(event);
  });
  window.addEventListener("resize", () => {
    syncDrawerOverlayMode();
    applyLayout();
    clampView();
    scheduleLayoutRender();
  });

  // The colour scale is drawn to the pixels it is displayed at, so every change of
  // that box (window, drawer drag, the strip's own container queries) has to redraw
  // it. Nothing inside the redraw changes the box, so this cannot loop.
  if (elements.colorbar && window.ResizeObserver) {
    new ResizeObserver(() => updateSharedColorbar()).observe(elements.colorbar);
  }
}

let layoutRenderTimer = null;

function layoutLimits() {
  const workspace = document.querySelector(".workspace").getBoundingClientRect();
  return {
    controlsWidth: [208, Math.min(380, workspace.width * 0.32)],
    railWidth: [280, Math.min(620, workspace.width * 0.42)],
  };
}

function clampLayoutValue(name, value) {
  const [minimum, maximum] = layoutLimits()[name];
  // A non-finite width would collapse the workspace grid (NaNpx track); fall back
  // to the default column size rather than clamping NaN into the layout.
  const safe = Number.isFinite(value) ? value : DEFAULT_LAYOUT[name];
  return Math.round(Math.min(maximum, Math.max(minimum, safe)));
}

// Width of a drawer collapsed down to its edge tab. Mirrors --drawer-tab-width in
// styles.css; the grid track is written from JS so the two must agree.
const DRAWER_TAB_WIDTH = 26;

// Below this width the drawers stop taking a grid track and float over the map, so
// the map keeps a usable width instead of being squeezed to nothing.
const DRAWER_OVERLAY_QUERY = "(max-width: 1100px)";

function applyLayout() {
  const workspace = document.querySelector(".workspace");
  if (!workspace) return;
  const overlaying = window.matchMedia(DRAWER_OVERLAY_QUERY).matches;
  shared.controlsWidth = clampLayoutValue("controlsWidth", shared.controlsWidth);
  shared.railWidth = clampLayoutValue("railWidth", shared.railWidth);
  if (!overlaying && shared.controlsWidth + shared.railWidth > workspace.clientWidth - 420) {
    shared.railWidth = Math.max(280, workspace.clientWidth - 420 - shared.controlsWidth);
  }
  workspace.style.setProperty("--controls-width", `${shared.controlsCollapsed ? DRAWER_TAB_WIDTH : shared.controlsWidth}px`);
  workspace.style.setProperty("--rail-width", `${shared.railCollapsed ? DRAWER_TAB_WIDTH : shared.railWidth}px`);
  workspace.style.setProperty("--controls-open-width", `${shared.controlsWidth}px`);
  workspace.style.setProperty("--rail-open-width", `${shared.railWidth}px`);
}

// Design width of every rail chart's viewBox. The SVG is stretched to the slot, so
// this over the slot's measured width is the factor the browser applies to user
// units, and the factor label sizes have to divide back out to stay put.
const CHART_VIEW_WIDTH = 360;

// How much a length declared in user units has to be inflated to render at its
// declared size in the slot as it is measured right now.
function chartTypeScale(node) {
  const slot = node && node.closest ? node.closest(".rail-chart-slot") : null;
  const width = slot ? slot.getBoundingClientRect().width : 0;
  return width > 0 ? CHART_VIEW_WIDTH / width : 1;
}

// Publish each slot's measured width so the stylesheet can pin label sizes. Charts
// are re-rendered into these slots constantly; observing the slots rather than the
// SVGs means the value survives every re-render.
function watchChartSlots() {
  const slots = [...document.querySelectorAll(".rail-chart-slot")];
  if (!slots.length || typeof ResizeObserver === "undefined") return;
  const publish = (slot) => {
    const width = slot.getBoundingClientRect().width;
    slot.style.setProperty("--chart-slot-width", String(Math.max(1, width)));
  };
  const observer = new ResizeObserver((entries) => {
    for (const entry of entries) publish(entry.target);
  });
  for (const slot of slots) {
    publish(slot);
    observer.observe(slot);
  }
}

function persistLayout() {
  localStorage.setItem("oceanbench.viewer.layout", JSON.stringify({
    controlsWidth: shared.controlsWidth,
    railWidth: shared.railWidth,
  }));
}

function scheduleLayoutRender() {
  clearTimeout(layoutRenderTimer);
  layoutRenderTimer = setTimeout(() => {
    clampView();
    redrawAllPanels();
    updateContextRail();
  }, 80);
}

function wireSplitter(element, name, axis, direction) {
  element.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    event.preventDefault();
    element.setPointerCapture(event.pointerId);
    element.classList.add("active");
    document.body.classList.add("resizing-layout");
    document.body.classList.toggle("row", axis === "y");
    const start = axis === "x" ? event.clientX : event.clientY;
    const startValue = shared[name];
    const move = (moveEvent) => {
      const coordinate = axis === "x" ? moveEvent.clientX : moveEvent.clientY;
      shared[name] = clampLayoutValue(name, startValue + direction * (coordinate - start));
      applyLayout();
      scheduleLayoutRender();
    };
    const end = () => {
      element.removeEventListener("pointermove", move);
      element.classList.remove("active");
      document.body.classList.remove("resizing-layout", "row");
      persistLayout();
      writeHash();
    };
    element.addEventListener("pointermove", move);
    element.addEventListener("pointerup", end, { once: true });
    element.addEventListener("pointercancel", end, { once: true });
  });
  element.addEventListener("dblclick", () => {
    shared[name] = DEFAULT_LAYOUT[name];
    applyLayout();
    persistLayout();
    scheduleLayoutRender();
    writeHash();
  });
}

function wireLayoutSplitters() {
  wireSplitter(elements["controls-map-splitter"], "controlsWidth", "x", 1);
  wireSplitter(elements["map-rail-splitter"], "railWidth", "x", -1);
}

// Collapse a drawer to its edge tab, restoring the previous width on expand. The
// collapsed state persists like the width.
function applyDrawerCollapsed(drawer, toggle, collapsed, label) {
  if (!drawer || !toggle) return;
  drawer.classList.toggle("collapsed", collapsed);
  toggle.setAttribute("aria-expanded", String(!collapsed));
  toggle.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} ${label}`);
}

function applyRailCollapsed() {
  applyDrawerCollapsed(elements["context-rail"], elements["rail-collapse"], shared.railCollapsed, "context rail");
  applyDrawerCollapsed(elements["left-drawer"], elements["left-collapse"], shared.controlsCollapsed, "controls");
  applyLayout();
}

// Below the overlay breakpoint an open drawer floats over the map instead of taking a
// track beside it, so leaving both open would bury the field. Fold them on the way in
// and hand back the wide-viewport state on the way out; the user can still open either
// one by hand while narrow.
let preOverlayCollapse = null;

function drawersOverlaying() {
  return window.matchMedia(DRAWER_OVERLAY_QUERY).matches;
}

function syncDrawerOverlayMode() {
  const overlaying = drawersOverlaying();
  if (overlaying === (preOverlayCollapse !== null)) return;
  if (overlaying) {
    preOverlayCollapse = { controls: shared.controlsCollapsed, rail: shared.railCollapsed };
    shared.controlsCollapsed = true;
    shared.railCollapsed = true;
  } else {
    shared.controlsCollapsed = preOverlayCollapse.controls;
    shared.railCollapsed = preOverlayCollapse.rail;
    preOverlayCollapse = null;
  }
  applyRailCollapsed();
}

function redrawOverlaysAll() {
  for (let i = 0; i < shared.layout; i += 1) {
    drawOverlays(panels[i]);
    drawTrajectoryFans(panels[i]);
  }
  updateSharedColorbar();
}

function markLayoutButtons() {
  for (const button of document.querySelectorAll(".layout-switch [data-layout]")) {
    const selected = Number(button.dataset.layout) === shared.layout;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
  // The side-by-side / swipe display switch is meaningful only with two forecasts.
  const displaySwitch = document.querySelector(".display-switch");
  if (displaySwitch) displaySwitch.hidden = shared.layout !== 2;
  markDisplayButtons();
}

function markDisplayButtons() {
  for (const button of document.querySelectorAll(".display-switch [data-display]")) {
    const selected = button.dataset.display === shared.displayMode;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
}

// PSD of the boxed region is fetch+CPU work; recompute the rail on a short debounce when the
// user pans / zooms / scrubs lead so the spectra track the viewport without jank.
let railUpdateTimer = null;
function scheduleRailUpdate() {
  clearTimeout(railUpdateTimer);
  railUpdateTimer = setTimeout(() => updateContextRail(), 220);
}

// ---- URL hash (every view state is a URL — §6) ------------------------------

let hashWriteTimer = null;
function scheduleHashWrite() {
  clearTimeout(hashWriteTimer);
  hashWriteTimer = setTimeout(writeHash, 250);
}

// The third token is the legacy per-panel display mode. It no longer exists — the
// comparison view is a shared dm choice now — but we keep writing a stable "field"
// so old parsers and roundtrips stay happy; readers tolerate any value (see below).
function encodePanel(panel) {
  return [panel.state.dataset, panel.state.variable, "field"].join(",");
}

function writeHash() {
  const parameters = new URLSearchParams();
  parameters.set("layout", String(shared.layout));
  parameters.set("s", String(shared.startIndex));
  parameters.set("l", String(shared.leadDay));
  parameters.set("z", view.zoom.toFixed(3));
  parameters.set("cx", view.centerNX.toFixed(4));
  parameters.set("cy", view.centerNY.toFixed(4));
  parameters.set("theme", shared.theme);
  if (shared.scope !== "single") parameters.set("scope", shared.scope);
  if (shared.yearMetric !== "error") parameters.set("metric", shared.yearMetric);
  if (shared.psdEnabled) {
    parameters.set("psdOn", "1");
  }
  if (shared.psdEnabled && shared.psdBox) {
    // The requested size travels in the link, so a shared URL reopens at the size the
    // user chose rather than at whatever a transient constraint clamped it to.
    const size = shared.psdBoxRequest || shared.psdBox;
    parameters.set("psd", [shared.psdBox.lon, shared.psdBox.lat, size.w, size.h].map((v) => v.toFixed(2)).join(","));
  }
  if (shared.overlayMode !== "none") parameters.set("ov", shared.overlayMode);
  if (shared.overlayMode === "column" && shared.columnPoint) {
    parameters.set("col", `${shared.columnPoint.lon.toFixed(3)},${shared.columnPoint.lat.toFixed(3)}`);
  }
  parameters.set("region", shared.region);
  if (shared.overlayMode === "eddies") parameters.set("eref", shared.eddyReference);
  if (shared.railCollapsed) parameters.set("rail", "collapsed");
  if (shared.controlsCollapsed) parameters.set("ctrl", "collapsed");
  parameters.set("rw", String(shared.railWidth));
  parameters.set("cw", String(shared.controlsWidth));
  if (shared.layout === 2) parameters.set("dm", shared.displayMode);
  parameters.set("play", shared.showParticles ? "1" : "0");
  parameters.set("spd", shared.particleSpeed.toFixed(1));
  for (let i = 0; i < shared.layout; i += 1) parameters.set(`p${i}`, encodePanel(panels[i]));
  const encoded = `#${parameters.toString()}`;
  if (encoded !== location.hash) history.replaceState(null, "", encoded);
}

function readHash() {
  const parameters = new URLSearchParams(location.hash.slice(1));
  // Only accept FINITE numbers from the hash; a malformed value (e.g. #z=abc)
  // must fall back to the current default rather than poisoning view/layout math
  // with NaN — which otherwise blanks the map ("zoom NaN×"), issues a 0.NaN.0.0
  // chunk fetch (404 + uncaught rejection), or sets grid tracks to NaNpx.
  const number = (key, fallback) => {
    if (!parameters.has(key)) return fallback;
    const value = Number(parameters.get(key));
    return Number.isFinite(value) ? value : fallback;
  };
  shared.layout = number("layout", shared.layout);
  shared.startIndex = number("s", shared.startIndex);
  shared.leadDay = number("l", shared.leadDay);
  view.zoom = number("z", view.zoom);
  view.centerNX = number("cx", view.centerNX);
  view.centerNY = number("cy", view.centerNY);
  if (parameters.has("theme")) shared.theme = parameters.get("theme");
  if (parameters.get("scope") === "year") shared.scope = "year";
  if (parameters.get("metric") === "bias") shared.yearMetric = "bias";
  shared.psdEnabled = parameters.get("psdOn") === "1";
  if (parameters.has("psd")) {
    const [lon, lat, w, h] = parameters.get("psd").split(",").map(Number);
    if ([lon, lat, w, h].every(Number.isFinite) && w > 0 && h > 0) {
      shared.psdBox = { lon, lat, w, h };
      shared.psdBoxRequest = { w, h };
      if (!parameters.has("psdOn")) shared.psdEnabled = true;
    }
  }
  if (parameters.has("ov")) shared.overlayMode = parameters.get("ov");
  if (parameters.has("col")) {
    const [lon, lat] = parameters.get("col").split(",").map(Number);
    if (Number.isFinite(lon) && Number.isFinite(lat)) shared.columnPoint = { lon, lat };
  }
  if (parameters.has("region")) shared.region = parameters.get("region");
  if (parameters.has("eref")) shared.eddyReference = parameters.get("eref");
  if (parameters.get("rail") === "0" || parameters.get("rail") === "collapsed") shared.railCollapsed = true;
  if (parameters.get("ctrl") === "0" || parameters.get("ctrl") === "collapsed") shared.controlsCollapsed = true;
  const railWidth = number("rw", null);
  if (railWidth !== null) shared.railWidth = Math.min(620, Math.max(280, railWidth));
  shared.controlsWidth = number("cw", shared.controlsWidth);
  if (["side", "swipe", "diff"].includes(parameters.get("dm"))) shared.displayMode = parameters.get("dm");
  if (parameters.has("play")) shared.showParticles = parameters.get("play") === "1";
  shared.particleSpeed = number("spd", shared.particleSpeed);
  return parameters;
}

function applyPanelHash(parameters) {
  for (let i = 0; i < shared.layout; i += 1) {
    const encoded = parameters.get(`p${i}`);
    if (!encoded) continue;
    const [dataset, variable, mode] = encoded.split(",");
    if (!panels[i]) panels[i] = buildPanel(i);
    // The third token is the legacy per-panel mode; it is ignored now (any value reads
    // as "field") except the old "currents" mode, which migrates to the currents
    // variable. A trailing fourth token (old difference dataset) is simply dropped.
    const migratedCurrents = mode === "currents";
    Object.assign(panels[i].state, {
      dataset: dataset || panels[i].state.dataset,
      variable: migratedCurrents ? CURRENTS_VARIABLE_SURFACE : variable || panels[i].state.variable,
    });
  }
}

// ---- status -----------------------------------------------------------------

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("error", isError);
  elements.status.hidden = !message;
}

function selectElements() {
  for (const id of [
    "start-date",
    "lead-day",
    "lead-ticks",
    "lead-value",
    "overlay-mode",
    "overlay-region",
    "eddy-reference",
    "eddy-reference-field",
    "overlay-note",
    "currents-group",
    "particles-play",
    "particle-speed",
    "speed-value",
    "reset-view",
    "theme-toggle",
    "about-toggle",
    "about-dialog",
    "about-close",
    "rail-collapse",
    "left-drawer",
    "left-collapse",
    "panel-grid",
    "colorbar",
    "map-legend",
    "year-legend",
    "layer-info",
    "status",
    "context-rail",
    "controls-map-splitter",
    "map-rail-splitter",
    "rail-subtitle",
    "rail-current-depth-note",
    "rail-forecast-toggle",
    "rail-lead-curve",
    "rail-skill-note",
    "rail-depth-profile-section",
    "rail-depth-profile",
    "rail-depth-profile-note",
    "psd-toggle",
    "rail-spectra",
    "rail-psd-note",
    "rail-year-rmsd-section",
    "rail-year-rmsd",
    "rail-year-rmsd-note",
    "rail-trajectory-section",
    "rail-trajectory-chart",
    "rail-trajectory-note",
    "trajectory-clear",
    "rail-column-section",
    "rail-column-chart",
    "rail-column-point",
    "rail-column-note",
    "column-clear",
    "rail-provenance",
  ]) {
    elements[id] = document.getElementById(id);
  }
}

// ---- boot -------------------------------------------------------------------

async function main() {
  selectElements();
  // Optional viewer-config.json (404 tolerated) can repoint the data root before the
  // first fetch; a ?data= query parameter still wins over it.
  await initializeViewerConfig();
  setStatus("Loading catalog…");
  const datasetsUrl = resolveViewerDataUrl(DATASETS_PATH);
  try {
    const response = await fetch(datasetsUrl, { cache: "no-cache" });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    datasetCatalog = (await response.json()).datasets;
  } catch (error) {
    // Name the resource that failed and what it means, rather than surfacing a raw
    // exception string the reader cannot act on.
    setStatus(
      `The viewer could not read its dataset catalog at ${datasetsUrl} (${error.message}), so there is nothing to show yet.`,
      true,
    );
    return;
  }
  if (!datasetCatalog.length) {
    setStatus("No datasets in ./data/datasets.json.", true);
    return;
  }

  const parameters = readHash();
  // Only 1 or 2 panels are supported; old 4-panel hashes degrade to 2.
  if (!Number.isFinite(shared.layout) || shared.layout < 1) shared.layout = 1;
  else if (shared.layout > 2) shared.layout = 2;
  // The difference view has no meaning in the year raster scope; a deep link that
  // asks for both falls back to side-by-side (mirrors the runtime scope switch).
  if (shared.scope === "year" && shared.displayMode === "diff") shared.displayMode = "side";
  applyTheme();
  applyScope();
  markMetricButtons();
  syncDrawerOverlayMode();
  applyRailCollapsed();
  applyLayout();
  watchChartSlots();
  elements["lead-value"].textContent = `Lead day ${shared.leadDay}`;
  elements["speed-value"].textContent = `${shared.particleSpeed.toFixed(1)}×`;
  elements["particles-play"].checked = shared.showParticles;
  elements["particle-speed"].value = String(shared.particleSpeed);
  elements["overlay-mode"].value = shared.overlayMode;
  elements["overlay-region"].value = shared.region;
  elements["eddy-reference"].value = shared.eddyReference;

  for (let i = 0; i < shared.layout; i += 1) if (!panels[i]) panels[i] = buildPanel(i);
  applyPanelHash(parameters);
  clampView();
  writeHash();

  // Insight index loads before the overlays that key off it. The score summary is not
  // awaited here: ensureScoresSummary fetches it when the rail first needs it.
  insightIndex = await loadInsightIndex();

  // Ensure the primary dataset store so start-date / lead options are known.
  // Warm every visible panel's store so variable/start selectors populate on first paint.
  await Promise.all(panels.slice(0, shared.layout).map((panel) => ensureStore(panel.state.dataset).catch(() => {})));
  const manifest = manifestFor(panels[0].state.dataset);
  updateSharedTimeControls(manifest);

  markLayoutButtons();
  syncPanelGrid();
  wireGlobalControls();
  wireStaticMethodNotes();
  wireEmbeddedTheme();

  clampView();
  writeHash();
  await renderAllPanels();
  updateCurrentsControlVisibility();
  updateSharedColorbar();
  await applyOverlayMode();
  await updateContextRail();
  // Probe column-store availability (shows/hides the "Water column (click)" option) and
  // replay a deep-linked clicked point once stores are known.
  await updateColumnModeAvailability();
  if (columnModeActive() && shared.columnPoint) {
    await readColumnProfileAt(shared.columnPoint.lon, shared.columnPoint.lat);
  }
  writeHash();
}

main();
