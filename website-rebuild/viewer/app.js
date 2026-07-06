// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// OceanBench viewer — comparison-first field explorer (contracts.md §6).
//
// Comparison is the primitive: 1/2/4 synchronized panels sharing viewport, lead day
// and start date; each panel is {dataset, variable, mode}. Modes are field, first-
// class difference (A − B in a diverging colormap centred on 0), and animated
// currents (GPU-style advected particles over uo/vo). Any field panel can A/B
// compare a second dataset by swipe divider or a blink key. Insight overlays (eddy
// census, Class-4 obs error, Lagrangian trajectory stub) attach as purpose-modes,
// never all at once. A context rail carries the quantitative curves (skill vs lead,
// PSD spectrum) for the active view, and a small-multiples strip shows error growth
// across leads. Every bit of view state lives in the URL hash.

import { loadStore, loadManifest, readLayer, readCoordinate, prefetchLayer } from "./modules/zarr.js";
import { COLORMAP_NAMES } from "./vendor/cmocean/colormaps.js";
import {
  fieldToImageData,
  fieldStatistics,
  symmetricRange,
  differenceField,
  resampleOntoGrid,
  drawColorbar,
} from "./modules/render.js";
import { startParticleField, makeVelocitySampler, speedMagnitudeField } from "./modules/particles.js";
import {
  loadInsightIndex,
  loadEddies,
  loadSpectra,
  loadScoresSummary,
  loadClass4,
  insightsFor,
  eddyFrame,
  spectraEntry,
  class4Points,
  loadTrajectories,
} from "./modules/insights.js";
import { drawEddyFrame, drawClass4Points, class4ErrorScale, EDDY_COLORS } from "./modules/overlays.js";
import { leadCurveSVG, spectraSVG, SERIES_COLORS } from "./modules/charts.js";

const DATASETS_URL = "./data/datasets.json";
const DIFFERENCE_COLORMAP = "balance";
const SPEED_COLORMAP = "speed";
const CURRENTS_MAX_SPEED = 1.2; // m/s mapping to the top of the speed colormap
const PARTICLE_MAGNITUDE_SCALE = 1.0;

// Shared state — linked across every panel (contracts.md §6).
const view = { zoom: 1, centerNX: 0.5, centerNY: 0.5 };
const shared = {
  startIndex: 0,
  leadDay: 1,
  theme: "dark",
  layout: 1,
  overlayMode: "none",
  region: "global",
  eddyReference: "glorys",
  particlesPlaying: true,
  particleSpeed: 1,
  railOpen: true,
};

const stores = new Map();
const manifests = new Map();
const coordinatesByLevel = new Map();
let datasetCatalog = [];
let insightIndex = null;
let scoresSummary = [];
const panels = [];
let activePanelIndex = 0;
const elements = {};

// ---- store / manifest / coordinate helpers (shared cache) -------------------

async function ensureStore(slug) {
  if (stores.has(slug)) return stores.get(slug);
  const descriptor = datasetCatalog.find((entry) => entry.slug === slug);
  if (!descriptor) throw new Error(`Unknown dataset ${slug}`);
  const [store, manifest] = await Promise.all([loadStore(descriptor.store), loadManifest(descriptor.manifest)]);
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

function otherSlug(slug) {
  return datasetCatalog.find((entry) => entry.slug !== slug)?.slug || slug;
}

function defaultPanelState(index) {
  const first = datasetCatalog[0].slug;
  const second = otherSlug(first);
  const dataset = index === 1 ? second : first;
  return {
    dataset,
    variable: "sea_surface_height_above_geoid",
    mode: "field",
    datasetB: otherSlug(dataset), // always a different dataset so diff/compare is meaningful
    compare: false,
    colormap: null,
  };
}

function buildPanel(index) {
  const container = document.createElement("div");
  container.className = "panel";
  container.dataset.index = String(index);
  container.innerHTML = `
    <div class="panel-head">
      <select class="panel-dataset" aria-label="Panel dataset"></select>
      <select class="panel-variable" aria-label="Panel variable"></select>
      <select class="panel-mode" aria-label="Panel mode">
        <option value="field">Field</option>
        <option value="diff">Difference</option>
        <option value="currents">Currents</option>
      </select>
      <select class="panel-dataset-b" aria-label="Compare dataset" hidden></select>
      <label class="panel-compare" hidden><input type="checkbox" class="panel-compare-toggle" /> swipe</label>
      <span class="spacer"></span>
      <span class="panel-badge"></span>
    </div>
    <div class="panel-canvas-wrap">
      <canvas class="panel-field"></canvas>
      <canvas class="panel-particles"></canvas>
      <canvas class="panel-overlay"></canvas>
      <div class="panel-readout" role="status"></div>
      <div class="panel-swipe-hint" hidden></div>
    </div>`;
  const panel = {
    index,
    container,
    state: defaultPanelState(index),
    els: {
      dataset: container.querySelector(".panel-dataset"),
      variable: container.querySelector(".panel-variable"),
      mode: container.querySelector(".panel-mode"),
      datasetB: container.querySelector(".panel-dataset-b"),
      compareField: container.querySelector(".panel-compare"),
      compareToggle: container.querySelector(".panel-compare-toggle"),
      badge: container.querySelector(".panel-badge"),
      wrap: container.querySelector(".panel-canvas-wrap"),
      field: container.querySelector(".panel-field"),
      particles: container.querySelector(".panel-particles"),
      overlay: container.querySelector(".panel-overlay"),
      readout: container.querySelector(".panel-readout"),
      swipeHint: container.querySelector(".panel-swipe-hint"),
    },
    offscreenA: null,
    offscreenB: null,
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
    blink: false,
    swipeX: 0.5,
    renderToken: 0,
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
  if (panel.state.datasetB === panel.state.dataset) panel.state.datasetB = otherSlug(panel.state.dataset);
  populateSelect(
    panel.els.dataset,
    datasetCatalog.map((entry) => ({ value: entry.slug, label: entry.label })),
    panel.state.dataset,
  );
  const manifest = manifestFor(panel.state.dataset);
  if (manifest) {
    if (!(panel.state.variable in manifest.variables)) panel.state.variable = Object.keys(manifest.variables)[0];
    populateSelect(
      panel.els.variable,
      Object.keys(manifest.variables).map((key) => ({ value: key, label: variableLabel(manifest, key) })),
      panel.state.variable,
    );
  }
  panel.els.mode.value = panel.state.mode;
  populateSelect(
    panel.els.datasetB,
    datasetCatalog.map((entry) => ({ value: entry.slug, label: entry.label })),
    panel.state.datasetB,
  );
  panel.els.compareField.hidden = panel.state.mode !== "field";
  panel.els.compareToggle.checked = panel.state.compare;
  panel.els.datasetB.hidden = !(panel.state.mode === "diff" || (panel.state.mode === "field" && panel.state.compare));
}

function wirePanel(panel) {
  panel.container.addEventListener("mousedown", () => setActivePanel(panel.index), true);
  panel.els.dataset.addEventListener("change", (event) => {
    panel.state.dataset = event.target.value;
    setActivePanel(panel.index);
    ensureStore(panel.state.dataset).then(() => {
      const manifest = manifestFor(panel.state.dataset);
      if (!(panel.state.variable in manifest.variables)) panel.state.variable = Object.keys(manifest.variables)[0];
      refreshPanelControls(panel);
      renderPanel(panel).then(() => updateSmallMultiples());
      writeHash();
    });
  });
  panel.els.variable.addEventListener("change", (event) => {
    panel.state.variable = event.target.value;
    setActivePanel(panel.index);
    renderPanel(panel).then(() => updateSmallMultiples());
    updateContextRail();
    writeHash();
  });
  panel.els.mode.addEventListener("change", (event) => {
    panel.state.mode = event.target.value;
    setActivePanel(panel.index);
    refreshPanelControls(panel);
    renderPanel(panel).then(() => updateSmallMultiples());
    updateCurrentsControlVisibility();
    writeHash();
  });
  panel.els.datasetB.addEventListener("change", (event) => {
    panel.state.datasetB = event.target.value;
    setActivePanel(panel.index);
    ensureStore(panel.state.datasetB)
      .then(() => renderPanel(panel))
      .then(() => updateSmallMultiples())
      .then(writeHash);
  });
  panel.els.compareToggle.addEventListener("change", (event) => {
    panel.state.compare = event.target.checked;
    setActivePanel(panel.index);
    refreshPanelControls(panel);
    const done = panel.state.compare ? ensureStore(panel.state.datasetB).then(() => renderPanel(panel)) : renderPanel(panel);
    done.then(() => updateSmallMultiples());
    writeHash();
  });

  const field = panel.els.field;
  field.addEventListener("mousedown", (event) => beginPanelDrag(panel, event));
  field.addEventListener("wheel", (event) => onPanelWheel(panel, event), { passive: false });
  field.addEventListener("mouseleave", () => {
    panel.els.readout.textContent = "";
  });
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
  return selectRenderLevel(manifestFor(slug));
}

async function renderPanel(panel) {
  const token = ++panel.renderToken;
  try {
    await ensureStore(panel.state.dataset);
    const manifest = manifestFor(panel.state.dataset);
    if (!(panel.state.variable in manifest.variables)) panel.state.variable = Object.keys(manifest.variables)[0];
    const level = selectRenderLevel(manifest);
    const start = Math.min(shared.startIndex, manifest.start_dates.length - 1);
    const leadIndex = shared.leadDay - 1;

    if (panel.state.mode === "currents") {
      await renderCurrentsPanel(panel, token, manifest, level, start, leadIndex);
    } else if (panel.state.mode === "diff") {
      await renderDifferencePanel(panel, token, manifest, level, start, leadIndex);
    } else {
      await renderFieldPanel(panel, token, manifest, level, start, leadIndex);
    }
    if (token !== panel.renderToken) return;
    resizePanelCanvases(panel);
    drawPanel(panel);
    drawOverlays(panel);
    updateSharedColorbar();
    setStatus("");
  } catch (error) {
    if (token === panel.renderToken) setStatus(String(error.message || error), true);
    console.error(error);
  }
}

async function readAlignedField(panel, sourceSlug, variable, level, start, leadIndex, targetLat, targetLon) {
  await ensureStore(sourceSlug);
  const layer = await readLayer(stores.get(sourceSlug), { variable, level, startIndex: start, leadIndex });
  const coordinates = await loadCoordinates(sourceSlug, level);
  if (!targetLat) return { field: layer, latitudes: coordinates.latitudes, longitudes: coordinates.longitudes };
  const aligned = resampleOntoGrid(layer, coordinates.latitudes, coordinates.longitudes, targetLat, targetLon);
  return { field: aligned, latitudes: targetLat, longitudes: targetLon, compressedBytes: layer.compressedBytes };
}

async function renderFieldPanel(panel, token, manifest, level, start, leadIndex) {
  const entry = manifest.variables[panel.state.variable];
  const primary = await readAlignedField(panel, panel.state.dataset, panel.state.variable, level, start, leadIndex);
  if (token !== panel.renderToken) return;
  const colormap = panel.state.colormap || entry.default_colormap;
  const range = entry.default_range;
  panel.field = primary.field;
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

  if (panel.state.compare && panel.state.datasetB) {
    await ensureStore(panel.state.datasetB);
    const compareLevel = renderLevelForSlug(panel.state.datasetB);
    const compare = await readAlignedField(
      panel,
      panel.state.datasetB,
      panel.state.variable,
      compareLevel,
      start,
      leadIndex,
      primary.latitudes,
      primary.longitudes,
    );
    if (token !== panel.renderToken) return;
    panel.offscreenB = colorize(compare.field, compare.latitudes, colormap, range);
    panel.edgesB = panel.edgesA;
  }
  prefetchNeighbours(panel, level, start, leadIndex);
}

async function renderDifferencePanel(panel, token, manifest, level, start, leadIndex) {
  const entry = manifest.variables[panel.state.variable];
  const primary = await readAlignedField(panel, panel.state.dataset, panel.state.variable, level, start, leadIndex);
  if (token !== panel.renderToken) return;
  await ensureStore(panel.state.datasetB);
  const compareLevel = renderLevelForSlug(panel.state.datasetB);
  const compare = await readAlignedField(
    panel,
    panel.state.datasetB,
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
  panel.latitudes = primary.latitudes;
  panel.longitudes = primary.longitudes;
  panel.edgesA = worldEdges(primary.latitudes, primary.longitudes);
  panel.offscreenA = colorize(difference, primary.latitudes, DIFFERENCE_COLORMAP, range);
  panel.offscreenB = null;
  panel.range = range;
  panel.colormap = DIFFERENCE_COLORMAP;
  panel.units = entry.units;
  panel.label = `${labelFor(panel.state.dataset)} − ${labelFor(panel.state.datasetB)} · ${prettyName(entry.standard_name)}`;
  stopParticles(panel);
  prefetchNeighbours(panel, level, start, leadIndex);
}

async function renderCurrentsPanel(panel, token, manifest, level, start, leadIndex) {
  const variables = currentDepthVariables(panel);
  if (!(variables.u in manifest.variables)) {
    setStatus(`${labelFor(panel.state.dataset)} has no velocity fields for currents mode`, true);
    return;
  }
  const uPrimary = await readAlignedField(panel, panel.state.dataset, variables.u, level, start, leadIndex);
  if (token !== panel.renderToken) return;
  const vPrimary = await readAlignedField(panel, panel.state.dataset, variables.v, level, start, leadIndex);
  if (token !== panel.renderToken) return;
  const speed = speedMagnitudeField(uPrimary.field, vPrimary.field);
  const range = [0, CURRENTS_MAX_SPEED];
  panel.field = speed;
  panel.latitudes = uPrimary.latitudes;
  panel.longitudes = uPrimary.longitudes;
  panel.edgesA = worldEdges(uPrimary.latitudes, uPrimary.longitudes);
  panel.offscreenA = colorize(speed, uPrimary.latitudes, SPEED_COLORMAP, range);
  panel.offscreenB = null;
  panel.range = range;
  panel.colormap = SPEED_COLORMAP;
  panel.units = "m/s";
  panel.label = `${labelFor(panel.state.dataset)} · currents (${variables.depth})`;
  panel.velocity = {
    sampler: makeVelocitySampler(uPrimary.field, vPrimary.field, uPrimary.latitudes, uPrimary.longitudes),
  };
  startPanelParticles(panel);
  prefetchNeighbours(panel, level, start, leadIndex);
}

function colorize(field, latitudes, colormap, range) {
  const flip = latitudes[0] < latitudes[latitudes.length - 1];
  const image = fieldToImageData(field, colormap, range, { flipVertical: flip, theme: shared.theme });
  const canvas = new OffscreenCanvas(field.width, field.height);
  canvas.getContext("2d").putImageData(image, 0, 0);
  return canvas;
}

function prefetchNeighbours(panel, level, start, leadIndex) {
  const manifest = manifestFor(panel.state.dataset);
  const maxLead = Math.max(...manifest.lead_days);
  for (const delta of [1, -1]) {
    const lead = shared.leadDay + delta;
    if (lead < 1 || lead > maxLead) continue;
    if (panel.state.mode === "currents") {
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
  const topLeft = projection.project(edges.nx0, edges.nyTop);
  const bottomRight = projection.project(edges.nx1, edges.nyBottom);
  context.drawImage(offscreen, topLeft.x, topLeft.y, bottomRight.x - topLeft.x, bottomRight.y - topLeft.y);
}

function drawPanel(panel) {
  const canvas = panel.els.field;
  const context = canvas.getContext("2d");
  context.fillStyle = shared.theme === "light" ? "#eef2f6" : "#080b11";
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (!panel.offscreenA) return;
  const projection = projectionFor(panel);
  context.imageSmoothingEnabled = false;

  const showBlink = panel.blink && panel.offscreenB;
  if (panel.state.mode === "field" && panel.state.compare && panel.offscreenB && !showBlink) {
    drawImageWorld(context, panel.offscreenA, panel.edgesA, projection);
    const dividerX = panel.swipeX * canvas.width;
    context.save();
    context.beginPath();
    context.rect(dividerX, 0, canvas.width - dividerX, canvas.height);
    context.clip();
    drawImageWorld(context, panel.offscreenB, panel.edgesB, projection);
    context.restore();
    context.strokeStyle = shared.theme === "light" ? "#1f6feb" : "#38bdf8";
    context.lineWidth = 2 * (window.devicePixelRatio || 1);
    context.beginPath();
    context.moveTo(dividerX, 0);
    context.lineTo(dividerX, canvas.height);
    context.stroke();
    panel.els.swipeHint.hidden = false;
    panel.els.swipeHint.textContent = `◀ ${labelFor(panel.state.dataset)}  |  ${labelFor(panel.state.datasetB)} ▶`;
  } else {
    drawImageWorld(context, showBlink ? panel.offscreenB : panel.offscreenA, showBlink ? panel.edgesB : panel.edgesA, projection);
    panel.els.swipeHint.hidden = !(panel.state.mode === "field" && panel.state.compare);
    if (!panel.els.swipeHint.hidden) panel.els.swipeHint.textContent = "hold B to blink";
  }
  updateParticleProjection(panel, projection);
  updatePanelBadge(panel);
}

function updatePanelBadge(panel) {
  const stats = panel.field ? fieldStatistics(panel.field) : { mean: NaN };
  const mean = Number.isFinite(stats.mean) ? stats.mean.toFixed(3) : "—";
  panel.els.badge.textContent = `${panel.units} · μ ${mean}`;
}

// ---- overlays ---------------------------------------------------------------

let overlayData = { eddies: null, class4: null, region: null };

async function loadOverlayData() {
  const slug = panels[activePanelIndex] ? panels[activePanelIndex].state.dataset : datasetCatalog[0].slug;
  const region = shared.region;
  const urls = insightsFor(insightIndex, "glonet_1_degree", region); // insights exist for glonet only
  void slug;
  overlayData.region = region;
  if (shared.overlayMode === "eddies") {
    overlayData.eddies = await loadEddies(urls.eddies);
  } else if (shared.overlayMode === "class4") {
    overlayData.class4 = await loadClass4(urls.class4_matchups);
  }
}

function drawOverlays(panel) {
  const canvas = panel.els.overlay;
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
  if (shared.overlayMode === "none") return;
  const projection = projectionFor(panel);
  const ratio = window.devicePixelRatio || 1;

  if (shared.overlayMode === "eddies" && overlayData.eddies) {
    const frame = eddyFrame(overlayData.eddies, shared.eddyReference, shared.leadDay);
    if (frame) drawEddyFrame(context, projection.project, frame.frame, { devicePixelRatio: ratio });
  } else if (shared.overlayMode === "class4" && overlayData.class4) {
    const manifest = manifestFor(panel.state.dataset);
    const entry = manifest && manifest.variables[panel.state.variable];
    const depthBin = class4DepthBin(entry);
    const startDate = manifest ? manifest.start_dates[Math.min(shared.startIndex, manifest.start_dates.length - 1)] : null;
    // Fewer points at low zoom (density-manage §4); refine as the user zooms in.
    const limit = Math.round(1500 + 3500 * Math.min(1, (view.zoom - 1) / 6));
    const points = class4Points(overlayData.class4, {
      variable: panel.state.variable,
      depthBin,
      leadDay: shared.leadDay,
      startDate,
      limit,
    });
    const scale = class4ErrorScale(points);
    drawClass4Points(context, projection.project, points, {
      devicePixelRatio: ratio,
      errorScale: scale,
      canvasWidth: canvas.width,
      canvasHeight: canvas.height,
    });
    panel.class4Scale = scale;
    panel.class4Count = points.length;
  }
}

function class4DepthBin(entry) {
  if (!entry) return null;
  if (entry.standard_name.includes("velocity")) return "15m";
  if (entry.standard_name === "sea_surface_height_above_geoid") return "surface";
  return "0-5m"; // temperature / salinity near-surface bin matching the surface viewer field
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
    speed: shared.particleSpeed,
    devicePixelRatio: window.devicePixelRatio || 1,
    playing: shared.particlesPlaying,
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

function updateParticleProjection(panel, projection) {
  if (!panel.particleContext) return;
  panel.particleContext.project = projection.project;
  panel.particleContext.viewport = visibleViewport(projection, panel.els.particles);
  panel.particleContext.theme = shared.theme;
  panel.particleContext.speed = shared.particleSpeed;
  panel.particleContext.playing = shared.particlesPlaying;
  panel.particleContext.devicePixelRatio = window.devicePixelRatio || 1;
}

function visibleViewport(projection, canvas) {
  const topLeft = projection.unproject(0, 0);
  const bottomRight = projection.unproject(canvas.width, canvas.height);
  return {
    minX: Math.max(0, Math.min(topLeft.nx, bottomRight.nx)),
    maxX: Math.min(1, Math.max(topLeft.nx, bottomRight.nx)),
    minY: Math.max(0, Math.min(topLeft.ny, bottomRight.ny)),
    maxY: Math.min(1, Math.max(topLeft.ny, bottomRight.ny)),
  };
}

// ---- pointer interaction (pan / zoom shared, hover per panel) ---------------

function beginPanelDrag(panel, event) {
  const projection = projectionFor(panel);
  if (panel.state.mode === "field" && panel.state.compare && panel.offscreenB) {
    const ratio = window.devicePixelRatio || 1;
    const rectangle = panel.els.field.getBoundingClientRect();
    const localX = (event.clientX - rectangle.left) * ratio;
    if (Math.abs(localX - panel.swipeX * panel.els.field.width) < 12 * ratio) {
      panel.draggingSwipe = true;
      return;
    }
  }
  panel.dragging = { x: event.clientX, y: event.clientY, centerNX: view.centerNX, centerNY: view.centerNY, projection };
}

function onGlobalMove(event) {
  for (const panel of panels) {
    if (panel.draggingSwipe) {
      const ratio = window.devicePixelRatio || 1;
      const rectangle = panel.els.field.getBoundingClientRect();
      panel.swipeX = Math.min(0.98, Math.max(0.02, ((event.clientX - rectangle.left) * ratio) / panel.els.field.width));
      drawPanel(panel);
      return;
    }
    if (panel.dragging) {
      const ratio = window.devicePixelRatio || 1;
      view.centerNX = panel.dragging.centerNX - ((event.clientX - panel.dragging.x) * ratio) / panel.dragging.projection.displayWidth;
      view.centerNY = panel.dragging.centerNY - ((event.clientY - panel.dragging.y) * ratio) / panel.dragging.projection.displayHeight;
      clampView();
      redrawAllPanels();
      scheduleHashWrite();
      return;
    }
  }
  updateHover(event);
}

function onGlobalUp() {
  let wasDragging = false;
  for (const panel of panels) {
    if (panel.dragging || panel.draggingSwipe) wasDragging = true;
    panel.dragging = null;
    panel.draggingSwipe = false;
  }
  if (wasDragging) writeHash();
}

function onPanelWheel(panel, event) {
  event.preventDefault();
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
  const factor = Math.exp(-event.deltaY * 0.0015);
  const previousZoom = view.zoom;
  view.zoom = Math.min(60, Math.max(1, view.zoom * factor));
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
      updateSmallMultiples();
    });
  } else {
    redrawAllPanels();
  }
  scheduleHashWrite();
}

function clampView() {
  view.centerNX = Math.min(1, Math.max(0, view.centerNX));
  view.centerNY = Math.min(1, Math.max(0, view.centerNY));
}

function updateHover(event) {
  for (const panel of panels) {
    const rectangle = panel.els.field.getBoundingClientRect();
    if (
      event.clientX < rectangle.left ||
      event.clientX > rectangle.right ||
      event.clientY < rectangle.top ||
      event.clientY > rectangle.bottom
    ) {
      continue;
    }
    if (!panel.field) continue;
    const ratio = window.devicePixelRatio || 1;
    const projection = projectionFor(panel);
    const point = projection.unproject((event.clientX - rectangle.left) * ratio, (event.clientY - rectangle.top) * ratio);
    const lon = point.nx * 360 - 180;
    const lat = 90 - point.ny * 180;
    const column = nearestIndex(panel.longitudes, lon);
    const row = nearestIndex(panel.latitudes, lat);
    if (column < 0 || row < 0) {
      panel.els.readout.textContent = "";
      continue;
    }
    const value = panel.field.data[row * panel.field.width + column];
    const valueText = Number.isNaN(value) ? "land / no data" : `${value.toFixed(3)} ${panel.units}`;
    panel.els.readout.textContent = `${lat.toFixed(2)}°, ${lon.toFixed(2)}° — ${valueText}`;
  }
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
  grid.dataset.layout = String(shared.layout);
  while (panels.length < shared.layout) {
    const panel = buildPanel(panels.length);
    panels.push(panel);
  }
  grid.innerHTML = "";
  for (let i = 0; i < shared.layout; i += 1) {
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
  updateSmallMultiples();
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
  for (let i = 0; i < shared.layout; i += 1) {
    resizePanelCanvases(panels[i]);
    drawPanel(panels[i]);
    drawOverlays(panels[i]);
  }
}

function renderAllPanels() {
  const jobs = [];
  for (let i = 0; i < shared.layout; i += 1) jobs.push(renderPanel(panels[i]));
  return Promise.all(jobs);
}

// ---- shared colorbar --------------------------------------------------------

function updateSharedColorbar() {
  const panel = panels[activePanelIndex];
  if (!panel || !panel.colormap || !panel.range) return;
  const sameVariable = panels
    .slice(0, shared.layout)
    .every((candidate) => candidate.state.variable === panel.state.variable && candidate.state.mode === panel.state.mode);
  const suffix = shared.layout > 1 ? (sameVariable ? " (shared)" : ` (panel ${activePanelIndex + 1})`) : "";
  drawColorbar(elements.colorbar, panel.colormap, panel.range, {
    label: `${panel.label} (${panel.units})${suffix}`,
    textColor: shared.theme === "light" ? "#14181d" : "#e6edf3",
  });
  const manifest = manifestFor(panel.state.dataset);
  elements["layer-info"].textContent = `start ${manifest.start_dates[Math.min(shared.startIndex, manifest.start_dates.length - 1)]} · lead day ${shared.leadDay} · zoom ${view.zoom.toFixed(1)}× · level ${selectRenderLevel(manifest)}`;
}

// ---- context rail -----------------------------------------------------------

async function updateContextRail() {
  const panel = panels[activePanelIndex];
  if (!panel) return;
  const manifest = manifestFor(panel.state.dataset);
  const entry = manifest && manifest.variables[panel.state.variable];
  const depth = entry ? entry.depth : null;
  elements["rail-subtitle"].textContent = `${panel.label} · ${shared.region}`;

  // Lead-time curve: every score series matching this variable/depth (region = summary's global).
  const depthKey = entry ? mapDepthToScoreDepth(entry) : null;
  const series = new Map();
  let unit = "";
  for (const row of scoresSummary) {
    if (row.variable !== panel.state.variable) continue;
    if (depthKey && row.depth !== depthKey) continue;
    unit = row.unit || unit;
    const key = row.reference || "reference";
    if (!series.has(key)) series.set(key, []);
    series.get(key).push(row);
  }
  elements["rail-lead-curve"].innerHTML = leadCurveSVG(series, { unit });

  // Spectrum for variable + region (only SSH spectra are produced today).
  const spectra = await loadSpectra(insightsFor(insightIndex, "glonet_1_degree", shared.region).spectra);
  const spectrumEntry = spectra ? spectraEntry(spectra, panel.state.variable, shared.eddyReference, shared.leadDay) : null;
  elements["rail-spectra"].innerHTML = spectraSVG(spectrumEntry, {});

  updateRailLegend(panel);
  void depth;
}

function mapDepthToScoreDepth(entry) {
  if (entry.standard_name.includes("velocity") && entry.depth === "15m") return "15m";
  if (entry.depth === "surface") return "surface";
  return entry.depth;
}

function updateRailLegend(panel) {
  const section = elements["rail-legend-section"];
  const container = elements["rail-legend"];
  if (shared.overlayMode === "none") {
    section.hidden = true;
    return;
  }
  section.hidden = false;
  if (shared.overlayMode === "eddies") {
    const frame = overlayData.eddies ? eddyFrame(overlayData.eddies, shared.eddyReference, shared.leadDay) : null;
    const counts = frame ? frame.frame : { matches: [], spurious: [], missed: [] };
    container.innerHTML =
      row(EDDY_COLORS.matched, "Matched", (counts.matches || []).length) +
      row(EDDY_COLORS.spurious, "Spurious (model only)", (counts.spurious || []).length) +
      row(EDDY_COLORS.missed, "Missed (reference only)", (counts.missed || []).length) +
      `<p class="dim">vs ${shared.eddyReference}, lead ${frame ? frame.frame.lead_day : "—"} (nearest available)</p>`;
  } else if (shared.overlayMode === "class4") {
    const scale = panel.class4Scale || 0;
    container.innerHTML =
      `<div class="row"><span class="swatch" style="background:${SERIES_COLORS.error}"></span>|obs − model|, brighter = larger error</div>` +
      `<p class="dim">${panel.class4Count || 0} points shown · scale ≈ ${scale ? scale.toFixed(3) : "—"} ${panel.units} · region ${shared.region}${overlayData.class4 && overlayData.class4.sampled ? " · sampled subset" : ""}</p>`;
  } else if (shared.overlayMode === "trajectories") {
    container.innerHTML = `<p class="dim">${trajectoryNote}</p>`;
  }
}

function row(color, label, count) {
  return `<div class="row"><span class="swatch" style="background:${color}"></span>${label} — <strong>${count}</strong></div>`;
}

// ---- global controls --------------------------------------------------------

let trajectoryNote = "";

function updateCurrentsControlVisibility() {
  const anyCurrents = panels.slice(0, shared.layout).some((panel) => panel.state.mode === "currents");
  elements["currents-group"].hidden = !anyCurrents;
}

async function applyOverlayMode() {
  const region = shared.region;
  elements["eddy-reference-field"].hidden = shared.overlayMode !== "eddies";
  const note = elements["overlay-note"];
  if (shared.overlayMode === "trajectories") {
    const result = await loadTrajectories(insightIndex, "glonet_1_degree", region);
    trajectoryNote = result.available ? "trajectories loaded" : `Trajectories: ${result.reason}.`;
    note.textContent = trajectoryNote;
  } else if (shared.overlayMode === "class4" && region !== "ibi") {
    note.textContent = "Class-4 match-ups are only available for the IBI region — switch region to IBI.";
  } else if (shared.overlayMode === "eddies" || shared.overlayMode === "class4") {
    note.textContent = "Overlay shows glonet_1_degree insights.";
  } else {
    note.textContent = "";
  }
  await loadOverlayData();
  for (let i = 0; i < shared.layout; i += 1) drawOverlays(panels[i]);
  updateContextRail();
}

function applyTheme() {
  document.documentElement.dataset.theme = shared.theme;
  elements["theme-toggle"].textContent = shared.theme === "light" ? "Dark theme" : "Light theme";
}

function wireGlobalControls() {
  for (const button of document.querySelectorAll(".layout-switch [data-layout]")) {
    button.addEventListener("click", () => {
      shared.layout = Number(button.dataset.layout);
      markLayoutButtons();
      syncPanelGrid();
      renderAllPanels().then(() => {
        updateSharedColorbar();
        updateContextRail();
        updateCurrentsControlVisibility();
        updateSmallMultiples();
      });
      writeHash();
    });
  }
  elements["start-date"].addEventListener("change", (event) => {
    shared.startIndex = Number(event.target.value);
    renderAllPanels().then(() => {
      loadOverlayData().then(() => redrawOverlaysAll());
      updateContextRail();
      updateSmallMultiples();
    });
    writeHash();
  });
  elements["lead-day"].addEventListener("input", (event) => {
    shared.leadDay = Number(event.target.value);
    elements["lead-value"].textContent = `day ${shared.leadDay}`;
    renderAllPanels().then(() => {
      redrawOverlaysAll();
      updateContextRail();
    });
    scheduleHashWrite();
  });
  elements["overlay-mode"].addEventListener("change", (event) => {
    shared.overlayMode = event.target.value;
    applyOverlayMode();
    writeHash();
  });
  elements["overlay-region"].addEventListener("change", (event) => {
    shared.region = event.target.value;
    applyOverlayMode();
    updateContextRail();
    writeHash();
  });
  elements["eddy-reference"].addEventListener("change", (event) => {
    shared.eddyReference = event.target.value;
    redrawOverlaysAll();
    updateContextRail();
    writeHash();
  });
  elements["particles-play"].addEventListener("change", (event) => {
    shared.particlesPlaying = event.target.checked;
    for (const panel of panels) if (panel.particleContext) panel.particleContext.playing = shared.particlesPlaying;
    writeHash();
  });
  elements["particle-speed"].addEventListener("input", (event) => {
    shared.particleSpeed = Number(event.target.value);
    elements["speed-value"].textContent = `${shared.particleSpeed.toFixed(1)}×`;
    for (const panel of panels) if (panel.particleContext) panel.particleContext.speed = shared.particleSpeed;
    scheduleHashWrite();
  });
  elements["theme-toggle"].addEventListener("click", () => {
    shared.theme = shared.theme === "light" ? "dark" : "light";
    applyTheme();
    renderAllPanels().then(() => {
      redrawOverlaysAll();
      updateSmallMultiples();
    });
    writeHash();
  });
  elements["rail-toggle"].addEventListener("click", () => {
    shared.railOpen = !shared.railOpen;
    elements["context-rail"].hidden = !shared.railOpen;
    writeHash();
  });
  elements["reset-view"].addEventListener("click", () => {
    const previousLevels = panels.slice(0, shared.layout).map((panel) => selectRenderLevel(manifestFor(panel.state.dataset)));
    view.zoom = 1;
    view.centerNX = 0.5;
    view.centerNY = 0.5;
    const needsRerender = panels.slice(0, shared.layout).some((panel, index) => previousLevels[index] !== selectRenderLevel(manifestFor(panel.state.dataset)));
    if (needsRerender) renderAllPanels().then(() => redrawOverlaysAll());
    else redrawAllPanels();
    writeHash();
  });

  window.addEventListener("mousemove", onGlobalMove);
  window.addEventListener("mouseup", onGlobalUp);
  window.addEventListener("resize", () => redrawAllPanels());
  window.addEventListener("keydown", (event) => {
    if (event.key.toLowerCase() === "b") {
      const panel = panels[activePanelIndex];
      if (panel && panel.offscreenB) {
        panel.blink = true;
        drawPanel(panel);
      }
    }
  });
  window.addEventListener("keyup", (event) => {
    if (event.key.toLowerCase() === "b") {
      for (const panel of panels) {
        if (panel.blink) {
          panel.blink = false;
          drawPanel(panel);
        }
      }
    }
  });
}

function redrawOverlaysAll() {
  for (let i = 0; i < shared.layout; i += 1) drawOverlays(panels[i]);
  updateRailLegend(panels[activePanelIndex]);
}

function markLayoutButtons() {
  for (const button of document.querySelectorAll(".layout-switch [data-layout]")) {
    button.classList.toggle("active", Number(button.dataset.layout) === shared.layout);
  }
}

// ---- small-multiples error strip (feature 6) --------------------------------

const STRIP_LEADS = [1, 3, 5, 7, 10];

async function updateSmallMultiples() {
  const strip = elements["small-multiples"];
  const panel = panels[activePanelIndex];
  // Needs an A and a B to difference: a diff panel, or a field panel with compare on.
  const hasPair = panel && (panel.state.mode === "diff" || (panel.state.mode === "field" && panel.state.compare && panel.state.datasetB));
  if (!hasPair) {
    strip.hidden = true;
    return;
  }
  strip.hidden = false;
  elements["strip-title"].textContent = `Error growth: ${labelFor(panel.state.dataset)} − ${labelFor(panel.state.datasetB)} · ${prettyName(manifestFor(panel.state.dataset).variables[panel.state.variable].standard_name)}`;
  const manifest = manifestFor(panel.state.dataset);
  const level = selectRenderLevel(manifest);
  const start = Math.min(shared.startIndex, manifest.start_dates.length - 1);
  const maxLead = Math.max(...manifest.lead_days);
  const leads = STRIP_LEADS.filter((lead) => lead <= maxLead);

  const diffs = [];
  await ensureStore(panel.state.datasetB);
  const compareLevel = renderLevelForSlug(panel.state.datasetB);
  for (const lead of leads) {
    const primary = await readAlignedField(panel, panel.state.dataset, panel.state.variable, level, start, lead - 1);
    const compare = await readAlignedField(
      panel,
      panel.state.datasetB,
      panel.state.variable,
      compareLevel,
      start,
      lead - 1,
      primary.latitudes,
      primary.longitudes,
    );
    diffs.push({ lead, field: differenceField(primary.field, compare.field), latitudes: primary.latitudes });
  }
  // Shared diverging scale across all leads (contracts.md §6).
  let magnitude = 0;
  for (const item of diffs) {
    const range = symmetricRange(item.field);
    magnitude = Math.max(magnitude, range[1]);
  }
  const sharedRange = [-magnitude || -1, magnitude || 1];

  const row = elements["strip-row"];
  row.innerHTML = "";
  const ratio = window.devicePixelRatio || 1;
  for (const item of diffs) {
    const cell = document.createElement("div");
    cell.className = "strip-cell";
    const canvas = document.createElement("canvas");
    const flip = item.latitudes[0] < item.latitudes[item.latitudes.length - 1];
    const image = fieldToImageData(item.field, DIFFERENCE_COLORMAP, sharedRange, { flipVertical: flip, theme: shared.theme });
    const displayWidth = Math.round((row.clientWidth - 16) / diffs.length);
    const displayHeight = Math.round((displayWidth * item.field.height) / item.field.width);
    canvas.width = Math.max(1, Math.round(displayWidth * ratio));
    canvas.height = Math.max(1, Math.round(displayHeight * ratio));
    const source = new OffscreenCanvas(item.field.width, item.field.height);
    source.getContext("2d").putImageData(image, 0, 0);
    const context = canvas.getContext("2d");
    context.imageSmoothingEnabled = false;
    context.drawImage(source, 0, 0, canvas.width, canvas.height);
    const caption = document.createElement("span");
    caption.textContent = `lead ${item.lead}`;
    cell.appendChild(canvas);
    cell.appendChild(caption);
    row.appendChild(cell);
  }
  elements["strip-title"].textContent += `  ·  ±${magnitude ? magnitude.toFixed(3) : "1"} ${panel.units}`;
}

// ---- URL hash (every view state is a URL — §6) ------------------------------

let hashWriteTimer = null;
function scheduleHashWrite() {
  clearTimeout(hashWriteTimer);
  hashWriteTimer = setTimeout(writeHash, 250);
}

function encodePanel(panel) {
  return [panel.state.dataset, panel.state.variable, panel.state.mode, panel.state.datasetB, panel.state.compare ? "1" : "0"].join(",");
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
  if (shared.overlayMode !== "none") parameters.set("ov", shared.overlayMode);
  parameters.set("region", shared.region);
  if (shared.overlayMode === "eddies") parameters.set("eref", shared.eddyReference);
  if (!shared.railOpen) parameters.set("rail", "0");
  parameters.set("play", shared.particlesPlaying ? "1" : "0");
  parameters.set("spd", shared.particleSpeed.toFixed(1));
  for (let i = 0; i < shared.layout; i += 1) parameters.set(`p${i}`, encodePanel(panels[i]));
  const encoded = `#${parameters.toString()}`;
  if (encoded !== location.hash) history.replaceState(null, "", encoded);
}

function readHash() {
  const parameters = new URLSearchParams(location.hash.slice(1));
  if (parameters.has("layout")) shared.layout = Number(parameters.get("layout"));
  if (parameters.has("s")) shared.startIndex = Number(parameters.get("s"));
  if (parameters.has("l")) shared.leadDay = Number(parameters.get("l"));
  if (parameters.has("z")) view.zoom = Number(parameters.get("z"));
  if (parameters.has("cx")) view.centerNX = Number(parameters.get("cx"));
  if (parameters.has("cy")) view.centerNY = Number(parameters.get("cy"));
  if (parameters.has("theme")) shared.theme = parameters.get("theme");
  if (parameters.has("ov")) shared.overlayMode = parameters.get("ov");
  if (parameters.has("region")) shared.region = parameters.get("region");
  if (parameters.has("eref")) shared.eddyReference = parameters.get("eref");
  if (parameters.get("rail") === "0") shared.railOpen = false;
  if (parameters.has("play")) shared.particlesPlaying = parameters.get("play") === "1";
  if (parameters.has("spd")) shared.particleSpeed = Number(parameters.get("spd"));
  return parameters;
}

function applyPanelHash(parameters) {
  for (let i = 0; i < shared.layout; i += 1) {
    const encoded = parameters.get(`p${i}`);
    if (!encoded) continue;
    const [dataset, variable, mode, datasetB, compare] = encoded.split(",");
    if (!panels[i]) panels[i] = buildPanel(i);
    Object.assign(panels[i].state, {
      dataset: dataset || panels[i].state.dataset,
      variable: variable || panels[i].state.variable,
      mode: mode || panels[i].state.mode,
      datasetB: datasetB || panels[i].state.datasetB,
      compare: compare === "1",
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
    "rail-toggle",
    "panel-grid",
    "small-multiples",
    "strip-title",
    "strip-row",
    "colorbar",
    "layer-info",
    "status",
    "context-rail",
    "rail-subtitle",
    "rail-lead-curve",
    "rail-spectra",
    "rail-legend",
    "rail-legend-section",
  ]) {
    elements[id] = document.getElementById(id);
  }
}

// ---- boot -------------------------------------------------------------------

async function main() {
  selectElements();
  setStatus("Loading catalog…");
  try {
    const response = await fetch(DATASETS_URL);
    if (!response.ok) throw new Error(`Cannot load ${DATASETS_URL} (${response.status})`);
    datasetCatalog = (await response.json()).datasets;
  } catch (error) {
    setStatus(`${error.message}. Populate ./data/ with viewer pyramids and datasets.json (see README).`, true);
    return;
  }
  if (!datasetCatalog.length) {
    setStatus("No datasets in ./data/datasets.json.", true);
    return;
  }

  const parameters = readHash();
  if (!Number.isFinite(shared.layout) || ![1, 2, 4].includes(shared.layout)) shared.layout = 1;
  applyTheme();
  elements["context-rail"].hidden = !shared.railOpen;
  elements["lead-value"].textContent = `day ${shared.leadDay}`;
  elements["speed-value"].textContent = `${shared.particleSpeed.toFixed(1)}×`;
  elements["particles-play"].checked = shared.particlesPlaying;
  elements["particle-speed"].value = String(shared.particleSpeed);
  elements["overlay-mode"].value = shared.overlayMode;
  elements["overlay-region"].value = shared.region;
  elements["eddy-reference"].value = shared.eddyReference;

  // Insight index + score summary load in the background; overlays/rail wait on them.
  insightIndex = await loadInsightIndex();
  scoresSummary = await loadScoresSummary(insightIndex);

  // Ensure the primary dataset store so start-date / lead options are known.
  for (let i = 0; i < shared.layout; i += 1) if (!panels[i]) panels[i] = buildPanel(i);
  applyPanelHash(parameters);
  // Warm every visible panel's store so variable/start selectors populate on first paint.
  await Promise.all(panels.slice(0, shared.layout).map((panel) => ensureStore(panel.state.dataset).catch(() => {})));
  const manifest = manifestFor(panels[0].state.dataset);
  populateSelect(
    elements["start-date"],
    manifest.start_dates.map((date, index) => ({ value: index, label: date })),
    shared.startIndex,
  );
  elements["lead-day"].min = String(Math.min(...manifest.lead_days));
  elements["lead-day"].max = String(Math.max(...manifest.lead_days));
  elements["lead-day"].value = String(shared.leadDay);

  markLayoutButtons();
  syncPanelGrid();
  wireGlobalControls();

  await renderAllPanels();
  updateCurrentsControlVisibility();
  updateSharedColorbar();
  await applyOverlayMode();
  await updateContextRail();
  await updateSmallMultiples();
  writeHash();
}

main();
