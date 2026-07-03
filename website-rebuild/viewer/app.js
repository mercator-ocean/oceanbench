// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// OceanBench viewer v1 — equirectangular field explorer (contracts.md §6).
// Single map panel, difference mode, lead-day scrubber, hover readout, perceptually
// uniform colormaps, zoom/pan. Every view state lives in the URL hash.

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

const DATASETS_URL = "./data/datasets.json";
const DEFAULT_DIFFERENCE_COLORMAP = "balance";

const state = {
  datasetA: null,
  datasetB: null,
  variable: null,
  startIndex: 0,
  leadDay: 1,
  difference: false,
  colormap: null,
  theme: "dark",
  zoom: 1,
  centerX: 0.5,
  centerY: 0.5,
};

const stores = new Map();
const manifests = new Map();
const coordinatesByLevel = new Map();
let datasetCatalog = [];
let renderToken = 0;
let currentField = null;
let currentLatitudes = null;
let currentLongitudes = null;
let currentFlipVertical = false;
let currentView = null;

const elements = {};

function selectElements() {
  for (const id of [
    "dataset-a",
    "dataset-b",
    "variable",
    "start-date",
    "lead-day",
    "lead-value",
    "difference-toggle",
    "dataset-b-field",
    "colormap",
    "theme-toggle",
    "map-canvas",
    "colorbar",
    "readout",
    "status",
    "layer-info",
    "reset-view",
  ]) {
    elements[id] = document.getElementById(id);
  }
}

function setStatus(message, isError = false) {
  elements.status.textContent = message;
  elements.status.classList.toggle("error", isError);
  elements.status.hidden = !message;
}

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

function chooseLevel(manifest) {
  // Level 0 is native (finest). A multi-level pyramid would pick a coarser level
  // when zoomed far out to cap tiles fetched; the single-level 1-degree data is
  // always level 0. Structured around the levels array so finer data Just Works.
  const levels = manifest.levels;
  if (levels.length === 1 || state.zoom > 1.5) return levels[0].level;
  const targetCells = Math.max(elements["map-canvas"].width, 720) / 2;
  let chosen = levels[0];
  for (const level of levels) {
    if (level.longitude_size >= targetCells) chosen = level;
  }
  return chosen.level;
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

function variableLabel(manifest, key) {
  const entry = manifest.variables[key];
  const depth = entry.depth === "surface" ? "surface" : entry.depth;
  return `${prettyName(entry.standard_name)} · ${depth}`;
}

function prettyName(standardName) {
  return standardName.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
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

function refreshControlsForDatasets() {
  const manifest = manifestFor(state.datasetA);
  populateSelect(
    elements.variable,
    Object.keys(manifest.variables).map((key) => ({ value: key, label: variableLabel(manifest, key) })),
    state.variable,
  );
  populateSelect(
    elements["start-date"],
    manifest.start_dates.map((date, index) => ({ value: index, label: date })),
    state.startIndex,
  );
  elements["lead-day"].min = String(Math.min(...manifest.lead_days));
  elements["lead-day"].max = String(Math.max(...manifest.lead_days));
  elements["lead-day"].value = String(state.leadDay);
  elements["lead-value"].textContent = `day ${state.leadDay}`;
  const colormapValue = state.difference ? DEFAULT_DIFFERENCE_COLORMAP : effectiveColormap();
  populateSelect(
    elements.colormap,
    COLORMAP_NAMES.map((name) => ({ value: name, label: name })),
    colormapValue,
  );
  elements["dataset-b-field"].hidden = !state.difference;
  elements["difference-toggle"].checked = state.difference;
}

function effectiveColormap() {
  if (state.difference) return DEFAULT_DIFFERENCE_COLORMAP;
  if (state.colormap) return state.colormap;
  const manifest = manifestFor(state.datasetA);
  return manifest.variables[state.variable].default_colormap;
}

function defaultRangeFor() {
  const manifest = manifestFor(state.datasetA);
  const entry = manifest.variables[state.variable];
  return entry.default_range;
}

async function render() {
  const token = ++renderToken;
  try {
    await ensureStore(state.datasetA);
    if (state.difference && state.datasetB) await ensureStore(state.datasetB);
    if (token !== renderToken) return;

    const manifest = manifestFor(state.datasetA);
    if (!(state.variable in manifest.variables)) state.variable = Object.keys(manifest.variables)[0];
    const level = chooseLevel(manifest);
    const layerRequest = { variable: state.variable, level, startIndex: state.startIndex, leadIndex: state.leadDay - 1 };

    setStatus("Fetching tiles…");
    const startedAt = performance.now();
    const fieldA = await readLayer(stores.get(state.datasetA), layerRequest);
    if (token !== renderToken) return;
    const coordinates = await loadCoordinates(state.datasetA, level);
    if (token !== renderToken) return;

    let field = fieldA;
    let range;
    let colormap = effectiveColormap();
    let compressedBytes = fieldA.compressedBytes;
    if (state.difference && state.datasetB) {
      const fieldB = await readLayer(stores.get(state.datasetB), layerRequest);
      if (token !== renderToken) return;
      const coordinatesB = await loadCoordinates(state.datasetB, level);
      if (token !== renderToken) return;
      compressedBytes += fieldB.compressedBytes;
      const alignedB = resampleOntoGrid(
        fieldB,
        coordinatesB.latitudes,
        coordinatesB.longitudes,
        coordinates.latitudes,
        coordinates.longitudes,
      );
      field = differenceField(fieldA, alignedB);
      range = symmetricRange(field);
      colormap = DEFAULT_DIFFERENCE_COLORMAP;
    } else {
      range = state.difference ? symmetricRange(field) : defaultRangeFor();
    }

    currentLatitudes = coordinates.latitudes;
    currentLongitudes = coordinates.longitudes;
    currentFlipVertical = coordinates.latitudes[0] < coordinates.latitudes[coordinates.latitudes.length - 1];
    currentField = field;

    const image = fieldToImageData(field, colormap, range, {
      flipVertical: currentFlipVertical,
      theme: state.theme,
    });
    paint(image);
    drawColorbar(elements.colorbar, colormap, range, {
      label: colorbarLabel(manifest),
      textColor: state.theme === "light" ? "#14181d" : "#e6edf3",
    });

    const statistics = fieldStatistics(field);
    const elapsed = performance.now() - startedAt;
    elements["layer-info"].textContent = layerInfoText(manifest, statistics, compressedBytes, elapsed);
    setStatus("");
    prefetchAdjacentLeads(layerRequest, manifest);
    writeHash();
  } catch (error) {
    if (token === renderToken) setStatus(String(error.message || error), true);
    console.error(error);
  }
}

function colorbarLabel(manifest) {
  const entry = manifest.variables[state.variable];
  if (state.difference && state.datasetB) {
    return `${labelFor(state.datasetA)} − ${labelFor(state.datasetB)} (${entry.units})`;
  }
  return `${prettyName(entry.standard_name)} (${entry.units})`;
}

function labelFor(slug) {
  const descriptor = datasetCatalog.find((entry) => entry.slug === slug);
  return descriptor ? descriptor.label : slug;
}

function layerInfoText(manifest, statistics, compressedBytes, elapsed) {
  const date = manifest.start_dates[state.startIndex];
  const kilobytes = (compressedBytes / 1024).toFixed(0);
  const mean = Number.isFinite(statistics.mean) ? statistics.mean.toFixed(3) : "—";
  return `start ${date} · lead day ${state.leadDay} · mean ${mean} · ${kilobytes} KB · ${elapsed.toFixed(0)} ms`;
}

function prefetchAdjacentLeads(layerRequest, manifest) {
  const maxLead = Math.max(...manifest.lead_days);
  for (const delta of [1, -1]) {
    const lead = state.leadDay + delta;
    if (lead < 1 || lead > maxLead) continue;
    prefetchLayer(stores.get(state.datasetA), { ...layerRequest, leadIndex: lead - 1 });
    if (state.difference && state.datasetB) {
      prefetchLayer(stores.get(state.datasetB), { ...layerRequest, leadIndex: lead - 1 });
    }
  }
}

// ---- map view (pan / zoom / hover) ------------------------------------------

let offscreen = null;

function paint(image) {
  if (!offscreen || offscreen.width !== image.width || offscreen.height !== image.height) {
    offscreen = new OffscreenCanvas(image.width, image.height);
  }
  offscreen.getContext("2d").putImageData(image, 0, 0);
  draw();
}

function computeView() {
  const canvas = elements["map-canvas"];
  const width = canvas.width;
  const height = canvas.height;
  const gridWidth = offscreen ? offscreen.width : 360;
  const gridHeight = offscreen ? offscreen.height : 168;
  const fitScale = Math.min(width / gridWidth, height / gridHeight);
  const scale = fitScale * state.zoom;
  const displayWidth = gridWidth * scale;
  const displayHeight = gridHeight * scale;
  const originX = width / 2 - state.centerX * displayWidth;
  const originY = height / 2 - state.centerY * displayHeight;
  return { scale, originX, originY, gridWidth, gridHeight, displayWidth, displayHeight };
}

function draw() {
  const canvas = elements["map-canvas"];
  const context = canvas.getContext("2d");
  context.fillStyle = state.theme === "light" ? "#eef2f6" : "#080b11";
  context.fillRect(0, 0, canvas.width, canvas.height);
  if (!offscreen) return;
  const view = computeView();
  currentView = view;
  context.imageSmoothingEnabled = false;
  context.drawImage(offscreen, view.originX, view.originY, view.displayWidth, view.displayHeight);
}

function resizeCanvas() {
  const canvas = elements["map-canvas"];
  const rectangle = canvas.parentElement.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rectangle.width * ratio);
  canvas.height = Math.round(rectangle.height * ratio);
  canvas.style.width = `${rectangle.width}px`;
  canvas.style.height = `${rectangle.height}px`;
  draw();
}

function pointerToCell(event) {
  if (!currentView || !currentField) return null;
  const canvas = elements["map-canvas"];
  const rectangle = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const x = (event.clientX - rectangle.left) * ratio;
  const y = (event.clientY - rectangle.top) * ratio;
  const column = Math.floor((x - currentView.originX) / currentView.scale);
  const screenRow = Math.floor((y - currentView.originY) / currentView.scale);
  if (column < 0 || column >= currentField.width || screenRow < 0 || screenRow >= currentField.height) return null;
  const dataRow = currentFlipVertical ? currentField.height - 1 - screenRow : screenRow;
  return { column, dataRow };
}

function updateReadout(event) {
  const cell = pointerToCell(event);
  if (!cell) {
    elements.readout.textContent = "";
    return;
  }
  const value = currentField.data[cell.dataRow * currentField.width + cell.column];
  const latitude = currentLatitudes[cell.dataRow];
  const longitude = currentLongitudes[cell.column];
  const manifest = manifestFor(state.datasetA);
  const units = manifest.variables[state.variable].units;
  const valueText = Number.isNaN(value) ? "land / no data" : `${value.toFixed(3)} ${units}`;
  elements.readout.textContent = `${latitude.toFixed(2)}°, ${longitude.toFixed(2)}° — ${valueText}`;
}

let dragging = null;

function beginDrag(event) {
  dragging = { x: event.clientX, y: event.clientY, centerX: state.centerX, centerY: state.centerY };
}

function duringDrag(event) {
  updateReadout(event);
  if (!dragging || !currentView) return;
  const ratio = window.devicePixelRatio || 1;
  state.centerX = dragging.centerX - ((event.clientX - dragging.x) * ratio) / currentView.displayWidth;
  state.centerY = dragging.centerY - ((event.clientY - dragging.y) * ratio) / currentView.displayHeight;
  clampCenter();
  draw();
}

function endDrag() {
  if (dragging) writeHash();
  dragging = null;
}

function clampCenter() {
  state.centerX = Math.min(1, Math.max(0, state.centerX));
  state.centerY = Math.min(1, Math.max(0, state.centerY));
}

function onWheel(event) {
  event.preventDefault();
  const factor = Math.exp(-event.deltaY * 0.0015);
  const previousZoom = state.zoom;
  state.zoom = Math.min(40, Math.max(1, state.zoom * factor));
  if (state.zoom === previousZoom) return;
  draw();
  scheduleHashWrite();
  maybeReloadLevel();
}

let levelReloadTimer = null;
function maybeReloadLevel() {
  clearTimeout(levelReloadTimer);
  levelReloadTimer = setTimeout(() => {
    const manifest = manifestFor(state.datasetA);
    if (manifest && manifest.levels.length > 1) render();
  }, 150);
}

// ---- URL hash (view state is a URL — §6) ------------------------------------

let hashWriteTimer = null;
function scheduleHashWrite() {
  clearTimeout(hashWriteTimer);
  hashWriteTimer = setTimeout(writeHash, 250);
}

function writeHash() {
  const parameters = new URLSearchParams();
  parameters.set("a", state.datasetA);
  parameters.set("v", state.variable);
  parameters.set("s", String(state.startIndex));
  parameters.set("l", String(state.leadDay));
  if (state.difference) parameters.set("diff", "1");
  if (state.datasetB) parameters.set("b", state.datasetB);
  if (!state.difference && state.colormap) parameters.set("cmap", state.colormap);
  parameters.set("z", state.zoom.toFixed(3));
  parameters.set("cx", state.centerX.toFixed(4));
  parameters.set("cy", state.centerY.toFixed(4));
  parameters.set("theme", state.theme);
  const encoded = `#${parameters.toString()}`;
  if (encoded !== location.hash) history.replaceState(null, "", encoded);
}

function readHash() {
  const parameters = new URLSearchParams(location.hash.slice(1));
  if (parameters.has("a")) state.datasetA = parameters.get("a");
  if (parameters.has("b")) state.datasetB = parameters.get("b");
  if (parameters.has("v")) state.variable = parameters.get("v");
  if (parameters.has("s")) state.startIndex = Number(parameters.get("s"));
  if (parameters.has("l")) state.leadDay = Number(parameters.get("l"));
  state.difference = parameters.get("diff") === "1";
  if (parameters.has("cmap")) state.colormap = parameters.get("cmap");
  if (parameters.has("z")) state.zoom = Number(parameters.get("z"));
  if (parameters.has("cx")) state.centerX = Number(parameters.get("cx"));
  if (parameters.has("cy")) state.centerY = Number(parameters.get("cy"));
  if (parameters.has("theme")) state.theme = parameters.get("theme");
}

// ---- wiring -----------------------------------------------------------------

function applyTheme() {
  document.documentElement.dataset.theme = state.theme;
  elements["theme-toggle"].textContent = state.theme === "light" ? "Dark theme" : "Light theme";
}

function wireControls() {
  elements["dataset-a"].addEventListener("change", (event) => {
    state.datasetA = event.target.value;
    ensureStore(state.datasetA).then(() => {
      const manifest = manifestFor(state.datasetA);
      if (!(state.variable in manifest.variables)) state.variable = Object.keys(manifest.variables)[0];
      state.startIndex = Math.min(state.startIndex, manifest.start_dates.length - 1);
      refreshControlsForDatasets();
      render();
    });
  });
  elements["dataset-b"].addEventListener("change", (event) => {
    state.datasetB = event.target.value;
    render();
  });
  elements.variable.addEventListener("change", (event) => {
    state.variable = event.target.value;
    if (!state.difference) state.colormap = null;
    refreshControlsForDatasets();
    render();
  });
  elements["start-date"].addEventListener("change", (event) => {
    state.startIndex = Number(event.target.value);
    render();
  });
  elements["lead-day"].addEventListener("input", (event) => {
    state.leadDay = Number(event.target.value);
    elements["lead-value"].textContent = `day ${state.leadDay}`;
    render();
  });
  elements["difference-toggle"].addEventListener("change", (event) => {
    state.difference = event.target.checked;
    if (state.difference && !state.datasetB) {
      state.datasetB = datasetCatalog.find((entry) => entry.slug !== state.datasetA)?.slug || null;
    }
    refreshControlsForDatasets();
    populateSelect(
      elements["dataset-b"],
      datasetCatalog.map((entry) => ({ value: entry.slug, label: entry.label })),
      state.datasetB,
    );
    render();
  });
  elements.colormap.addEventListener("change", (event) => {
    state.colormap = event.target.value;
    render();
  });
  elements["theme-toggle"].addEventListener("click", () => {
    state.theme = state.theme === "light" ? "dark" : "light";
    applyTheme();
    render();
    writeHash();
  });
  elements["reset-view"].addEventListener("click", () => {
    state.zoom = 1;
    state.centerX = 0.5;
    state.centerY = 0.5;
    draw();
    writeHash();
  });

  const canvas = elements["map-canvas"];
  canvas.addEventListener("mousedown", beginDrag);
  window.addEventListener("mousemove", duringDrag);
  window.addEventListener("mouseup", endDrag);
  canvas.addEventListener("mouseleave", () => {
    elements.readout.textContent = "";
  });
  canvas.addEventListener("wheel", onWheel, { passive: false });
  window.addEventListener("resize", resizeCanvas);
  window.addEventListener("hashchange", () => {
    // React only to external hash edits, not our own replaceState.
  });
}

async function main() {
  selectElements();
  applyTheme();
  setStatus("Loading catalog…");
  try {
    const response = await fetch(DATASETS_URL);
    if (!response.ok) throw new Error(`Cannot load ${DATASETS_URL} (${response.status})`);
    datasetCatalog = (await response.json()).datasets;
  } catch (error) {
    setStatus(
      `${error.message}. Populate ./data/ with viewer pyramids and a datasets.json (see README).`,
      true,
    );
    return;
  }
  if (datasetCatalog.length === 0) {
    setStatus("No datasets in ./data/datasets.json.", true);
    return;
  }

  readHash();
  if (!state.datasetA || !datasetCatalog.some((entry) => entry.slug === state.datasetA)) {
    state.datasetA = datasetCatalog[0].slug;
  }
  await ensureStore(state.datasetA);
  const manifest = manifestFor(state.datasetA);
  if (!state.variable || !(state.variable in manifest.variables)) {
    state.variable = "sea_surface_height_above_geoid" in manifest.variables
      ? "sea_surface_height_above_geoid"
      : Object.keys(manifest.variables)[0];
  }
  if (!state.datasetB) state.datasetB = datasetCatalog.find((entry) => entry.slug !== state.datasetA)?.slug || null;

  populateSelect(
    elements["dataset-a"],
    datasetCatalog.map((entry) => ({ value: entry.slug, label: entry.label })),
    state.datasetA,
  );
  populateSelect(
    elements["dataset-b"],
    datasetCatalog.map((entry) => ({ value: entry.slug, label: entry.label })),
    state.datasetB,
  );
  refreshControlsForDatasets();
  wireControls();
  resizeCanvas();
  await render();
}

main();
