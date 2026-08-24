// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// The view state every panel is linked to: the shared viewport (`view`) and everything
// the two panels agree on (`shared`). Both are mutable by design, since the render path
// reads them on every frame and copying them per frame would buy nothing.
//
// The fields whose values come from a closed vocabulary are written through the setters
// below rather than assigned directly. Two of the writers take values from outside the
// program (the URL hash and a <select> element), and an unchecked assignment there puts a
// value in the state that no CSS rule and no branch matches, which shows up as a blank
// map rather than as an error. The setters drop what they do not recognise and report
// whether anything changed, so a caller can skip the re-render.

import {
  DISPLAY_MODES,
  DISPLAY_SIDE_BY_SIDE,
  EDDY_REFERENCES,
  EDDY_REFERENCE_GLORYS,
  OVERLAY_MODES,
  OVERLAY_NONE,
  REGIONS,
  REGION_GLOBAL,
  SCOPES,
  SCOPE_SINGLE_DATE,
  THEMES,
  THEME_LIGHT,
  YEAR_METRICS,
  YEAR_METRIC_ABSOLUTE_ERROR,
} from "./view-modes.js";

// The global default view is centred on the prime meridian (centerNX = (lon + 180) / 360,
// so nx 0.5 ≡ lon 0°). A Pacific-centred default put the dateline down the middle of the
// first frame, which splits the Atlantic across both edges of the map and opens the viewer
// on a seam. Rendering already tiles wrapped longitude copies, so panning to the Pacific
// costs nothing.
export const GLOBAL_DEFAULT_CENTER_NX = 0.5;

// Shared state, linked across every panel (contracts.md §6).
export const view = { zoom: 1, centerNX: GLOBAL_DEFAULT_CENTER_NX, centerNY: 0.5 };
export const DEFAULT_LAYOUT = { controlsWidth: 256, railWidth: 352 };
const savedLayout = JSON.parse(localStorage.getItem("oceanbench.viewer.layout") || "null") || {};
export const shared = {
  startIndex: 0,
  leadDay: 1,
  theme: THEME_LIGHT,
  layout: 1,
  // "single" = per-start-date fields (the default view); "year" = precomputed
  // whole-year error-geography raster + RMSE-by-start diagnostics.
  scope: SCOPE_SINGLE_DATE,
  // Year-scope map metric: "error" = time-mean |obs − model| (sequential), "bias" =
  // time-mean signed model − obs (diverging, centred 0). Single-forecast scope ignores it.
  yearMetric: YEAR_METRIC_ABSOLUTE_ERROR,
  // PSD rectangle tool: { lon, lat, w, h } in degrees (centre + size). Disabled by
  // default so the initial map stays clean; enabling creates a centred default box.
  psdEnabled: false,
  psdBox: null,
  // The size the user asked for, kept apart from the clamped size actually drawn: a
  // second forecast can raise the minimum box size, and the box must shrink back to the
  // requested size once that constraint goes away. { w, h } in degrees; null when unset.
  psdBoxRequest: null,
  overlayMode: OVERLAY_NONE,
  // Water-column click point { lon, lat } (profile-on-click mode); null when unset. Kept
  // in the hash so a link reproduces the clicked profile.
  columnPoint: null,
  region: REGION_GLOBAL,
  eddyReference: EDDY_REFERENCE_GLORYS,
  showParticles: true,
  particleSpeed: 1,
  railCollapsed: localStorage.getItem("oceanbench.viewer.railCollapsed") === "1",
  controlsCollapsed: localStorage.getItem("oceanbench.viewer.controlsCollapsed") === "1",
  controlsWidth: Number(savedLayout.controlsWidth) || DEFAULT_LAYOUT.controlsWidth,
  railWidth: Number(savedLayout.railWidth) || Number(localStorage.getItem("oceanbench.viewer.railWidth")) || DEFAULT_LAYOUT.railWidth,
  // 2-forecast display: "side" (two panels) or "swipe" (one map, F1 left / F2 right).
  displayMode: DISPLAY_SIDE_BY_SIDE,
  // Which forecast the rail shows when 2 forecasts carry different variables.
  railForecast: 0,
};

function assign(field, allowed, value) {
  if (!allowed.includes(value)) return false;
  if (shared[field] === value) return false;
  shared[field] = value;
  return true;
}

export function setSharedScope(value) {
  return assign("scope", SCOPES, value);
}

export function setSharedDisplayMode(value) {
  return assign("displayMode", DISPLAY_MODES, value);
}

export function setSharedOverlayMode(value) {
  return assign("overlayMode", OVERLAY_MODES, value);
}

export function setSharedYearMetric(value) {
  return assign("yearMetric", YEAR_METRICS, value);
}

export function setSharedRegion(value) {
  return assign("region", REGIONS, value);
}

export function setSharedEddyReference(value) {
  return assign("eddyReference", EDDY_REFERENCES, value);
}

export function setSharedTheme(value) {
  return assign("theme", THEMES, value);
}
