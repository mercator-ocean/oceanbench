// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// The closed sets of view-mode names, in one place.
//
// These strings are compared in dozens of places across the viewer, written into the URL
// hash, and read a second time in a different language: styles.css selects on
// :root[data-scope="year"], .panel-grid[data-display="swipe"] and
// .panel-grid[data-display="diff"], and index.html carries the same values in the
// data-scope / data-display / data-metric attributes of the switch buttons. So adding a
// mode is never a one-file change, and before this module there was no list to consult.
//
// INVARIANT: the value of every constant here is also a CSS attribute value and a URL
// hash token. Renaming one silently breaks old shared links and the stylesheet, neither
// of which any test would notice. Add modes; do not rename them.

export const SCOPE_SINGLE_DATE = "single";
export const SCOPE_WHOLE_YEAR = "year";
export const SCOPES = Object.freeze([SCOPE_SINGLE_DATE, SCOPE_WHOLE_YEAR]);

export const DISPLAY_SIDE_BY_SIDE = "side";
export const DISPLAY_SWIPE = "swipe";
export const DISPLAY_DIFFERENCE = "diff";
export const DISPLAY_MODES = Object.freeze([DISPLAY_SIDE_BY_SIDE, DISPLAY_SWIPE, DISPLAY_DIFFERENCE]);

export const OVERLAY_NONE = "none";
export const OVERLAY_CLASS4 = "class4";
export const OVERLAY_EDDIES = "eddies";
export const OVERLAY_TRAJECTORIES = "trajectories";
export const OVERLAY_WATER_COLUMN = "column";
export const OVERLAY_MODES = Object.freeze([
  OVERLAY_NONE,
  OVERLAY_CLASS4,
  OVERLAY_EDDIES,
  OVERLAY_TRAJECTORIES,
  OVERLAY_WATER_COLUMN,
]);

export const YEAR_METRIC_ABSOLUTE_ERROR = "error";
export const YEAR_METRIC_BIAS = "bias";
export const YEAR_METRICS = Object.freeze([YEAR_METRIC_ABSOLUTE_ERROR, YEAR_METRIC_BIAS]);

export const REGION_GLOBAL = "global";
export const REGION_IBI = "ibi";
export const REGIONS = Object.freeze([REGION_GLOBAL, REGION_IBI]);

export const EDDY_REFERENCE_GLORYS = "glorys";
export const EDDY_REFERENCE_GLO12 = "glo12";
export const EDDY_REFERENCES = Object.freeze([EDDY_REFERENCE_GLORYS, EDDY_REFERENCE_GLO12]);

export const THEME_LIGHT = "light";
export const THEME_DARK = "dark";
export const THEMES = Object.freeze([THEME_LIGHT, THEME_DARK]);
