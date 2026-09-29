// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// "Transparent science" content map: every panel, chart and overlay legend that
// COMPUTES something carries a small "?" affordance whose popover text lives here.
// This is the single place scientists edit that copy, the popover engine
// (method-popover.js) only renders it. Bodies may contain `{placeholder}` tokens
// filled from the `dynamicFields` passed to attachMethodNote (see method-popover.js).
//
// Every note states plainly what is display-only vs. what feeds a metric, and where
// the real (offline) numbers come from, so nothing on the map is mistaken for an
// official score.

import { escapeHtml } from "./format.js";

export const METHOD_NOTES = {
  // Main map when showing a forecast field. {dataset} = the panel's dataset label.
  "field-map": {
    title: "Forecast field",
    body: "Fields from {dataset}, as compressed tiles. Display only; metrics use the raw outputs.",
  },

  // Difference display mode (Forecast 1 − Forecast 2).
  "diff-view": {
    title: "Difference view",
    body: "Forecast 1 minus Forecast 2 per pixel, at display resolution. Scale centered on zero.",
  },

  // Class-4 obs error overlay legend.
  "class4-legend": {
    title: "Class IV match-ups",
    body:
      "Each point is a real observation (altimetry SSH, in-situ T/S, drifter currents at 15 m), " +
      "colored by |obs − model| with the model interpolated to it. Low zoom draws a sample; " +
      "statistics use all points. SSH as sea level anomaly: GLO12 mean dynamic topography, " +
      "shift −0.1148 m (climatology −0.1329 m). QC is upstream; no outlier rejection here.",
  },

  // Skill vs lead day chart (rail-lead-curve).
  "lead-curve": {
    title: "RMSE vs lead day",
    body:
      "Official Class IV RMSE per lead day, pooled over the 52 starts of 2024. Band: 95% " +
      "bootstrap interval. Computed offline.",
  },

  // RMSE vs depth vertical profile chart (rail-depth-profile).
  "depth-profile": {
    title: "RMSE vs depth",
    body:
      "Class IV RMSE per depth bin at this lead, all 2024 match-ups, official method. " +
      "Temperature's top metre is its own surface bin. Hover for the obs count.",
  },

  // RMSE / bias by start date chart (rail-year-rmsd).
  "year-rmsd": {
    title: "RMSE / bias by start date",
    body:
      "Class IV RMSE per start date, pooled over its match-ups (official method). Bias mode: " +
      "pooled mean of model minus obs. Band: 95% per start (bootstrap for RMSE, mean ± 1.96 " +
      "sd/√n for bias); obs are spatially correlated, so read it as a lower bound. Click a " +
      "point to open that date.",
  },

  // Year error geography map / its colorbar.
  "year-geography": {
    title: "Year error geography",
    body:
      "Mean |obs − model| per cell over the 52 starts (model − obs in bias mode), on a 2° grid " +
      "(0.25° for IBI). Same obs and interpolation as the Class IV scores; SSH as sea level " +
      "anomaly (shift −0.1148 m, climatology −0.1329 m); no outlier rejection.",
  },

  // Eddies overlay legend. {params} is rendered as a live parameter list from the census
  // json; when absent, only the fixed text below shows.
  "eddies-legend": {
    title: "Eddy detection",
    body:
      "{params}Eddies are detected as closed sea surface height anomaly contours (Chelton " +
      "et al. 2011 family). With two forecasts, eddy centres of the same polarity are " +
      "matched within 200 km. This measures agreement between the forecasts, not accuracy " +
      "against observations. Detection runs on each dataset's own native grid, so counts " +
      "are not comparable between a 1-degree and a 1/12-degree dataset.",
  },

  // Live power spectrum (rail-psd, live FFT over the map rectangle).
  psd: {
    title: "Live power spectrum",
    body:
      "A spectrum measures how much variation the boxed region holds at each size, not " +
      "where that variation sits. The height of a curve is the energy at that size, so " +
      "the two panels can be read against each other size by size: where a forecast sits " +
      "below the reference it is smoothing that size away. Where a curve ends is its grid " +
      "limit: no model is drawn at sizes its own grid cannot carry, so a coarse model's " +
      "curve stops well before a fine one's. " +
      "With the GLORYS reanalysis in the other panel, the marked vertical line is " +
      "the effective resolution after Ballarotta et al. (2019, Ocean Science, " +
      "doi:10.5194/os-15-1091-2019), measured on the reference grid so it does not depend " +
      "on which panel holds the reference: the size below which the forecast disagrees " +
      "with the reference by more than half the signal. Below that size the forecast is " +
      "mostly wrong about where things are or when they happen, however much energy it " +
      "carries there. A model can match GLO12 energy for energy at 100 km and still have " +
      "an effective resolution of 500 km, because its eddies are in the wrong places. " +
      "Ballarotta et al. measure against independent along-track altimetry; taking a model " +
      "reanalysis as the reference is an adaptation of their method. GLORYS is published " +
      "here on a 1° grid, so against it an effective resolution below about 220 km (two " +
      "grid cells) cannot be measured. GLO12 is listed as a forecast, not a reference: its " +
      "lead 1 is its nowcast. " +
      "Computed in the browser on the model's finest published grid, averaged in blocks " +
      "onto the square grid the transform needs. Exploratory. Hann window, land filled " +
      "with the region mean, then the two-dimensional spectrum is SUMMED over wavenumber " +
      "rings and divided by the ring width, so the curve is an isotropic spectral density " +
      "(field units squared per cycle per km) whose integral over wavenumber is the " +
      "variance of the box. For currents the curve is the kinetic energy spectrum " +
      "KE(k) = 0.5 (PSD_u + PSD_v), not the spectrum of the speed magnitude. The box size " +
      "is capped so the estimate stays reliable. In compare mode both forecasts share one " +
      "box; models with very different resolution cannot share a fair one. Near its grid " +
      "scale every model is damped by its own dissipation, so compare models only at " +
      "sizes both resolve.",
  },

  // Trajectories overlay.
  trajectories: {
    title: "Illustrative trajectories",
    body:
      "Click to seed particles advected in each forecast's displayed currents (RK2, 6 h steps). " +
      "Illustrative only, not scored. The official Lagrangian metric runs offline: 10,000 seeds " +
      "globally (IBI scaled by ocean area, min 2,000; OceanParcels RK4) in the forecast and in " +
      "each reference (GLORYS, GLO12), " +
      "mean separation scored at leads 2 to N−1 only. It measures agreement between models, not " +
      "with drifters.",
  },

  // Water-column profile (profile-on-click) chart.
  "column-profile": {
    title: "Water column",
    body:
      "Model T and S near the clicked point, at this start and lead, from a 1° 16-bit copy of " +
      "the forecast (not its native grid). Display only.",
  },

  // Currents particle animation overlay.
  currents: {
    title: "Current animation",
    body:
      "Animated particles following the displayed current field. Decorative; no metric " +
      "is derived from it.",
  },

  // Data-provenance line at the bottom of the context rail.
  "data-provenance": {
    title: "Data provenance",
    body:
      "Pipeline version and date of this dataset's artifacts, generated together and " +
      "reconciled against the raw match-ups.",
  },
};

// Order in which the eddy-census parameters render, with human labels and a formatter.
// Keys are the exact snake_case fields of the census json's `parameters` block.
const EDDY_PARAMETER_ROWS = [
  ["amplitude_threshold_meters", "amplitude above outermost closed contour", (v) => `\u2265 ${v} m`],
  ["min_eddy_area_km2", "min area", (v) => `${Number(v).toLocaleString("en-US")} km²`],
  ["max_eddy_area_km2", "max area", (v) => `${Number(v).toLocaleString("en-US")} km²`],
  ["min_peak_separation_km", "min peak separation", (v) => `${v} km`],
  ["max_match_distance_km", "max match distance", (v) => `${v} km`],
  // Background sigma is a (latitude, longitude) pair of Gaussian sigmas in km, applied in
  // degrees: converted at 5° N (the global grids' mean latitude) on every domain, IBI included.
  // Half power sits near 7.5 sigma, compared with Chelton et al. (2011)'s 10° x 20° block.
  ["background_sigma_km", "background sigma", backgroundSigmaLabel],
  ["contour_level_step_meters", "contour step", (v) => `${v} m`],
  ["min_contour_convexity", "min convexity", (v) => `${v}`],
  ["max_abs_latitude_degrees", "max abs latitude", (v) => `${v}°`],
  ["apply_contour_filtering", "contour filtering", (v) => (v ? "on" : "off")],
  ["oceanbench_version", "oceanbench version", (v) => `${v}`],
];

function backgroundSigmaLabel(value) {
  const [latitudeKm, longitudeKm] = Array.isArray(value) ? value : [value, value];
  const degreeKm = (Math.PI * 6371) / 180;
  const latitude = Number(latitudeKm) / degreeKm;
  const longitude = Number(longitudeKm) / (degreeKm * Math.cos((5 * Math.PI) / 180));
  return (
    `sigma ${latitude.toFixed(2)}° lat × ${longitude.toFixed(2)}° lon, half power about ` +
    `${(7.5 * latitude).toFixed(0)}° × ${(7.5 * longitude).toFixed(0)}° (Chelton et al. 2011: 10° × 20°)`
  );
}

// Render the live eddy census `parameters` block into an HTML fragment (a small
// definition list) to substitute for the {params} token. Returns "" if absent.
export function renderEddyParameters(parameters) {
  if (!parameters || typeof parameters !== "object") return "";
  const rows = EDDY_PARAMETER_ROWS.filter(([key]) => parameters[key] != null).map(
    ([key, label, format]) =>
      `<div class="method-param"><span>${label}</span><strong>${escapeHtml(String(format(parameters[key])))}</strong></div>`,
  );
  if (!rows.length) return "";
  return `<div class="method-params">${rows.join("")}</div>`;
}
