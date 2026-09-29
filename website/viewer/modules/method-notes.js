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
      "{params}Eddies = closed contours of full SSH after a Gaussian high-pass (Chelton et " +
      "al. 2011 family), not of a sea level anomaly. With two forecasts, same-polarity " +
      "centres within the larger eddy radius (at least 50 km) are paired nearest first. " +
      "Chance = the same pairing with F2 shifted 5° east and west, averaged. Agreement, " +
      "not accuracy. No census for 1° products: their eddies span a few grid cells.",
  },

  // Live power spectrum (rail-psd, live FFT over the map rectangle).
  psd: {
    title: "Live power spectrum",
    body:
      "How much variation the box holds at each size. Curve height is the energy at that " +
      "size; a forecast below the reference is smoothing that size away. A curve stops at " +
      "its own grid limit. " +
      "With GLORYS in one panel and a forecast in the other, the dashed curve is their " +
      "difference and the vertical line is the scale of disagreement with GLORYS: the size " +
      "at which the difference reaches half the GLORYS spectrum (the ratio of Ballarotta " +
      "et al. 2019, taken against a reanalysis instead of altimetry). It is measured on the " +
      "GLORYS grid, a finer field being block-averaged onto a coarser one. GLORYS is not " +
      "independent: the ML models are trained on it and forecasts start from analyses close " +
      "to it, so at short leads this mostly measures the initial state, not the model's " +
      "resolution. The chart below the spectrum shows it against lead. " +
      "Computed in the browser on each model's native cells, exploratory: Hann window, land " +
      "filled with the box mean (warned from 2% land), zero-padded, isotropic density " +
      "(units squared per cycle per km) up to the coarser axis's grid limit. Currents: " +
      "KE = 0.5 (PSD_u + PSD_v). Drag the box on the map, resize it by its handles. " +
      "Compare models only at sizes both resolve.",
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
  // Background sigma is a (latitude, longitude) pair of Gaussian sigmas in km, applied in
  // degrees: converted at 5° N (the global grids' mean latitude) on every domain, IBI included.
  // The high-pass 1 - exp(-k²σ²/2) reaches half power at kσ = 1.567, a wavelength of 4.01σ.
  ["background_sigma_km", "background sigma", backgroundSigmaLabel],
  ["contour_level_step_meters", "contour step", (v) => `${v} m`],
  ["min_contour_convexity", "min convexity", (v) => `${v}`],
  ["max_abs_latitude_degrees", "max abs latitude", (v) => `${v}°`],
  ["apply_contour_filtering", "contour filtering", (v) => (v ? "on" : "off")],
  ["oceanbench_version", "oceanbench version", (v) => `${v}`],
];

const HIGH_PASS_HALF_POWER_SIGMAS = 4.01;

function backgroundSigmaLabel(value) {
  const [latitudeKm, longitudeKm] = Array.isArray(value) ? value : [value, value];
  const degreeKm = (Math.PI * 6371) / 180;
  const latitude = Number(latitudeKm) / degreeKm;
  const longitude = Number(longitudeKm) / (degreeKm * Math.cos((5 * Math.PI) / 180));
  return (
    `sigma ${latitude.toFixed(2)}° × ${longitude.toFixed(2)}°, high-pass half power ` +
    `~${(HIGH_PASS_HALF_POWER_SIGMAS * latitude).toFixed(2)}° × ${(HIGH_PASS_HALF_POWER_SIGMAS * longitude).toFixed(2)}°`
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
