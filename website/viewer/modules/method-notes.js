// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// "Transparent science" content map: every panel, chart and overlay legend that
// COMPUTES something carries a small "?" affordance whose popover text lives here.
// This is the single place scientists edit that copy — the popover engine
// (method-popover.js) only renders it. Bodies may contain `{placeholder}` tokens
// filled from the `dynamicFields` passed to attachMethodNote (see method-popover.js).
//
// Every note states plainly what is display-only vs. what feeds a metric, and where
// the real (offline) numbers come from, so nothing on the map is mistaken for an
// official score.

export const METHOD_NOTES = {
  // Main map when showing a forecast field. {dataset} = the panel's dataset label.
  "field-map": {
    title: "Forecast field",
    body:
      "Forecast fields from {dataset}, streamed as compressed image tiles. Display only. " +
      "All metrics are computed offline from the raw model outputs.",
  },

  // Difference display mode (Forecast 1 − Forecast 2).
  "diff-view": {
    title: "Difference view",
    body:
      "Forecast 1 minus Forecast 2, computed per pixel in the browser at display " +
      "resolution. The color scale is centered at zero.",
  },

  // Class-4 obs error overlay legend.
  "class4-legend": {
    title: "Class-4 match-ups",
    body:
      "Each point is a real observation (altimetry SSH, Argo T/S profiles, drifter " +
      "currents at 15 m) colored by |obs − model| for the selected start and lead. At " +
      "low zoom only a sample of points is drawn; statistics always use all of them. The " +
      "model is interpolated to each observation location. SSH is compared as sea level " +
      "anomaly (GLO12 mean dynamic topography, datum shift −0.1148 m). Quality control " +
      "comes from the upstream data products; OceanBench rejects no outliers.",
  },

  // Skill vs lead day chart (rail-lead-curve).
  "lead-curve": {
    title: "RMSD vs lead day",
    body:
      "Official Class-4 RMSD against observations for each lead day, pooled over the 52 " +
      "start dates of 2024. The shaded band is a 95% bootstrap confidence interval. " +
      "Computed offline by the oceanbench library.",
  },

  // RMSD vs depth vertical profile chart (rail-depth-profile).
  "depth-profile": {
    title: "RMSD vs depth",
    body:
      "Class-4 RMSD against observations per depth bin, pooled over all match-ups of the " +
      "year at the selected lead day. Same observation set and method as the official " +
      "scores. Hover a point for the observation count.",
  },

  // RMSD / bias by start date chart (rail-year-rmsd).
  "year-rmsd": {
    title: "RMSD / bias by start date",
    body:
      "Class-4 RMSD for each start date, pooled over all match-ups of that start, same " +
      "method as the official scores. Bias mode shows the pooled mean of model minus obs. " +
      "The shaded band is a 95% bootstrap confidence interval. Click a point to open that " +
      "start date.",
  },

  // Year error geography map / its colorbar.
  "year-geography": {
    title: "Year error geography",
    body:
      "Mean |obs − model| per grid cell over all 52 start dates (signed model − obs in " +
      "bias mode), computed from the full match-up set on a 2° global grid (0.25° for " +
      "IBI). Same observation set and interpolation as the Class-4 scores. SSH is " +
      "compared as sea level anomaly (GLO12 mean dynamic topography, datum shift −0.1148 " +
      "m). No outlier rejection.",
  },

  // Eddies overlay legend. {params} is rendered as a live parameter list from the census
  // json; when absent, only the fixed text below shows.
  "eddies-legend": {
    title: "Eddy detection",
    body:
      "{params}Eddies are detected as closed sea surface height anomaly contours (Chelton " +
      "et al. 2011 family). With two forecasts, eddy centres of the same polarity are " +
      "matched within 200 km. This measures agreement between the forecasts, not accuracy " +
      "against observations.",
  },

  // Live power spectrum (rail-psd, live FFT over the map rectangle).
  psd: {
    title: "Live power spectrum",
    body:
      "Power spectrum of the boxed region, computed in the browser on the model's finest " +
      "published grid. Exploratory; the official spectra are on the scores page. Hann " +
      "window, land filled with the region mean, radially averaged, normalized so models " +
      "of different resolution are comparable. The box size is capped so the estimate " +
      "stays reliable. In compare mode both forecasts share one box; models with very " +
      "different resolution cannot share a fair one. Near its grid scale every model is " +
      "damped by its own dissipation, so compare models only at scales both resolve.",
  },

  // Trajectories overlay.
  trajectories: {
    title: "Illustrative trajectories",
    body:
      "Particles seeded by clicking are advected in each forecast's displayed current " +
      "field (RK2, 6-hour steps, on the model's finest published grid). Illustrative only; " +
      "nothing here feeds a score. The official Lagrangian metric is computed offline: " +
      "10,000 seeds advected hourly (OceanParcels RK4) in the forecast and in the GLO12 " +
      "reference, scored as their mean separation per lead day. It measures agreement " +
      "between models; drifter observations only enter the Class-4 currents diagnostic.",
  },

  // Currents particle animation overlay.
  currents: {
    title: "Current animation",
    body:
      "Animated particles following the displayed current field. Decorative; no metric " +
      "is derived from it.",
  },
};

// Order in which the eddy-census parameters render, with human labels and a formatter.
// Keys are the exact snake_case fields of the census json's `parameters` block.
const EDDY_PARAMETER_ROWS = [
  ["amplitude_threshold_meters", "amplitude threshold", (v) => `${v} m`],
  ["min_eddy_area_km2", "min area", (v) => `${Number(v).toLocaleString("en-US")} km²`],
  ["max_eddy_area_km2", "max area", (v) => `${Number(v).toLocaleString("en-US")} km²`],
  ["min_peak_separation_km", "min peak separation", (v) => `${v} km`],
  ["max_match_distance_km", "max match distance", (v) => `${v} km`],
  ["background_sigma_km", "background sigma", (v) => `${Number(v).toFixed(0)} km`],
  ["contour_level_step_meters", "contour step", (v) => `${v} m`],
  ["min_contour_convexity", "min convexity", (v) => `${v}`],
  ["max_abs_latitude_degrees", "max abs latitude", (v) => `${v}°`],
  ["apply_contour_filtering", "contour filtering", (v) => (v ? "on" : "off")],
  ["oceanbench_version", "oceanbench version", (v) => `${v}`],
];

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

function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character]),
  );
}
