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
      "Forecast fields from {dataset}, streamed from OceanBench viewer pyramids " +
      "(uint16 scale/offset per tile, DEFLATE). Display only — all metrics are computed " +
      "offline from the raw model outputs.",
  },

  // Difference display mode (Forecast 1 − Forecast 2).
  "diff-view": {
    title: "Difference view",
    body:
      "Forecast 1 − Forecast 2, computed per pixel in the browser at display resolution. " +
      "Diverging scale centered at 0.",
  },

  // Class-4 obs error overlay legend.
  "class4-legend": {
    title: "Class-4 match-ups",
    body:
      "Match-ups: model values at real observation locations (altimetry SSH, Argo T/S " +
      "profiles, drifter u/v at 15 m). Points show |obs − model| for the selected " +
      "start/lead. Full observation set for the selected pair (display thinning at low " +
      "zoom; statistics always use all points). Observations are the pre-QC'd 2024 set " +
      "(project-oceanbench/public/observations2024/*.zarr); the model is interpolated to " +
      "each obs by bilinear horizontal plus vertical interpolation. SSH is scored as SLA " +
      "(GLO12 MDT + datum shift −0.1148). No outlier rejection is applied anywhere.",
  },

  // Skill vs lead day chart (rail-lead-curve).
  "lead-curve": {
    title: "RMSD vs lead day",
    body:
      "Official Class-4 RMSD vs observations per lead day, aggregated over 52 start dates " +
      "(2024). Shaded band: 95% CI from 1000-iteration bootstrap over start dates. " +
      "Computed offline by the oceanbench library.",
  },

  // RMSD / bias by start date chart (rail-year-rmsd).
  "year-rmsd": {
    title: "RMSD / bias by start date",
    body:
      "Class-4 RMSD per start date, same method as the official scores (pooled over all " +
      "match-ups for that start). In bias mode: pooled mean(model − obs) per start date. " +
      "Shaded band: 95% CI, bootstrap per start date. " +
      "Click a point to open that start date.",
  },

  // Year error geography map / its colorbar.
  "year-geography": {
    title: "Year error geography",
    body:
      "Time-mean |obs − model| (or signed model − obs in bias mode) per grid cell over " +
      "all 52 start dates, from the full Class-4 match-up set. Global grid 2°, IBI 0.25°. " +
      "In bias mode the hover shows ±1 SE. " +
      "Observations are the pre-QC'd 2024 set " +
      "(project-oceanbench/public/observations2024/*.zarr); the model is interpolated to " +
      "each obs by bilinear horizontal plus vertical interpolation. SSH is scored as SLA " +
      "(GLO12 MDT + datum shift −0.1148). No outlier rejection is applied anywhere.",
  },

  // Eddies overlay legend. {params} is rendered as a live parameter list from the census
  // json; when absent, only the fixed text below shows.
  "eddies-legend": {
    title: "Eddy detection",
    body:
      "{params}Detection: closed SSH-anomaly contours (Chelton et al. 2011-family method). " +
      "Two forecasts: greedy nearest-neighbour matching by eddy centres, same polarity, " +
      "≤200 km — agreement between forecasts, not ground truth.",
  },

  // Live power spectrum (rail-psd, live FFT over the map rectangle).
  psd: {
    title: "Live power spectrum",
    body:
      "Spectrum of the boxed region on the map, computed in-browser at the model's native " +
      "(finest published) grid: Hann window, land mean-filled, radially averaged FFT. The " +
      "Mean-filled land can damp coastal spectra, so open-ocean boxes are preferable. " +
      "box size is capped so the spectrum always reflects native resolution — wavelengths " +
      "from 2× the grid spacing up to the box size. In compare mode one shared box drives " +
      "both forecasts; a coarser model's curve simply stops at its own resolution limit. " +
      "Exploratory diagnostic — official spectral metrics live on the scores page.",
  },

  // Trajectories overlay.
  trajectories: {
    title: "Illustrative trajectories",
    body:
      "Illustrative RK2 advection (6-hour steps) of 20 click-seeded particles through each " +
      "forecast's current fields at the displayed depth (surface or 15 m), always on the " +
      "model's native (finest published) grid regardless of zoom. The official " +
      "Lagrangian score is different: OceanParcels RK4 (hourly), 10,000 area-weighted seeds, " +
      "advected in surface currents of the forecast and of the GLORYS/GLO12 reference from " +
      "identical seeds — a model-vs-model transport-agreement metric (mean separation in km " +
      "per lead day). No observation-based Lagrangian metric exists yet; real drifter " +
      "observations appear only in the Class-4 currents diagnostic (15 m). Nothing computed " +
      "here feeds any score.",
  },

  // Currents particle animation overlay.
  currents: {
    title: "Current animation",
    body:
      "Illustrative particle animation of the displayed current field. Decorative " +
      "visualization; no metric is derived from it.",
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
