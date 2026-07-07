// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Loaders and helpers for the "Entire year" scope of the viewer. Two artifacts,
// one per dataset/region, referenced from insights.json:
//   year_error_geography → a precomputed raster of the time-mean |obs − model|
//     over every start date, per raw variable and lead day; rendered as a map layer.
//   year_rmsd_by_start → a per-start-date RMSD series per raw variable/lead, drawn
//     as a line chart whose points drill down into the single-forecast scope.
// Both are fetched lazily and memoised by URL.

import { resolveViewerDataUrl } from "../config.js";

const jsonCache = new Map();

function fetchYearJSON(url) {
  const resolvedUrl = resolveViewerDataUrl(url);
  if (jsonCache.has(resolvedUrl)) return jsonCache.get(resolvedUrl);
  const promise = (async () => {
    const response = await fetch(resolvedUrl, { cache: "no-cache" });
    if (!response.ok) throw new Error(`${resolvedUrl} -> HTTP ${response.status}`);
    return response.json();
  })();
  jsonCache.set(resolvedUrl, promise);
  return promise;
}

export async function loadYearGeography(url) {
  if (!url) return null;
  return fetchYearJSON(url).catch(() => null);
}

export async function loadYearRmsd(url) {
  if (!url) return null;
  return fetchYearJSON(url).catch(() => null);
}

// Map a viewer variable key (a manifest standard_name, or a derived currents key)
// onto the short name used in the year artifacts (SSH, T, S, u, v). Derived current
// speed maps to the eastward component, captioned honestly by the caller.
const VARIABLE_TO_YEAR = {
  sea_surface_height_above_geoid: { short: "SSH", unit: "m" },
  sea_water_potential_temperature: { short: "T", unit: "°C" },
  sea_water_salinity: { short: "S", unit: "PSU" },
  eastward_sea_water_velocity: { short: "u", unit: "m/s" },
  eastward_sea_water_velocity_15m: { short: "u", unit: "m/s" },
  northward_sea_water_velocity: { short: "v", unit: "m/s" },
  northward_sea_water_velocity_15m: { short: "v", unit: "m/s" },
};

export function yearVariableMapping(variableKey) {
  if (variableKey === "current_speed" || variableKey === "current_speed_15m") {
    return { short: "u", unit: "m/s", component: "eastward velocity (u)" };
  }
  const entry = VARIABLE_TO_YEAR[variableKey];
  if (!entry) return null;
  return { ...entry, component: null };
}

// Build a renderable field ({data, width, height} + lat/lon axes) from the year
// error-geography artifact for a short variable name at the requested lead day.
// null cells (no observation) become NaN. Returns null when the variable/lead is
// absent from the artifact.
export function buildYearGeographyField(geography, shortName, leadDay) {
  if (!geography || !geography.grid || !geography.variables) return null;
  const variable = geography.variables[shortName];
  if (!variable || !variable.leads) return null;
  const flat = variable.leads[String(leadDay)];
  if (!Array.isArray(flat)) return null;
  const { lat0, dlat, nlat, lon0, dlon, nlon } = geography.grid;
  if (flat.length !== nlat * nlon) return null;
  const data = new Float32Array(flat.length);
  for (let i = 0; i < flat.length; i += 1) {
    const value = flat[i];
    data[i] = value == null ? NaN : value;
  }
  const latitudes = Array.from({ length: nlat }, (_, i) => lat0 + i * dlat);
  const longitudes = Array.from({ length: nlon }, (_, j) => lon0 + j * dlon);
  return { field: { data, width: nlon, height: nlat }, latitudes, longitudes };
}

function leadArray(record, leadDay) {
  return record && Array.isArray(record[String(leadDay)]) ? record[String(leadDay)] : null;
}

function yearCountArray(variable, leadDay, { bias = false } = {}) {
  if (!variable) return null;
  // Current artifacts use a parallel per-lead count array for the |error| raster.
  // Accept a few explicit names so older/newer publishes degrade without breaking
  // hover readouts. Bias reuses the same sampling when it has no dedicated count.
  const candidates = bias
    ? [variable.bias_n, variable.bias_counts, variable.bias_count, variable.n, variable.counts, variable.count, variable.leads_n, variable.leads_counts]
    : [variable.n, variable.counts, variable.count, variable.leads_n, variable.leads_counts];
  for (const candidate of candidates) {
    const flat = leadArray(candidate, leadDay);
    if (flat) return flat;
  }
  return null;
}

export function buildYearObservationCounts(geography, shortName, leadDay, options = {}) {
  if (!geography || !geography.grid || !geography.variables) return null;
  const variable = geography.variables[shortName];
  const flat = yearCountArray(variable, leadDay, options);
  if (!flat) return null;
  const { nlat, nlon } = geography.grid;
  if (flat.length !== nlat * nlon) return null;
  const data = new Uint32Array(flat.length);
  for (let i = 0; i < flat.length; i += 1) {
    const value = Number(flat[i]);
    data[i] = Number.isFinite(value) && value > 0 ? Math.round(value) : 0;
  }
  return { data, width: nlon, height: nlat };
}

// Robust upper bound (98th percentile) of finite |obs − model| across the requested
// lead for a short variable, used to build one shared color scale across panels
// showing the same variable. A percentile rather than the raw max keeps a handful of
// coastal/polar outlier cells from washing out the whole map.
export function yearGeographyMax(geography, shortName, leadDay) {
  const built = buildYearGeographyField(geography, shortName, leadDay);
  if (!built) return 0;
  const finite = [];
  for (const value of built.field.data) if (Number.isFinite(value)) finite.push(value);
  if (!finite.length) return 0;
  finite.sort((a, b) => a - b);
  const index = Math.min(finite.length - 1, Math.floor(finite.length * 0.98));
  return finite[index];
}

// Build a renderable signed-bias field (mean(model − obs) per cell) from the year
// error-geography artifact, mirroring buildYearGeographyField but reading the parallel
// `bias` structure the data agent appends. Returns null when the bias field is absent,
// so callers can fall back to the |error| path or a "not available" note.
export function buildYearBiasField(geography, shortName, leadDay) {
  if (!geography || !geography.grid || !geography.variables) return null;
  const variable = geography.variables[shortName];
  if (!variable || !variable.bias) return null;
  const flat = variable.bias[String(leadDay)];
  if (!Array.isArray(flat)) return null;
  const { lat0, dlat, nlat, lon0, dlon, nlon } = geography.grid;
  if (flat.length !== nlat * nlon) return null;
  const data = new Float32Array(flat.length);
  for (let i = 0; i < flat.length; i += 1) {
    const value = flat[i];
    data[i] = value == null ? NaN : value;
  }
  const latitudes = Array.from({ length: nlat }, (_, i) => lat0 + i * dlat);
  const longitudes = Array.from({ length: nlon }, (_, j) => lon0 + j * dlon);
  return { field: { data, width: nlon, height: nlat }, latitudes, longitudes };
}

// Symmetric upper bound for the diverging bias scale: the 98th percentile of the finite
// |bias| values, so a balanced [-M, +M] range keeps the diverging colormap centred on 0
// while a handful of extreme cells do not wash out the map. Returns 0 when bias absent.
export function yearBiasMax(geography, shortName, leadDay) {
  const built = buildYearBiasField(geography, shortName, leadDay);
  if (!built) return 0;
  const finite = [];
  for (const value of built.field.data) if (Number.isFinite(value)) finite.push(Math.abs(value));
  if (!finite.length) return 0;
  finite.sort((a, b) => a - b);
  const index = Math.min(finite.length - 1, Math.floor(finite.length * 0.98));
  return finite[index];
}

// Lead-independent y-bound for the RMSD/bias-by-start rail chart: the max finite value
// (max |value| when signed) across EVERY lead's series of a variable. Scrubbing the
// lead slider then moves the curve within a constant frame instead of rescaling the
// axis on each lead. Pure function of the loaded artifact — it only changes when the
// dataset/variable/region (or metric) changes, never with the selected lead.
export function yearRmsdSeriesMax(rmsd, shortName, { signed = false } = {}) {
  if (!rmsd || !rmsd.variables) return 0;
  const variable = rmsd.variables[shortName];
  if (!variable || !variable.leads) return 0;
  let maximum = 0;
  const consider = (values) => {
    if (!Array.isArray(values)) return;
    for (const value of values) {
      if (Number.isFinite(value) && Math.abs(value) > maximum) maximum = Math.abs(value);
    }
  };
  for (const entry of Object.values(variable.leads)) {
    if (!entry) continue;
    // Include the CI band edges so the fixed axis never clips the shaded interval.
    if (signed) {
      consider(entry.bias);
      consider(entry.bias_ci_low);
      consider(entry.bias_ci_high);
    } else {
      consider(entry.rmsd);
      consider(entry.rmsd_ci_high);
    }
  }
  return maximum;
}

// Per-start-date RMSD series for a short variable at the requested lead day, or null.
// When the artifact carries a parallel `bias` array (area-weighted mean(model − obs) per
// start), it is returned too; otherwise `bias` is null and the caller stays in |error|.
export function yearRmsdSeries(rmsd, shortName, leadDay) {
  if (!rmsd || !rmsd.variables) return null;
  const variable = rmsd.variables[shortName];
  if (!variable || !variable.leads) return null;
  const entry = variable.leads[String(leadDay)];
  if (!entry || !Array.isArray(entry.dates)) return null;
  const ciLow = Array.isArray(entry.rmsd_ci_low) ? entry.rmsd_ci_low : null;
  const ciHigh = Array.isArray(entry.rmsd_ci_high) ? entry.rmsd_ci_high : null;
  const biasCiLow = Array.isArray(entry.bias_ci_low) ? entry.bias_ci_low : null;
  const biasCiHigh = Array.isArray(entry.bias_ci_high) ? entry.bias_ci_high : null;
  return {
    dates: entry.dates,
    rmsd: entry.rmsd || [],
    bias: Array.isArray(entry.bias) ? entry.bias : null,
    counts: entry.n || [],
    // Parallel 95% CI arrays when the artifact carries them (old artifacts omit them → null,
    // and the chart degrades to a plain line with no band).
    ciLow,
    ciHigh,
    biasCiLow,
    biasCiHigh,
    depthBin: variable.depth_bin,
  };
}

// Per-cell standard error of the bias (std(model − obs)/sqrt(n)) for a short variable at a lead,
// or null when the artifact predates the bias_se array. Parallel to buildYearObservationCounts so
// the hover readout can annotate a bias cell with "± SE".
export function buildYearBiasStandardError(geography, shortName, leadDay) {
  if (!geography || !geography.grid || !geography.variables) return null;
  const variable = geography.variables[shortName];
  if (!variable || !variable.bias_se) return null;
  const flat = leadArray(variable.bias_se, leadDay);
  if (!flat) return null;
  const { nlat, nlon } = geography.grid;
  if (flat.length !== nlat * nlon) return null;
  const data = new Float32Array(flat.length);
  for (let i = 0; i < flat.length; i += 1) {
    const value = Number(flat[i]);
    data[i] = Number.isFinite(value) ? value : NaN;
  }
  return { data, width: nlon, height: nlat };
}
