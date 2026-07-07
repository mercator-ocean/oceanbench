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
    const response = await fetch(resolvedUrl);
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

// Per-start-date RMSD series for a short variable at the requested lead day, or null.
export function yearRmsdSeries(rmsd, shortName, leadDay) {
  if (!rmsd || !rmsd.variables) return null;
  const variable = rmsd.variables[shortName];
  if (!variable || !variable.leads) return null;
  const entry = variable.leads[String(leadDay)];
  if (!entry || !Array.isArray(entry.dates)) return null;
  return { dates: entry.dates, rmsd: entry.rmsd || [], counts: entry.n || [], depthBin: variable.depth_bin };
}
