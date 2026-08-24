// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// CROSS-PAGE CONTRACT: website/scores-summary.js imports the named exports below, so this
// is the one file under website/viewer/modules/ that another page depends on; changing or
// removing an export breaks the scores page, which no viewer probe loads. The other half
// of the contract is ../config.js. Both stay inside website/viewer/ because the directory
// is published on its own as well as through the Quarto site, and a module that reaches
// outside it would 404 in that deployment.
//
// Data layer for the scores page: it loads the published score summary through the
// viewer's own insight index (so the data-root override precedence in config.js is the
// only one that exists) and turns the flat row list into the selections the page shows.
//
// The published rows are already lead-time resolved: one row per challenger, region,
// variable, depth, metric and lead day, carrying the mean, its bootstrap confidence
// interval and the skill against the 1 degree persistence baseline. Nothing is
// recomputed here, so the page never touches scores.parquet.

import { loadInsightIndex, loadScoresSummary } from "./insights.js";
import { resolveViewerDataUrl } from "../config.js";

const VARIABLE_LABELS = {
  sea_surface_height_above_geoid: "sea surface height",
  sea_water_potential_temperature: "temperature",
  sea_water_salinity: "salinity",
  eastward_sea_water_velocity: "zonal current",
  northward_sea_water_velocity: "meridional current",
  geostrophic_eastward_sea_water_velocity: "zonal geostrophic current",
  geostrophic_northward_sea_water_velocity: "meridional geostrophic current",
  ocean_mixed_layer_thickness: "mixed layer depth",
};

// Variables read left to right in the order a forecast user asks about them, not
// alphabetically: sea level first, then the thermohaline pair, then currents.
const VARIABLE_ORDER = [
  "sea_surface_height_above_geoid",
  "sea_water_potential_temperature",
  "sea_water_salinity",
  "eastward_sea_water_velocity",
  "northward_sea_water_velocity",
  "geostrophic_eastward_sea_water_velocity",
  "geostrophic_northward_sea_water_velocity",
  "ocean_mixed_layer_thickness",
];

const REFERENCE_LABELS = {
  glorys: "GLORYS reanalysis",
  glo12: "GLO12 analysis",
  observations: "observations (Class 4)",
};

const REGION_LABELS = { global: "Global", ibi: "IBI" };

const METRIC_LABELS = { rmsd: "RMSE", class4_rmsd: "Class 4 RMSE" };

const DEPTH_ORDER = [
  "surface",
  "0-5m",
  "15m",
  "5-100m",
  "50m",
  "100m",
  "100-300m",
  "200m",
  "300m",
  "300-600m",
  "500m",
];

// Fallback display names for the published slugs. The optional challengers.json beside
// the data overrides these when it is published; it is absent from the current preview
// prefix, so the page must read correctly without it.
const CHALLENGER_LABELS = {
  glonet: "GLONET",
  xihe: "XiHe",
  wenhai: "WenHai",
  langya: "LangYa",
  glo12: "GLO12",
  climatology: "Climatology",
  persistence: "Persistence",
};

const BASELINE_CHALLENGERS = new Set(["climatology", "persistence"]);

// Series colour slots, assigned to the challenger family and never to its rank, so a
// filter or a re-sort never repaints a line. The slot order is the validated one; the
// page resolves each slot to a theme-specific hex through a CSS variable.
const CHALLENGER_SERIES_SLOTS = {
  glonet: 1,
  xihe: 2,
  wenhai: 3,
  langya: 4,
  climatology: 5,
  persistence: 6,
};

const ONE_DEGREE_SUFFIX = "_1_degree";
const SKILL_FIELD = "skill_vs_persistence_1_degree";
const CHALLENGER_REGISTRY_PATH = "./data/challengers.json";
const MISSING = "none";

let registry = new Map();

export function challengerFamily(slug) {
  return slug.endsWith(ONE_DEGREE_SUFFIX) ? slug.slice(0, -ONE_DEGREE_SUFFIX.length) : slug;
}

export function challengerTrack(slug) {
  return slug.endsWith(ONE_DEGREE_SUFFIX) ? "one_degree" : "native";
}

export function challengerSeriesSlot(slug) {
  return CHALLENGER_SERIES_SLOTS[challengerFamily(slug)] ?? 6;
}

export function challengerLabel(slug) {
  const registered = registry.get(slug)?.display_name;
  if (registered) return registered;
  const family = challengerFamily(slug);
  const base = CHALLENGER_LABELS[family] ?? family;
  return challengerTrack(slug) === "one_degree" ? `${base} (1°)` : base;
}

export function isBaselineChallenger(slug) {
  const registered = registry.get(slug)?.is_baseline;
  if (registered !== undefined) return Boolean(registered);
  return BASELINE_CHALLENGERS.has(challengerFamily(slug));
}

export function variableLabel(variable) {
  return VARIABLE_LABELS[variable] ?? variable ?? "";
}

export function referenceLabel(reference) {
  return REFERENCE_LABELS[reference] ?? reference;
}

export function regionLabel(region) {
  return REGION_LABELS[region] ?? region;
}

export function metricLabel(metric) {
  return METRIC_LABELS[metric] ?? metric;
}

export function trackLabel(track) {
  return track === "one_degree" ? "1 degree" : "Native resolution";
}

/**
 * Build the labeller for a column set. The depth belongs in a label only when it carries
 * information the rest of the page does not already give: a depth-less variable never
 * shows one, a surface-only variable is just its name, and a variable published at several
 * depths at once names every one of them. That last case is why the Class 4 columns read
 * "temperature surface" beside "temperature 0-5m" while a gridded table at one selected
 * level just reads "temperature".
 */
export function columnLabeller(columns) {
  const depthsByVariable = new Map();
  for (const column of columns) {
    if (!depthsByVariable.has(column.variable)) depthsByVariable.set(column.variable, new Set());
    if (column.depth) depthsByVariable.get(column.variable).add(column.depth);
  }
  return (column) => {
    const label = variableLabel(column.variable);
    if (!column.depth) return label;
    const isOnlyDepth = (depthsByVariable.get(column.variable)?.size ?? 0) <= 1;
    if (column.depth === "surface" && isOnlyDepth) return label;
    return `${label} ${column.depth}`;
  };
}

/** Round to a fixed number of decimals chosen from the magnitude, or a dash when absent. */
export function formatMean(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  const magnitude = Math.abs(value);
  if (magnitude >= 100) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

export function formatSkill(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return null;
  const percent = value * 100;
  return `${percent >= 0 ? "+" : ""}${percent.toFixed(0)}%`;
}

function depthRank(depth) {
  const index = DEPTH_ORDER.indexOf(depth);
  return index === -1 ? DEPTH_ORDER.length : index;
}

function variableRank(variable) {
  const index = VARIABLE_ORDER.indexOf(variable);
  return index === -1 ? VARIABLE_ORDER.length : index;
}

// The two display orders above are also what a consumer building its own table has to
// sort by, so they are exported rather than copied.
export const depthDisplayRank = depthRank;
export const variableDisplayRank = variableRank;

export function sortDepths(depths) {
  return [...depths].sort((first, second) => depthRank(first) - depthRank(second));
}

function distinct(values) {
  return [...new Set(values)];
}

export function columnKey(column) {
  return `${column.variable}|${column.depth ?? MISSING}`;
}

function rowColumnKey(row) {
  return `${row.variable}|${row.depth ?? MISSING}`;
}

/**
 * Load the published summary rows and the optional challenger registry.
 *
 * Both go through the viewer's insight index and `resolveViewerDataUrl`, so the data root
 * resolves exactly as it does for the map: window config, then `?data=`, then the
 * viewer-config.json side-car, then the published bucket prefix. The registry is optional
 * and a missing one leaves the built-in labels in place.
 */
export async function loadScores() {
  const index = await loadInsightIndex();
  const rows = await loadScoresSummary(index);
  registry = await loadChallengerRegistry();
  return rows.filter((row) => Number.isFinite(row.mean));
}

async function loadChallengerRegistry() {
  try {
    const response = await fetch(resolveViewerDataUrl(CHALLENGER_REGISTRY_PATH), { cache: "no-cache" });
    if (!response.ok) return new Map();
    const parsed = await response.json();
    if (!parsed || typeof parsed !== "object") return new Map();
    return new Map(Object.entries(parsed));
  } catch (error) {
    return new Map();
  }
}

/** The distinct values the selectors offer, derived from the rows themselves. */
export function buildCatalog(rows) {
  const references = distinct(rows.map((row) => row.reference)).sort((first, second) => {
    const order = ["observations", "glorys", "glo12"];
    return order.indexOf(first) - order.indexOf(second);
  });
  return {
    regions: distinct(rows.map((row) => row.region)).sort(),
    references,
    leadDays: distinct(rows.map((row) => row.lead_day)).sort((first, second) => first - second),
    years: distinct(rows.map((row) => row.year)).sort(),
    tracks: distinct(rows.map((row) => challengerTrack(row.challenger))),
  };
}

export function metricForReference(reference) {
  return reference === "observations" ? "class4_rmsd" : "rmsd";
}

/**
 * Depths offered for a reference. Class 4 depths are per-variable ranges rather than one
 * shared level set, so the observations view shows every variable/depth pair at once and
 * offers no depth selector; the gridded references share one level set and do.
 */
export function depthsForReference(rows, reference, region) {
  if (metricForReference(reference) === "class4_rmsd") return [];
  return sortDepths(
    distinct(
      rows
        .filter((row) => row.reference === reference && row.region === region && row.depth)
        .map((row) => row.depth),
    ),
  );
}

function matchesScope(row, selection) {
  return (
    row.region === selection.region &&
    row.reference === selection.reference &&
    challengerTrack(row.challenger) === selection.track
  );
}

// A column is a variable at a depth. Against a gridded reference the selected level plus
// the depth-less variables (mixed layer depth, geostrophic currents) are shown; against
// observations every published variable/depth pair is a column.
function matchesDepth(row, selection) {
  if (metricForReference(selection.reference) === "class4_rmsd") return true;
  return row.depth === null || row.depth === selection.depth;
}

export function tableColumns(rows, selection) {
  const columns = new Map();
  for (const row of rows) {
    if (!matchesScope(row, selection) || !matchesDepth(row, selection)) continue;
    if (row.lead_day !== selection.leadDay) continue;
    const key = rowColumnKey(row);
    if (!columns.has(key)) columns.set(key, { variable: row.variable, depth: row.depth, unit: row.unit });
  }
  return [...columns.values()].sort((first, second) => {
    const byVariable = variableRank(first.variable) - variableRank(second.variable);
    return byVariable !== 0 ? byVariable : depthRank(first.depth) - depthRank(second.depth);
  });
}

export function tableChallengers(rows, selection) {
  const slugs = distinct(
    rows.filter((row) => matchesScope(row, selection) && row.lead_day === selection.leadDay).map((row) => row.challenger),
  );
  return slugs.sort((first, second) => challengerLabel(first).localeCompare(challengerLabel(second)));
}

/** Index the rows once by challenger, column and lead day so lookups stay constant time. */
export function indexRows(rows, selection) {
  const index = new Map();
  for (const row of rows) {
    if (!matchesScope(row, selection)) continue;
    index.set(`${row.challenger}|${rowColumnKey(row)}|${row.lead_day}`, row);
  }
  return index;
}

export function cellFor(index, challenger, column, leadDay) {
  return index.get(`${challenger}|${columnKey(column)}|${leadDay}`) ?? null;
}

export function skillOf(row) {
  return row ? row[SKILL_FIELD] : null;
}

/**
 * Order the challengers for display. Lower error is better, so a ranked column sorts
 * ascending first; challengers with no value for that column always sink to the bottom
 * rather than sorting as zero. With no ranked column the neutral alphabetical order of
 * `tableChallengers` is kept.
 */
export function rankChallengers(challengers, index, column, leadDay, direction) {
  if (!column || direction === 0) return [...challengers];
  return [...challengers]
    .map((challenger) => ({ challenger, value: cellFor(index, challenger, column, leadDay)?.mean ?? null }))
    .sort((first, second) => {
      if (first.value === null) return 1;
      if (second.value === null) return -1;
      return direction * (first.value - second.value);
    })
    .map((entry) => entry.challenger);
}

/** The lead-day series for one challenger and column, shaped for `leadCurveSVG`. */
export function leadSeries(index, challenger, column, leadDays) {
  const series = [];
  for (const leadDay of leadDays) {
    const row = cellFor(index, challenger, column, leadDay);
    if (row) series.push({ lead_day: leadDay, mean: row.mean, ci_low: row.ci_low, ci_high: row.ci_high });
  }
  return series;
}
