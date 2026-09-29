// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Feeds the site's scores tables from the published scores-summary.json.
//
// The tables themselves are the ones the site has always drawn: this module only reshapes
// the published rows into the bundle `interactive-scores.js` reads, then hands it over
// through `window.OceanBenchScores.render`. Nothing about the table markup, the baseline
// colouring, the depth pills or the section tabs changes.
//
// The published rows are already lead-time resolved (one row per challenger, region,
// variable, depth, metric and lead day, with its bootstrap interval and its skill against
// the 1 degree persistence baseline), so the whole year is drawn as published. A shorter
// period is recomputed in the browser from the per-start scores.parquet, fetched only then.
//
// The data root resolves exactly as it does for the viewer: window config, then `?data=`,
// then the viewer-config.json side-car, then the published bucket prefix.

import { initializeViewerConfig, resolveViewerDataUrl, viewerDataBaseUrl } from "./viewer/config.js";
import { aggregatePeriod, periodStartIndices, readPerStartScores } from "./viewer/modules/scores-periods.js";
import {
  challengerFamily,
  challengerLabel,
  depthDisplayRank,
  formatMean,
  formatSkill,
  isBaselineChallenger,
  loadScores,
  regionLabel,
  variableDisplayRank,
  variableLabel,
} from "./viewer/modules/scores-data.js";

// One published reference per site section, in the order the tables are built. The two
// gridded references come first so the depth pills are offered the shared level set rather
// than the per-variable Class 4 ranges.
const REFERENCE_SECTIONS = [
  { reference: "glorys", section: "reanalysis", suffix: "glorys" },
  { reference: "glo12", section: "analysis", suffix: "glo12" },
  { reference: "observations", section: "observations", suffix: "observations" },
];

const MIXED_LAYER_VARIABLE = "ocean_mixed_layer_thickness";
const VERSION_KEY = "published";
const FLAT_DEPTH = "flat";

const state = { rows: [] };

// scores.parquet sits at the benchmark root, two levels above the viewer data directory.
const PER_START_SCORES_PATH = "../../scores.parquet";
const WHOLE_YEAR = "year";
const CUSTOM = "custom";
const QUARTERS = [
  { value: "jan-mar", label: "Jan-Mar", from: "01-01", to: "03-31" },
  { value: "apr-jun", label: "Apr-Jun", from: "04-01", to: "06-30" },
  { value: "jul-sep", label: "Jul-Sep", from: "07-01", to: "09-30" },
  { value: "oct-dec", label: "Oct-Dec", from: "10-01", to: "12-31" },
];

const period = {
  year: null,
  active: WHOLE_YEAR,
  from: null,
  to: null,
  note: "",
  perStart: null,
  results: new Map(),
  request: 0,
};

function sectionFor(reference) {
  return REFERENCE_SECTIONS.find((entry) => entry.reference === reference) ?? null;
}

// A row with no depth is a diagnostic variable (mixed layer depth, geostrophic currents);
// those are the site's "physically consistent diagnostic variables" tables.
function metricKeyFor(row, suffix) {
  if (row.depth) return `rmsd_variables_${suffix}`;
  if (row.variable === MIXED_LAYER_VARIABLE) return `rmsd_mld_${suffix}`;
  return `rmsd_geostrophic_${suffix}`;
}

function metricTitles() {
  const titles = {};
  for (const { suffix } of REFERENCE_SECTIONS) {
    titles[`rmsd_variables_${suffix}`] = "Forecast variables";
    titles[`rmsd_mld_${suffix}`] = "RMSD of Mixed Layer Depth";
    titles[`rmsd_geostrophic_${suffix}`] = "RMSD of Geostrophic Currents";
  }
  return titles;
}

// Class 4 depths are per-variable ranges rather than one shared level set, so the
// observations section is laid out as the site lays it out: one table per group of depths
// that share a variable set.
function sectionConfigurations() {
  const sections = {};
  for (const { section, suffix, reference } of REFERENCE_SECTIONS) {
    sections[section] = {
      depth_metric: `rmsd_variables_${suffix}`,
      flat_metrics: [`rmsd_mld_${suffix}`, `rmsd_geostrophic_${suffix}`],
    };
    if (reference !== "observations") continue;
    sections[section].flat_metrics = [];
    sections[section].depth_groups = [
      { depths: ["0-5m", "5-100m", "100-300m", "300-600m"], variables: ["temperature", "salinity"] },
      { depths: ["surface"], variables: ["sea surface height", "temperature"], show_depth_label: true },
      { depths: ["15m"], variables: ["zonal current", "meridional current"], show_depth_label: true },
    ];
  }
  return sections;
}

// The challenger track is a chip in the controls, so the row label carries the model name
// alone rather than repeating the resolution.
function modelLabel(challenger) {
  return challengerLabel(challengerFamily(challenger));
}

// Depth and variable order is insertion order in the bundle, so the rows are sorted once
// into the order the tables read in.
function displayOrdered(rows) {
  return [...rows].sort((first, second) => {
    const byDepth = depthDisplayRank(first.depth) - depthDisplayRank(second.depth);
    if (byDepth !== 0) return byDepth;
    const byVariable = variableDisplayRank(first.variable) - variableDisplayRank(second.variable);
    if (byVariable !== 0) return byVariable;
    return first.lead_day - second.lead_day;
  });
}

// The confidence interval and the skill do not fit in a cell that has to stay readable at
// five lead days per variable, so they ride along as the cell's tooltip note.
function annotationFor(row) {
  const parts = [];
  if (Number.isFinite(row.ci_low) && Number.isFinite(row.ci_high)) {
    parts.push(`±95% CI: ${formatMean((row.ci_high - row.ci_low) / 2)}`);
  }
  // The published column is named after the 1 degree baseline, but each row carries its own
  // baseline slug (native persistence on the native track).
  const skill = formatSkill(row.skill_vs_persistence_1_degree);
  if (skill && row.skill_baseline) parts.push(`Skill vs ${challengerLabel(row.skill_baseline)}: ${skill}`);
  return parts.join("\n");
}

// Lagrangian rows carry no variable; name their column so the header is not blank. Keep in
// sync with TRAJECTORY_VARIABLE_LABEL in interactive-scores.js.
function variableLabelFor(row) {
  if (row.variable == null && String(row.metric).startsWith("lagrangian")) return "trajectory separation";
  return variableLabel(row.variable);
}

function buildBundle(rows) {
  const regions = {};
  for (const row of displayOrdered(rows)) {
    const section = sectionFor(row.reference);
    if (!section) continue;

    const region = (regions[row.region] ??= {
      display_name: regionLabel(row.region),
      challengers: {},
      challenger_names: [],
    });
    const challenger = (region.challengers[row.challenger] ??= {});
    const score = (challenger[metricKeyFor(row, section.suffix)] ??= { depths: {} });
    const depth = (score.depths[row.depth ?? FLAT_DEPTH] ??= { variables: {} });
    const variable = (depth.variables[variableLabelFor(row)] ??= {
      unit: row.unit ?? "",
      standard_name: row.variable,
      data: {},
      annotations: {},
    });

    const leadDay = String(row.lead_day);
    variable.data[leadDay] = row.mean;
    const annotation = annotationFor(row);
    if (annotation) variable.annotations[leadDay] = annotation;
  }

  for (const region of Object.values(regions)) {
    region.challenger_names = Object.keys(region.challengers).sort((first, second) => {
      const byLabel = modelLabel(first).localeCompare(modelLabel(second));
      return byLabel !== 0 ? byLabel : first.localeCompare(second);
    });
  }

  const regionOrder = Object.keys(regions).sort((first, second) => (first === "global" ? -1 : second === "global" ? 1 : first.localeCompare(second)));
  // Persistence and climatology are skill floors rather than competitors. Marking them as
  // baselines is what keeps them out of the default table and out of the default
  // comparison reference, exactly as the site treats them.
  const challengerLabels = {};
  const challengerCategories = {};
  for (const region of Object.values(regions)) {
    for (const challenger of region.challenger_names) {
      challengerLabels[challenger] = modelLabel(challenger);
      challengerCategories[challenger] = isBaselineChallenger(challenger) ? "baseline" : "model";
    }
  }

  return {
    versions: {
      [VERSION_KEY]: {
        regions,
        region_order: regionOrder,
        region_labels: Object.fromEntries(regionOrder.map((region) => [region, regionLabel(region)])),
        region_metadata: window.OCEANBENCH_REGION_METADATA ?? {},
        challenger_labels: challengerLabels,
        challenger_categories: challengerCategories,
      },
    },
    version_order: [VERSION_KEY],
    default_version: VERSION_KEY,
    metric_titles: metricTitles(),
    sections: sectionConfigurations(),
  };
}

/* -- period -------------------------------------------------------------------------- */

function forecastCount(count) {
  return `${count} forecast${count === 1 ? "" : "s"}`;
}

function setPeriodRange(value, from, to) {
  const quarter = QUARTERS.find((entry) => entry.value === value);
  period.active = value;
  if (value === WHOLE_YEAR) {
    period.from = `${period.year}-01-01`;
    period.to = `${period.year}-12-31`;
  } else if (quarter) {
    period.from = `${period.year}-${quarter.from}`;
    period.to = `${period.year}-${quarter.to}`;
  } else {
    period.from = from;
    period.to = to;
  }
}

function readPeriodFromUrl() {
  const value = new URLSearchParams(window.location.search).get("period");
  if (!value) return;
  if (QUARTERS.some((entry) => entry.value === value)) {
    setPeriodRange(value);
    return;
  }
  const custom = /^(\d{4}-\d{2}-\d{2})_(\d{4}-\d{2}-\d{2})$/.exec(value);
  if (custom) setPeriodRange(CUSTOM, custom[1], custom[2]);
}

function periodControl() {
  const urlValue = period.active === WHOLE_YEAR
    ? null
    : period.active === CUSTOM
      ? `${period.from}_${period.to}`
      : period.active;
  return {
    options: [
      { value: WHOLE_YEAR, label: "Whole year" },
      ...QUARTERS.map(({ value, label }) => ({ value, label })),
      { value: CUSTOM, label: "Custom" },
    ],
    active: period.active,
    from: period.from,
    to: period.to,
    min: `${period.year}-01-01`,
    max: `${period.year}-12-31`,
    note: period.note,
    urlValue,
    onSelect: selectPeriod,
  };
}

async function loadPerStartScores() {
  period.perStart ??= fetch(resolveViewerDataUrl(PER_START_SCORES_PATH))
    .then((response) => {
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      return response.arrayBuffer();
    })
    .then(readPerStartScores)
    .catch((error) => {
      period.perStart = null;
      throw error;
    });
  return period.perStart;
}

// Rows for the current range, or null when no forecast starts inside it.
async function periodRows() {
  const key = `${period.from}_${period.to}`;
  if (period.results.has(key)) return period.results.get(key);
  const perStart = await loadPerStartScores();
  const indices = periodStartIndices(perStart, period.from, period.to);
  const result = indices.length === 0
    ? null
    : {
        count: indices.length,
        rows: aggregatePeriod(perStart, state.rows, indices).filter((row) => Number.isFinite(row.mean)),
      };
  period.results.set(key, result);
  return result;
}

function wholeYearNote() {
  return forecastCount(Math.max(0, ...state.rows.map((row) => row.n_starts ?? 0)));
}

// Resolves to the bundle to draw, or null when the tables should stay as they are.
async function bundleForPeriod() {
  if (period.active === WHOLE_YEAR) {
    period.note = wholeYearNote();
    return buildBundle(state.rows);
  }
  if (period.from > period.to) {
    period.note = "Start is after end";
    return null;
  }
  try {
    const result = await periodRows();
    if (!result) {
      period.note = "No forecast starts in this range";
      return null;
    }
    period.note = forecastCount(result.count);
    return buildBundle(result.rows);
  } catch (error) {
    period.note = `Could not load (${error.message})`;
    return null;
  }
}

async function selectPeriod(value, from, to) {
  if (value === period.active && value !== CUSTOM) return;
  setPeriodRange(value, from, to);
  const request = ++period.request;
  period.note = value === WHOLE_YEAR ? wholeYearNote() : "Loading";
  window.OceanBenchScores.update(null, { period: periodControl() });
  // Let the loading note paint before the aggregation holds the main thread.
  await new Promise((resolve) => setTimeout(resolve, 30));
  const bundle = await bundleForPeriod();
  if (request !== period.request) return;
  window.OceanBenchScores.update(bundle, { period: periodControl() });
}

/* -- boot ------------------------------------------------------------------------------ */

function reportStatus(message, isError) {
  const status = document.getElementById("scores-status");
  if (!status) return;
  status.textContent = message;
  status.hidden = false;
  status.classList.toggle("scores-status-error", Boolean(isError));
}

async function boot() {
  try {
    await initializeViewerConfig();
    state.rows = await loadScores();
    if (!state.rows.length) throw new Error(`no score rows were read from ${viewerDataBaseUrl()}`);
    period.year = state.rows[0].year;
    setPeriodRange(WHOLE_YEAR);
    readPeriodFromUrl();
    const bundle = (await bundleForPeriod()) ?? buildBundle(state.rows);
    window.OceanBenchScores.render(bundle, { reportLinks: false, period: periodControl() });

    const status = document.getElementById("scores-status");
    if (status) status.hidden = true;
  } catch (error) {
    reportStatus(
      `The published scores could not be loaded (${error.message}). The tables below stay empty; ` +
        `reload the page or point it at another data root with the ?data= parameter.`,
      true,
    );
  }
}

boot();
