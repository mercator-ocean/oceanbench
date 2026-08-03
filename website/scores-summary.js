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
// the 1 degree persistence baseline), so nothing is recomputed here.
//
// The data root resolves exactly as it does for the viewer: window config, then `?data=`,
// then the viewer-config.json side-car, then the published bucket prefix.

import { initializeViewerConfig, viewerDataBaseUrl } from "./viewer/config.js";
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
    titles[`rmsd_variables_${suffix}`] = "Forecasted variables";
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
    parts.push(`95% CI half width: ${formatMean((row.ci_high - row.ci_low) / 2)}`);
  }
  const skill = formatSkill(row.skill_vs_persistence_1_degree);
  if (skill) parts.push(`Skill vs 1 degree persistence: ${skill}`);
  return parts.join("\n");
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
    const variable = (depth.variables[variableLabel(row.variable)] ??= {
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
    window.OceanBenchScores.render(buildBundle(state.rows), { reportLinks: false });

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
