// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Entry point for the scores page. The viewer owns the fields, this page owns the
// numbers: a ranked challenger by variable table and the error growth against lead day.
//
// The data root is not resolved here. `initializeViewerConfig` and the insight index are
// the viewer's own, so the window config, the `?data=` query parameter, the
// viewer-config.json side-car and the published bucket prefix keep their single order of
// precedence across both pages.

import { initializeViewerConfig } from "./config.js";
import {
  buildCatalog,
  columnKey,
  columnLabeller,
  depthsForReference,
  indexRows,
  loadScores,
  metricForReference,
  metricLabel,
  rankChallengers,
  referenceLabel,
  regionLabel,
  tableChallengers,
  tableColumns,
  trackLabel,
} from "./modules/scores-data.js";
import { renderErrorGrowth, renderTable, wireChartCursor } from "./modules/scores-view.js";

const HEADLINE_VARIABLE = "sea_surface_height_above_geoid";
const DEFAULT_LEAD_DAY = 5;

const element = (id) => document.getElementById(id);

const state = {
  rows: [],
  catalog: null,
  theme: "light",
  selection: { region: null, reference: null, track: "native", depth: null, leadDay: DEFAULT_LEAD_DAY },
  sort: { columnKey: null, direction: 0 },
  chartColumnKey: null,
};

function fillSelect(select, options, selected, labelFor) {
  select.replaceChildren();
  for (const option of options) {
    const node = document.createElement("option");
    node.value = String(option);
    node.textContent = labelFor ? labelFor(option) : String(option);
    if (String(option) === String(selected)) node.selected = true;
    select.appendChild(node);
  }
}

function applyTheme() {
  document.documentElement.dataset.theme = state.theme;
  element("theme-toggle").textContent = state.theme === "light" ? "Dark theme" : "Light theme";
}

function currentColumns() {
  return tableColumns(state.rows, state.selection);
}

function headlineColumn(columns) {
  return columns.find((column) => column.variable === HEADLINE_VARIABLE) ?? columns[0] ?? null;
}

function refreshDepthControl() {
  const control = element("depth-control");
  const depths = depthsForReference(state.rows, state.selection.reference, state.selection.region);
  if (!depths.length) {
    control.hidden = true;
    state.selection.depth = null;
    return;
  }
  control.hidden = false;
  if (!depths.includes(state.selection.depth)) state.selection.depth = depths[0];
  fillSelect(element("depth-select"), depths, state.selection.depth);
}

// The columns change with the reference, the depth and the track, so any selection that
// names a column has to be revalidated rather than carried over blindly.
function refreshColumnDependentSelections() {
  const columns = currentColumns();
  const keys = columns.map(columnKey);
  if (!keys.includes(state.sort.columnKey)) {
    const fallback = headlineColumn(columns);
    state.sort = fallback ? { columnKey: columnKey(fallback), direction: 1 } : { columnKey: null, direction: 0 };
  }
  if (!keys.includes(state.chartColumnKey)) {
    const fallback = headlineColumn(columns);
    state.chartColumnKey = fallback ? columnKey(fallback) : null;
  }
  const labelOf = columnLabeller(columns);
  fillSelect(element("chart-variable-select"), keys, state.chartColumnKey, (key) => {
    const column = columns.find((candidate) => columnKey(candidate) === key);
    return column ? labelOf(column) : key;
  });
}

function toggleSort(key) {
  if (state.sort.columnKey !== key) state.sort = { columnKey: key, direction: 1 };
  else if (state.sort.direction === 1) state.sort = { columnKey: key, direction: -1 };
  else state.sort = { columnKey: null, direction: 0 };
  render();
}

function render() {
  const { selection } = state;
  const columns = currentColumns();
  const index = indexRows(state.rows, selection);
  const metric = metricLabel(metricForReference(selection.reference));
  const labelOf = columnLabeller(columns);
  const sortedColumn = columns.find((column) => columnKey(column) === state.sort.columnKey) ?? null;
  const challengers = rankChallengers(
    tableChallengers(state.rows, selection),
    index,
    sortedColumn,
    selection.leadDay,
    state.sort.direction,
  );

  const depthText = selection.depth ? `, ${selection.depth}` : "";
  element("table-note").textContent =
    `${metric} against ${referenceLabel(selection.reference)}, ${regionLabel(selection.region)}, ` +
    `${trackLabel(selection.track).toLowerCase()} track, lead day ${selection.leadDay}${depthText}. ` +
    "Each cell is the mean over forecast starts with half the width of its bootstrap 95% confidence interval. " +
    "Lower is better.";
  element("clear-ranking").hidden = state.sort.direction === 0;

  renderTable(element("scores-table"), {
    challengers,
    columns,
    index,
    leadDay: selection.leadDay,
    sort: state.sort,
    labelOf,
    onSort: toggleSort,
  });

  const chartColumn = columns.find((column) => columnKey(column) === state.chartColumnKey) ?? null;
  renderErrorGrowth(element("chart-host"), element("chart-legend"), {
    challengers,
    column: chartColumn,
    index,
    leadDays: state.catalog.leadDays,
    metric,
    labelOf,
  });
  wireChartCursor(element("chart-host"));
  element("chart-note").textContent = chartColumn
    ? `${metric} for ${labelOf(chartColumn)} against ${referenceLabel(selection.reference)}, ` +
      `${regionLabel(selection.region)}. Shaded bands are the bootstrap 95% confidence interval.`
    : `Nothing is published for ${referenceLabel(selection.reference)} on the ` +
      `${trackLabel(selection.track).toLowerCase()} track in ${regionLabel(selection.region)}.`;
}

function wireControls() {
  const { catalog, selection } = state;
  fillSelect(element("region-select"), catalog.regions, selection.region, regionLabel);
  fillSelect(element("reference-select"), catalog.references, selection.reference, referenceLabel);
  fillSelect(element("track-select"), catalog.tracks, selection.track, trackLabel);
  fillSelect(element("lead-select"), catalog.leadDays, selection.leadDay, (lead) => `day ${lead}`);
  refreshDepthControl();
  refreshColumnDependentSelections();

  element("region-select").addEventListener("change", (event) => {
    selection.region = event.target.value;
    refreshDepthControl();
    refreshColumnDependentSelections();
    render();
  });
  element("reference-select").addEventListener("change", (event) => {
    selection.reference = event.target.value;
    refreshDepthControl();
    refreshColumnDependentSelections();
    render();
  });
  element("track-select").addEventListener("change", (event) => {
    selection.track = event.target.value;
    refreshColumnDependentSelections();
    render();
  });
  element("depth-select").addEventListener("change", (event) => {
    selection.depth = event.target.value;
    refreshColumnDependentSelections();
    render();
  });
  element("lead-select").addEventListener("change", (event) => {
    selection.leadDay = Number(event.target.value);
    render();
  });
  element("chart-variable-select").addEventListener("change", (event) => {
    state.chartColumnKey = event.target.value;
    render();
  });
  element("clear-ranking").addEventListener("click", () => {
    state.sort = { columnKey: null, direction: 0 };
    render();
  });
  element("theme-toggle").addEventListener("click", () => {
    state.theme = state.theme === "light" ? "dark" : "light";
    applyTheme();
    render();
  });
}

async function boot() {
  const status = element("status");
  applyTheme();
  try {
    await initializeViewerConfig();
    state.rows = await loadScores();
    if (!state.rows.length) throw new Error("the published scores summary is empty");
    state.catalog = buildCatalog(state.rows);
    state.selection.region = state.catalog.regions.includes("global") ? "global" : state.catalog.regions[0];
    state.selection.reference = state.catalog.references[0];
    state.selection.track = state.catalog.tracks.includes("native") ? "native" : state.catalog.tracks[0];
    if (!state.catalog.leadDays.includes(state.selection.leadDay)) {
      state.selection.leadDay = state.catalog.leadDays[Math.min(2, state.catalog.leadDays.length - 1)];
    }
    wireControls();
    render();
    status.hidden = true;
    element("scores-main").hidden = false;
    element("controls").hidden = false;
  } catch (error) {
    status.className = "status error";
    status.textContent = `Could not load the scores summary: ${error.message}`;
  }
}

boot();
