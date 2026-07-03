// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// v0 score page (contracts.md §1 NO-RANKING, §3.4 derived stats).
//
// scores.parquet is the canonical artifact: it is read directly in the browser with the
// vendored hyparquet reader and the per-start means are aggregated here. The bootstrap
// confidence intervals and skill-vs-baseline are expensive to recompute in the browser, so
// they are read from the precomputed scores-summary.json emitted next to the parquet by the
// publish step. Neither imposes a rank order; the scorecard sorts only on explicit user click.

import { parquetReadObjects } from "./vendor/hyparquet/hyparquet.min.js";

const DATA = {
  parquet: "./data/scores.parquet",
  summary: "./data/scores-summary.json",
  challengers: "./data/challengers.json",
};

const VARIABLE_LABELS = {
  sea_surface_height_above_geoid: "sea surface height",
  sea_water_potential_temperature: "temperature",
  sea_water_salinity: "salinity",
  northward_sea_water_velocity: "meridional current",
  eastward_sea_water_velocity: "zonal current",
  ocean_mixed_layer_thickness: "mixed layer depth",
  geostrophic_northward_sea_water_velocity: "meridional geostrophic current",
  geostrophic_eastward_sea_water_velocity: "zonal geostrophic current",
};

const REFERENCE_LABELS = {
  glorys: "GLORYS reanalysis",
  glo12: "GLO12 analysis",
  observations: "observations (Class-4)",
};

const METRIC_PHRASE = { rmsd: "RMSD", class4_rmsd: "Class-4 RMSD" };
const DEPTH_ORDER = ["surface", "50m", "100m", "200m", "300m", "500m", "0-5m", "5-100m", "15m", "100-300m", "300-600m"];
const LEAD_OPTIONS = [1, 3, 5, 7, 10];
const NULL_KEY = "∅";

const state = {
  aggregated: new Map(), // identity key -> aggregated cell
  summaryByKey: new Map(), // identity key -> {ci_low, ci_high, skill...}
  challengerMeta: new Map(), // slug -> {display_name, is_baseline}
  skillBaseline: null, // slug of baseline the summary carries skill against, if any
  selection: { region: null, reference: null, depth: null, lead: 5, year: null, showSkill: false },
  sort: { column: null, direction: 0 }, // 0 neutral, 1 ascending, -1 descending
  summarySelection: { variable: null, depth: null },
};

const $ = (id) => document.getElementById(id);

function identityKey(fields) {
  return [
    fields.challenger,
    fields.year,
    fields.region,
    fields.metric,
    fields.reference,
    fields.variable ?? NULL_KEY,
    fields.depth ?? NULL_KEY,
    fields.lead_day,
  ].join("|");
}

function nullable(value) {
  return value === null || value === undefined || value === "" ? null : value;
}

async function loadParquet(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} -> HTTP ${response.status}`);
  const buffer = await response.arrayBuffer();
  const file = {
    byteLength: buffer.byteLength,
    async slice(start, end) {
      return buffer.slice(start, end ?? buffer.byteLength);
    },
  };
  return parquetReadObjects({ file, rowFormat: "object" });
}

async function loadJSON(url, optional = false) {
  const response = await fetch(url);
  if (!response.ok) {
    if (optional) return null;
    throw new Error(`${url} -> HTTP ${response.status}`);
  }
  return response.json();
}

// Aggregate per-start rows into the displayed point estimate: a plain mean of the per-start
// values, except Class-4 RMSD which recombines n-weighted (sqrt(sum(v^2 n) / sum(n))) — the
// same pooling the pipeline and scores-summary.json use.
function aggregateRows(rows) {
  const groups = new Map();
  for (const row of rows) {
    const fields = {
      challenger: row.challenger,
      year: Number(row.year),
      region: row.region,
      metric: row.metric,
      reference: nullable(row.reference),
      variable: nullable(row.variable),
      depth: nullable(row.depth),
      lead_day: Number(row.lead_day),
    };
    const key = identityKey(fields);
    let group = groups.get(key);
    if (!group) {
      group = { ...fields, unit: row.unit, sum: 0, sumSquaresWeighted: 0, weight: 0, count: 0 };
      groups.set(key, group);
    }
    const value = Number(row.value);
    if (!Number.isFinite(value)) continue;
    group.count += 1;
    if (row.metric === "class4_rmsd") {
      const n = Number(row.n) || 0;
      group.sumSquaresWeighted += value * value * n;
      group.weight += n;
    } else {
      group.sum += value;
    }
  }

  const aggregated = new Map();
  for (const [key, group] of groups) {
    const mean =
      group.metric === "class4_rmsd"
        ? Math.sqrt(group.sumSquaresWeighted / group.weight)
        : group.sum / group.count;
    aggregated.set(key, {
      challenger: group.challenger,
      year: group.year,
      region: group.region,
      metric: group.metric,
      reference: group.reference,
      variable: group.variable,
      depth: group.depth,
      lead_day: group.lead_day,
      unit: group.unit,
      mean,
      n_starts: group.count,
    });
  }
  return aggregated;
}

function indexSummary(records) {
  const byKey = new Map();
  let skillBaseline = null;
  for (const record of records) {
    const fields = {
      challenger: record.challenger,
      year: Number(record.year),
      region: record.region,
      metric: record.metric,
      reference: nullable(record.reference),
      variable: nullable(record.variable),
      depth: nullable(record.depth),
      lead_day: Number(record.lead_day),
    };
    byKey.set(identityKey(fields), record);
    const skillField = Object.keys(record).find((name) => name.startsWith("skill_vs_"));
    if (skillField) skillBaseline = skillField.slice("skill_vs_".length);
  }
  return { byKey, skillBaseline };
}

function distinct(values) {
  return [...new Set(values)];
}

function sortedDepths(depths) {
  return depths.slice().sort((a, b) => {
    const ia = DEPTH_ORDER.indexOf(a);
    const ib = DEPTH_ORDER.indexOf(b);
    return (ia === -1 ? 99 : ia) - (ib === -1 ? 99 : ib);
  });
}

function formatValue(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  const magnitude = Math.abs(value);
  if (magnitude >= 100) return value.toFixed(1);
  if (magnitude >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function variableLabel(variable) {
  return VARIABLE_LABELS[variable] ?? variable ?? "—";
}

function challengerLabel(slug) {
  return state.challengerMeta.get(slug)?.display_name ?? slug;
}

function isBaseline(slug) {
  return Boolean(state.challengerMeta.get(slug)?.is_baseline);
}

function orderedChallengers() {
  const slugs = distinct([...state.aggregated.values()].map((cell) => cell.challenger));
  const baselines = slugs.filter(isBaseline).sort((a, b) => challengerLabel(a).localeCompare(challengerLabel(b)));
  const others = slugs.filter((slug) => !isBaseline(slug)).sort((a, b) => challengerLabel(a).localeCompare(challengerLabel(b)));
  return { baselines, others };
}

// ---- selectors -----------------------------------------------------------

function metricForReference(reference) {
  return reference === "observations" ? "class4_rmsd" : "rmsd";
}

function populateSelect(select, options, selected, labelFor) {
  select.replaceChildren();
  for (const option of options) {
    const element = document.createElement("option");
    element.value = String(option);
    element.textContent = labelFor ? labelFor(option) : String(option);
    if (String(option) === String(selected)) element.selected = true;
    select.appendChild(element);
  }
}

function buildControls() {
  const cells = [...state.aggregated.values()];
  const years = distinct(cells.map((cell) => cell.year)).sort();
  state.selection.year = years[years.length - 1];

  const regions = distinct(cells.map((cell) => cell.region)).sort();
  state.selection.region = regions[0];
  populateSelect($("region-select"), regions, state.selection.region);

  const references = distinct(cells.map((cell) => cell.reference).filter(Boolean)).sort();
  state.selection.reference = references.includes("glorys") ? "glorys" : references[0];
  populateSelect($("reference-select"), references, state.selection.reference, (r) => REFERENCE_LABELS[r] ?? r);

  const leads = distinct(cells.map((cell) => cell.lead_day)).sort((a, b) => a - b);
  const leadChoices = LEAD_OPTIONS.filter((lead) => leads.includes(lead));
  if (!leadChoices.includes(state.selection.lead)) state.selection.lead = leadChoices[Math.min(2, leadChoices.length - 1)];
  populateSelect($("lead-select"), leadChoices, state.selection.lead, (l) => `day ${l}`);

  refreshDepthControl();

  const hasBaselines = orderedChallengers().baselines.length > 0;
  const skillControl = $("skill-control");
  skillControl.hidden = !(hasBaselines && state.skillBaseline);

  $("region-select").addEventListener("change", (event) => { state.selection.region = event.target.value; render(); });
  $("reference-select").addEventListener("change", (event) => {
    state.selection.reference = event.target.value;
    state.sort = { column: null, direction: 0 };
    refreshDepthControl();
    render();
  });
  $("depth-select").addEventListener("change", (event) => { state.selection.depth = event.target.value; render(); });
  $("lead-select").addEventListener("change", (event) => { state.selection.lead = Number(event.target.value); render(); });
  $("skill-toggle").addEventListener("change", (event) => { state.selection.showSkill = event.target.checked; render(); });
  $("reset-sort").addEventListener("click", () => { state.sort = { column: null, direction: 0 }; render(); });

  $("summary-variable-select").addEventListener("change", (event) => {
    state.summarySelection.variable = event.target.value;
    refreshSummaryDepthControl();
    renderSummaries();
  });
  $("summary-depth-select").addEventListener("change", (event) => { state.summarySelection.depth = event.target.value; renderSummaries(); });
}

function refreshDepthControl() {
  const control = $("depth-control");
  if (metricForReference(state.selection.reference) === "class4_rmsd") {
    control.hidden = true;
    state.selection.depth = null;
    return;
  }
  control.hidden = false;
  const depths = sortedDepths(
    distinct(
      [...state.aggregated.values()]
        .filter((cell) => cell.metric === "rmsd" && cell.reference === state.selection.reference && cell.depth)
        .map((cell) => cell.depth),
    ),
  );
  if (!depths.includes(state.selection.depth)) state.selection.depth = depths[0];
  populateSelect($("depth-select"), depths, state.selection.depth);
}

// ---- scorecard columns ---------------------------------------------------

function scorecardColumns() {
  const { region, reference, depth, lead, year } = state.selection;
  const metric = metricForReference(reference);
  const cells = [...state.aggregated.values()].filter(
    (cell) =>
      cell.region === region &&
      cell.year === year &&
      cell.metric === metric &&
      cell.lead_day === lead &&
      (metric === "class4_rmsd" ? true : cell.reference === reference),
  );

  const relevant = cells.filter((cell) =>
    metric === "class4_rmsd" ? true : cell.depth === null || cell.depth === depth,
  );

  const seen = new Map();
  for (const cell of relevant) {
    const columnKey = `${cell.variable ?? NULL_KEY}|${cell.depth ?? NULL_KEY}`;
    if (!seen.has(columnKey)) {
      seen.set(columnKey, { variable: cell.variable, depth: cell.depth, unit: cell.unit });
    }
  }
  return [...seen.values()].sort((a, b) => {
    const va = variableLabel(a.variable);
    const vb = variableLabel(b.variable);
    if (va !== vb) return va.localeCompare(vb);
    return sortedDepths([a.depth ?? "", b.depth ?? ""])[0] === (a.depth ?? "") ? -1 : 1;
  });
}

function columnHeader(column) {
  const label = variableLabel(column.variable);
  const depthSuffix = column.depth && column.depth !== "surface" ? ` @ ${column.depth}` : "";
  return `${label}${depthSuffix}`;
}

function cellFor(challenger, column) {
  const { region, reference, lead, year } = state.selection;
  const metric = metricForReference(reference);
  const key = identityKey({
    challenger,
    year,
    region,
    metric,
    reference: metric === "class4_rmsd" ? "observations" : reference,
    variable: column.variable,
    depth: column.depth,
    lead_day: lead,
  });
  const aggregated = state.aggregated.get(key);
  if (!aggregated) return null;
  const summary = state.summaryByKey.get(key);
  return { ...aggregated, summary };
}

// ---- rendering -----------------------------------------------------------

function render() {
  renderScorecard();
  refreshSummaryControls();
  renderSummaries();
}

function renderScorecard() {
  const columns = scorecardColumns();
  const { baselines, others } = orderedChallengers();

  const note = $("scorecard-note");
  const metric = metricForReference(state.selection.reference);
  const depthText = metric === "class4_rmsd" ? "" : ` at ${state.selection.depth}`;
  note.textContent = `${METRIC_PHRASE[metric]} vs ${REFERENCE_LABELS[state.selection.reference]} — ${state.selection.region}, day ${state.selection.lead}${depthText}. Mean ± 95% CI over forecast starts.`;

  const thead = $("scorecard").querySelector("thead");
  const headerRow = document.createElement("tr");
  headerRow.appendChild(headerCell("model", null, "left"));
  columns.forEach((column, index) => headerRow.appendChild(headerCell(columnHeader(column), index, "right", column.unit)));
  thead.replaceChildren(headerRow);

  let sortedOthers = others;
  if (state.sort.column !== null && columns[state.sort.column]) {
    const column = columns[state.sort.column];
    sortedOthers = others
      .map((slug) => ({ slug, value: cellFor(slug, column)?.mean ?? null }))
      .sort((a, b) => {
        if (a.value === null) return 1;
        if (b.value === null) return -1;
        return state.sort.direction * (a.value - b.value);
      })
      .map((entry) => entry.slug);
  }

  const tbody = $("scorecard").querySelector("tbody");
  tbody.replaceChildren();
  for (const slug of [...baselines, ...sortedOthers]) {
    tbody.appendChild(scorecardRow(slug, columns));
  }
  $("reset-sort").hidden = state.sort.column === null;
}

function headerCell(text, columnIndex, align, unit) {
  const th = document.createElement("th");
  th.style.textAlign = align;
  const label = document.createElement("span");
  label.textContent = text;
  th.appendChild(label);
  if (unit) {
    const unitSpan = document.createElement("span");
    unitSpan.className = "unit";
    unitSpan.textContent = unit;
    th.appendChild(unitSpan);
  }
  if (columnIndex !== null) {
    if (state.sort.column === columnIndex) {
      const marker = document.createElement("span");
      marker.className = "sort-marker";
      marker.textContent = state.sort.direction === 1 ? "▲" : "▼";
      th.appendChild(marker);
    }
    th.addEventListener("click", () => toggleSort(columnIndex));
    th.title = "Sort by this column (click to cycle ascending / descending / neutral)";
  } else {
    th.style.cursor = "default";
  }
  return th;
}

function toggleSort(columnIndex) {
  if (state.sort.column !== columnIndex) state.sort = { column: columnIndex, direction: 1 };
  else if (state.sort.direction === 1) state.sort.direction = -1;
  else state.sort = { column: null, direction: 0 };
  renderScorecard();
}

function scorecardRow(slug, columns) {
  const tr = document.createElement("tr");
  if (isBaseline(slug)) tr.className = "baseline";
  const nameCell = document.createElement("td");
  nameCell.textContent = challengerLabel(slug);
  tr.appendChild(nameCell);

  for (const column of columns) {
    const td = document.createElement("td");
    const cell = cellFor(slug, column);
    if (!cell) {
      td.innerHTML = '<span class="cell-empty">—</span>';
      tr.appendChild(td);
      continue;
    }
    const mean = document.createElement("span");
    mean.className = "cell-mean";
    mean.textContent = formatValue(cell.mean);
    td.appendChild(mean);
    if (cell.summary && Number.isFinite(cell.summary.ci_low) && Number.isFinite(cell.summary.ci_high)) {
      const half = (cell.summary.ci_high - cell.summary.ci_low) / 2;
      const ci = document.createElement("span");
      ci.className = "cell-ci";
      ci.textContent = `± ${formatValue(half)}`;
      td.appendChild(ci);
    }
    if (state.selection.showSkill && state.skillBaseline && !isBaseline(slug) && cell.summary) {
      const skillValue = cell.summary[`skill_vs_${state.skillBaseline}`];
      if (Number.isFinite(skillValue)) {
        const skill = document.createElement("span");
        skill.className = "cell-skill";
        skill.textContent = `skill ${(skillValue * 100).toFixed(0)}%`;
        td.appendChild(skill);
      }
    }
    tr.appendChild(td);
  }
  return tr;
}

// ---- plain-language summaries --------------------------------------------

function refreshSummaryControls() {
  const { region, reference, lead, year } = state.selection;
  const metric = metricForReference(reference);
  const variables = distinct(
    [...state.aggregated.values()]
      .filter((cell) => cell.region === region && cell.year === year && cell.metric === metric && cell.lead_day === lead &&
        (metric === "class4_rmsd" ? true : cell.reference === reference))
      .map((cell) => cell.variable),
  ).sort((a, b) => variableLabel(a).localeCompare(variableLabel(b)));
  if (!variables.includes(state.summarySelection.variable)) state.summarySelection.variable = variables[0];
  populateSelect($("summary-variable-select"), variables, state.summarySelection.variable, variableLabel);
  refreshSummaryDepthControl();
}

function summaryDepths() {
  const { region, reference, lead, year } = state.selection;
  const metric = metricForReference(reference);
  return sortedDepths(
    distinct(
      [...state.aggregated.values()]
        .filter((cell) => cell.region === region && cell.year === year && cell.metric === metric &&
          cell.lead_day === lead && cell.variable === state.summarySelection.variable &&
          (metric === "class4_rmsd" ? true : cell.reference === reference) && cell.depth)
        .map((cell) => cell.depth),
    ),
  );
}

function refreshSummaryDepthControl() {
  const depths = summaryDepths();
  const control = $("summary-depth-control");
  if (depths.length === 0) {
    control.hidden = true;
    state.summarySelection.depth = null;
    return;
  }
  control.hidden = false;
  if (!depths.includes(state.summarySelection.depth)) state.summarySelection.depth = depths[0];
  populateSelect($("summary-depth-select"), depths, state.summarySelection.depth);
}

function renderSummaries() {
  const { region, reference, lead, year } = state.selection;
  const metric = metricForReference(reference);
  const container = $("summary-cards");
  container.replaceChildren();
  const { baselines, others } = orderedChallengers();

  for (const slug of [...baselines, ...others]) {
    const key = identityKey({
      challenger: slug,
      year,
      region,
      metric,
      reference: metric === "class4_rmsd" ? "observations" : reference,
      variable: state.summarySelection.variable,
      depth: state.summarySelection.depth,
      lead_day: lead,
    });
    const cell = state.aggregated.get(key);
    if (!cell) continue;
    const summary = state.summaryByKey.get(key);
    container.appendChild(summaryCard(slug, cell, summary));
  }
  if (!container.hasChildNodes()) {
    const empty = document.createElement("p");
    empty.className = "section-note";
    empty.textContent = "No data for this selection.";
    container.appendChild(empty);
  }
}

function summaryCard(slug, cell, summary) {
  const card = document.createElement("div");
  card.className = "summary-card";
  const heading = document.createElement("h3");
  heading.textContent = challengerLabel(slug) + (isBaseline(slug) ? " (baseline)" : "");
  card.appendChild(heading);

  const sentence = document.createElement("p");
  const ciText =
    summary && Number.isFinite(summary.ci_low) && Number.isFinite(summary.ci_high)
      ? ` (95% CI ${formatValue(summary.ci_low)}–${formatValue(summary.ci_high)})`
      : "";
  sentence.innerHTML =
    `At day ${state.selection.lead}, ${challengerLabel(slug)}'s ${variableLabel(cell.variable)} ` +
    `${METRIC_PHRASE[cell.metric]} vs ${REFERENCE_LABELS[state.selection.reference]} is ` +
    `<span class="figure">${formatValue(cell.mean)} ${cell.unit}</span>${ciText}, ` +
    `over ${cell.n_starts} forecast starts.`;
  card.appendChild(sentence);
  return card;
}

// ---- boot ----------------------------------------------------------------

async function boot() {
  const status = $("status");
  try {
    const [rows, summaryRecords, challengers] = await Promise.all([
      loadParquet(DATA.parquet),
      loadJSON(DATA.summary, true),
      loadJSON(DATA.challengers, true),
    ]);

    state.aggregated = aggregateRows(rows);
    if (summaryRecords) {
      const indexed = indexSummary(summaryRecords);
      state.summaryByKey = indexed.byKey;
      state.skillBaseline = indexed.skillBaseline;
    }
    if (challengers) {
      for (const [slug, meta] of Object.entries(challengers)) {
        state.challengerMeta.set(slug, meta);
      }
    }

    buildControls();
    render();
    status.hidden = true;
    $("main").hidden = false;
  } catch (error) {
    status.className = "status error";
    status.textContent = `Could not load scores: ${error.message}`;
    console.error(error);
  }
}

boot();
