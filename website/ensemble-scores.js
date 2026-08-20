// SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

const BLOCK_CONTAINERS = {
  "observations_rmsd": "ensemble-observations-rmsd",
  "gridded_rmsd": "ensemble-gridded-rmsd",
  "observations_crps": "ensemble-observations-crps",
  "observations_spread_error_ratio": "ensemble-observations-ssr",
  "gridded_crps": "ensemble-gridded-crps",
  "gridded_spread_error_ratio": "ensemble-gridded-ssr",
};

const REDUCED_START_MARKER = "*";

function readEnsembleScores() {
  const element = document.getElementById("ensemble-scores-data");
  if (!element) return null;
  return JSON.parse(element.textContent);
}

function formatValue(row, value) {
  if (value === null || value === undefined) return "";
  return value.toFixed(row.decimals);
}

function systemClass(row, systems) {
  return systems[row.system] && systems[row.system].kind === "Deterministic" ? ' class="baseline-row"' : "";
}

function buildHeader(leadDays) {
  let thead = "<thead>";
  thead += '<tr><th class="model-col">Variable</th><th class="model-col">Depth</th>';
  thead += '<th class="model-col">System</th><th class="model-col">Unit</th>';
  thead += `<th class="var-header" colspan="${leadDays.length}">Lead day</th></tr>`;
  thead += '<tr><th class="model-col lead-day-label" colspan="4">Lead days</th>';
  for (const day of leadDays) {
    thead += `<th class="lead-day">${day}</th>`;
  }
  thead += "</tr></thead>";
  return thead;
}

function buildBody(block, systems) {
  let tbody = "<tbody>";
  let previousVariable = null;
  for (const row of block.rows) {
    if (row.variable !== previousVariable) {
      tbody += `<tr class="depth-separator"><th class="depth-separator-cell">${row.variable}</th>`;
      tbody += `<td colspan="${3 + block.lead_days.length}" style="border: none;"></td></tr>`;
    }
    previousVariable = row.variable;
    tbody += `<tr${systemClass(row, systems)}>`;
    tbody += `<th class="model-col">${row.variable}</th>`;
    tbody += `<td class="score-value-cell">${row.depth_band}</td>`;
    tbody += `<td class="score-value-cell">${row.system_label}</td>`;
    tbody += `<td class="score-value-cell">${row.unit}</td>`;
    for (let index = 0; index < block.lead_days.length; index++) {
      const day = block.lead_days[index];
      const value = row.values[index];
      const marker = row.reduced_start_leads.includes(day) ? REDUCED_START_MARKER : "";
      const title = value === null || value === undefined
        ? ""
        : `${row.variable}, ${row.depth_band}, ${row.system_label}, lead day ${day}`;
      tbody += `<td class="score-value-cell" title="${title}">${formatValue(row, value)}${marker}</td>`;
    }
    tbody += "</tr>";
  }
  tbody += "</tbody>";
  return tbody;
}

function renderBlock(block, systems, containerId) {
  const container = document.getElementById(containerId);
  if (!container || !block) return;
  let markup = `<p class="ensemble-table-note">${block.note}</p>`;
  markup += '<div class="score-table-wrapper"><table class="score-table depth-table">';
  markup += buildHeader(block.lead_days);
  markup += buildBody(block, systems);
  markup += "</table></div>";
  container.innerHTML = markup;
}

function renderEnsembleScores() {
  const scores = readEnsembleScores();
  if (!scores) return;
  for (const [blockKey, containerId] of Object.entries(BLOCK_CONTAINERS)) {
    renderBlock(scores.blocks[blockKey], scores.systems, containerId);
  }
}

document.addEventListener("DOMContentLoaded", renderEnsembleScores);
