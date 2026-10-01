// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// The viewer's Period control: which forecast starts a period covers, and the rail scores
// recomputed over those starts from the per-start rows of scores.parquet (read with
// modules/scores-periods.js, the Scores page's aggregation).
//
// Loaded on demand (the per-start reader pulls in the parquet decoder), so a viewer that
// never picks a period does not pay for it; the vocabulary lives in period-range.js.
//
// The whole year never comes through here: the rail keeps reading the published
// scores-summary.json and rmsd-by-depth.json for it. A period reuses those artifacts only
// for their shape (which rows, which depth bins, which leads) and recomputes every value.
//
// Pure module with no DOM access, so the parity check in qa/ runs it under Node.

import { aggregatePeriod, identityKey, periodStartIndices, readPerStartScores } from "./scores-periods.js";

/**
 * Lead-curve rows over a period: the published summary rows of one series, aggregated over
 * the period's starts exactly as the Scores page does (pooled Class 4 RMSE, bootstrap band
 * over the starts). Rows the period has no value for are dropped.
 */
export function periodLeadRows(perStart, rows, startIndices) {
  return aggregatePeriod(perStart, rows, startIndices).filter((row) => Number.isFinite(row.mean));
}

// The Class 4 RMSE and bias of one (variable, depth bin, lead) over a set of starts:
// RMSE pooled as sqrt(sum(v^2 n) / sum(n)), bias the n-weighted mean, n the match-up count.
function pooledCell(perStart, rmsdKey, biasKey, keep) {
  const rmsdGroup = perStart.groups.get(rmsdKey);
  if (!rmsdGroup) return { rmsd: null, bias: null, n: null };
  let squares = 0;
  let count = 0;
  rmsdGroup.starts.forEach((start, index) => {
    const value = rmsdGroup.values[index];
    if (!keep[start] || !Number.isFinite(value)) return;
    squares += value * value * rmsdGroup.counts[index];
    count += rmsdGroup.counts[index];
  });
  if (!(count > 0)) return { rmsd: null, bias: null, n: null };
  let bias = null;
  const biasGroup = perStart.groups.get(biasKey);
  if (biasGroup) {
    let total = 0;
    let weight = 0;
    biasGroup.starts.forEach((start, index) => {
      const value = biasGroup.values[index];
      if (!keep[start] || !Number.isFinite(value)) return;
      total += value * biasGroup.counts[index];
      weight += biasGroup.counts[index];
    });
    if (weight > 0) bias = total / weight;
  }
  return { rmsd: Math.sqrt(squares / count), bias, n: count };
}

/**
 * An rmsd-by-depth artifact over a period: the published year artifact's variables, depth
 * bins and leads, every cell recomputed from the per-start Class 4 rows of `challenger`,
 * `year` and `region` over the starts in `startIndices`.
 */
export function periodDepthArtifact(perStart, yearArtifact, { challenger, year, region }, startIndices) {
  const keep = new Uint8Array(perStart.starts.length);
  for (const index of startIndices) keep[index] = 1;
  const key = (metric, variable, depth, lead) =>
    identityKey({ challenger, year, region, metric, reference: "observations", variable, depth, lead_day: lead, band: null, polarity: null });
  const variables = {};
  for (const [name, entry] of Object.entries((yearArtifact && yearArtifact.variables) || {})) {
    if (!Array.isArray(entry.depth_bins) || !Array.isArray(entry.leads)) continue;
    const cells = entry.depth_bins.map((bin) =>
      entry.leads.map((lead) => pooledCell(perStart, key("class4_rmsd", name, bin, lead), key("class4_bias", name, bin, lead), keep)),
    );
    variables[name] = {
      depth_bins: entry.depth_bins,
      leads: entry.leads,
      rmsd: cells.map((row) => row.map((cell) => cell.rmsd)),
      bias: cells.map((row) => row.map((cell) => cell.bias)),
      n: cells.map((row) => row.map((cell) => cell.n)),
    };
  }
  return { variables };
}

export { periodStartIndices, readPerStartScores };
