// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Reading the published scores summary from the viewer's point of view: which column of a
// summary row carries the number, which depth key a variable's scores live under, and how
// several rows for one lead day collapse into a mean with a confidence interval. The
// aggregation is here rather than in scores-data.js because the scores page reads rows the
// publisher already aggregated, while the viewer regroups them per panel selection.

import { class4DepthBin } from "./class4-index.js";

export function aggregateLeadSeries(grouped) {
  const series = new Map();
  for (const [key, rows] of grouped) {
    const byLead = new Map();
    for (const row of rows) {
      const leadDay = Number(row.lead_day);
      const value = scoreValue(row);
      if (!Number.isFinite(leadDay) || !Number.isFinite(value)) continue;
      if (!byLead.has(leadDay)) byLead.set(leadDay, []);
      byLead.get(leadDay).push({ row, value });
    }
    const aggregated = [];
    for (const [leadDay, values] of byLead) {
      const mean = values.reduce((total, item) => total + item.value, 0) / values.length;
      let ciLow = mean;
      let ciHigh = mean;
      if (values.length === 1) {
        const row = values[0].row;
        ciLow = Number.isFinite(row.ci_low) ? row.ci_low : mean;
        ciHigh = Number.isFinite(row.ci_high) ? row.ci_high : mean;
      } else {
        const variance = values.reduce((total, item) => total + (item.value - mean) ** 2, 0) / (values.length - 1);
        const error = 1.96 * Math.sqrt(variance / values.length);
        ciLow = mean - error;
        ciHigh = mean + error;
      }
      aggregated.push({ lead_day: leadDay, mean, ci_low: ciLow, ci_high: ciHigh });
    }
    if (aggregated.length) series.set(key, aggregated.sort((a, b) => a.lead_day - b.lead_day));
  }
  return series;
}

export function scoreValue(row) {
  for (const key of ["mean", "value", "rmse", "rmsd", "score"]) {
    const value = Number(row[key]);
    if (Number.isFinite(value)) return value;
  }
  return NaN;
}

export function mapDepthToScoreDepth(entry) {
  if (entry.standard_name.includes("velocity") && entry.depth === "15m") return "15m";
  if (entry.depth === "surface") return "surface";
  return entry.depth;
}

export function scoreDepthKeys(entry) {
  const keys = [];
  const class4Bin = class4DepthBin(entry);
  if (class4Bin) keys.push(class4Bin);
  const legacyDepth = mapDepthToScoreDepth(entry);
  if (legacyDepth && !keys.includes(legacyDepth)) keys.push(legacyDepth);
  return keys;
}
