// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Reading the published scores summary from the viewer's point of view: which column of a
// summary row carries the number, which single depth key a variable's scores live under, and
// how the published rows of one selection become a lead series.

import { class4DepthBin } from "./class4-index.js";

// One published row per lead day: its mean and bootstrap CI are used as published. Rows are
// never averaged here, and a CI is never recomputed in the browser.
export function aggregateLeadSeries(grouped) {
  const series = new Map();
  for (const [key, rows] of grouped) {
    const byLead = new Map();
    for (const row of rows) {
      const leadDay = Number(row.lead_day);
      const value = scoreValue(row);
      if (!Number.isFinite(leadDay) || !Number.isFinite(value) || byLead.has(leadDay)) continue;
      byLead.set(leadDay, {
        lead_day: leadDay,
        mean: value,
        ci_low: Number.isFinite(row.ci_low) ? row.ci_low : value,
        ci_high: Number.isFinite(row.ci_high) ? row.ci_high : value,
      });
    }
    const aggregated = [...byLead.values()];
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

// Exactly one published depth key per variable, the same bin the Class-4 overlay and the
// year artifacts use, so every obs-based view of a variable reads one obs population.
export function scoreDepthKeys(entry) {
  const key = class4DepthBin(entry) || mapDepthToScoreDepth(entry);
  return key ? [key] : [];
}

// Human label for a published obs depth bin.
export function depthBinLabel(entry, key) {
  if (key === "surface" && entry && entry.standard_name === "sea_water_potential_temperature") return "obs shallower than 1 m";
  if (key === "0-5m") return "obs 0-5 m";
  if (key === "15m") return "obs at 15 m";
  return key ? `obs ${key}` : "";
}
