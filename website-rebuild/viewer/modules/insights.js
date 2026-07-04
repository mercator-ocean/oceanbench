// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Loaders for the insight artifacts the viewer overlays and context rail read
// (contracts.md §4): eddy census (JSON, snake_case schema), realism spectra
// (JSON), the aggregated score summary (mean ± CI per lead), and Class-4 obs
// match-ups (parquet, read with the vendored hyparquet). Everything is fetched
// lazily and memoised by URL — a panel only pays for the overlay it turns on.

import { parquetReadObjects } from "../vendor/hyparquet/hyparquet.min.js";

const INDEX_URL = "./data/insights.json";
const jsonCache = new Map();
const parquetCache = new Map();
let indexPromise = null;

export function loadInsightIndex() {
  if (!indexPromise) indexPromise = fetchJSON(INDEX_URL).catch(() => null);
  return indexPromise;
}

async function fetchJSON(url) {
  if (jsonCache.has(url)) return jsonCache.get(url);
  const promise = (async () => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url} -> HTTP ${response.status}`);
    return response.json();
  })();
  jsonCache.set(url, promise);
  return promise;
}

export async function loadEddies(url) {
  if (!url) return null;
  return fetchJSON(url).catch(() => null);
}

export async function loadSpectra(url) {
  if (!url) return null;
  return fetchJSON(url).catch(() => null);
}

export async function loadScoresSummary(index) {
  if (!index || !index.scores_summary) return [];
  return fetchJSON(index.scores_summary).catch(() => []);
}

/** Read the whole Class-4 match-up parquet once (decimated for the viewer). */
export async function loadClass4(url) {
  if (!url) return null;
  if (parquetCache.has(url)) return parquetCache.get(url);
  const promise = (async () => {
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
  })().catch(() => null);
  parquetCache.set(url, promise);
  return promise;
}

/** Insight URLs for a (dataset slug, region) pair, or empty object. */
export function insightsFor(index, slug, region) {
  return (index && index.datasets && index.datasets[slug] && index.datasets[slug][region]) || {};
}

/** Eddy frame for the requested reference and the available lead nearest `leadDay`. */
export function eddyFrame(eddies, reference, leadDay) {
  if (!eddies || !Array.isArray(eddies.references)) return null;
  const entry = eddies.references.find((candidate) => candidate.reference === reference) || eddies.references[0];
  if (!entry || !entry.frames.length) return null;
  let best = entry.frames[0];
  for (const frame of entry.frames) {
    if (Math.abs(frame.lead_day - leadDay) < Math.abs(best.lead_day - leadDay)) best = frame;
  }
  return { frame: best, reference: entry.reference, variable: eddies.variable };
}

/** Spectra entry for the requested variable/reference and available lead nearest leadDay. */
export function spectraEntry(spectra, variable, reference, leadDay) {
  if (!spectra || !Array.isArray(spectra.entries)) return null;
  const candidates = spectra.entries.filter(
    (entry) => entry.variable === variable && (!reference || entry.reference === reference),
  );
  if (!candidates.length) return null;
  let best = candidates[0];
  for (const entry of candidates) {
    if (Math.abs(entry.lead_day - leadDay) < Math.abs(best.lead_day - leadDay)) best = entry;
  }
  return best;
}

/**
 * Class-4 points for the active (variable, depth bin, lead day, start date),
 * spatially thinned to at most `limit` points so a whole altimeter month does not
 * choke the overlay at low zoom. Thinning is a deterministic stride (density-manage
 * §4), refined by the panel as the user zooms into fewer visible points.
 */
export function class4Points(rows, { variable, depthBin, leadDay, startDate, limit = 4000 }) {
  if (!rows) return [];
  const matched = [];
  for (const row of rows) {
    if (row.variable !== variable) continue;
    if (depthBin && row.depth_bin !== depthBin) continue;
    if (row.lead_day !== leadDay) continue;
    if (startDate && row.start_date !== startDate) continue;
    matched.push(row);
  }
  if (matched.length <= limit) return matched;
  const stride = Math.ceil(matched.length / limit);
  const thinned = [];
  for (let i = 0; i < matched.length; i += stride) thinned.push(matched[i]);
  return thinned;
}

/** Trajectories are not yet produced (contracts.md §4 lists the kind; no artifact exists). */
export async function loadTrajectories(index, slug, region) {
  const urls = insightsFor(index, slug, region);
  if (!urls.trajectories) return { available: false, reason: "no trajectories artifact produced yet (stub loader)" };
  return fetchJSON(urls.trajectories)
    .then((data) => ({ available: true, data }))
    .catch(() => ({ available: false, reason: "trajectories artifact failed to load" }));
}
