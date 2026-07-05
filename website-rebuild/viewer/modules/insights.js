// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Loaders for the insight artifacts the viewer overlays and context rail read
// (contracts.md §4): eddy census (JSON, snake_case schema), realism spectra
// (JSON), the aggregated score summary (mean ± CI per lead), and Class-4 obs
// match-ups (parquet, read with the vendored hyparquet). Everything is fetched
// lazily and memoised by URL — a panel only pays for the overlay it turns on.

import { parquetReadObjects, parquetMetadataAsync } from "../vendor/hyparquet/hyparquet.min.js";

const INDEX_URL = "./data/insights.json";
const jsonCache = new Map();
const parquetCache = new Map();
let indexPromise = null;

// A whole-file fetch is only acceptable for the small regional match-up parquets
// (the IBI file is ~0.9 MB). The global match-up parquets are multi-gigabyte
// (glonet/global and xihe/global are ~7.5 GB / 306M rows) and MUST NOT be pulled
// whole into a browser: they are streamed with HTTP Range requests, reading the
// footer then a bounded number of leading row groups for a sampled overlay.
const WHOLE_FILE_MAX_BYTES = 50 * 1024 * 1024;
// Byte budget for the sampled read of a large file. Row groups in the global
// parquets are ~35 MB each; a ~40 MB budget selects the first row group (~1M
// rows) and stops, keeping every individual Range request (per column chunk,
// <10 MB here) and the total download far below the 64 MB cap a browser needs.
const RANGE_BYTE_BUDGET = 40 * 1024 * 1024;

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

/** An AsyncBuffer for hyparquet backed by HTTP Range requests (no whole-file fetch). */
function httpRangeAsyncBuffer(url, byteLength) {
  return {
    byteLength,
    async slice(start, end) {
      const last = (end ?? byteLength) - 1;
      const response = await fetch(url, { headers: { Range: `bytes=${start}-${last}` } });
      if (response.status !== 206 && response.status !== 200) {
        throw new Error(`${url} range ${start}-${last} -> HTTP ${response.status}`);
      }
      return response.arrayBuffer();
    },
  };
}

async function resolveByteLength(url, hint) {
  if (Number.isFinite(hint) && hint > 0) return hint;
  const response = await fetch(url, { method: "HEAD" });
  if (!response.ok) throw new Error(`${url} HEAD -> HTTP ${response.status}`);
  const length = Number(response.headers.get("content-length"));
  if (!Number.isFinite(length) || length <= 0) throw new Error(`${url} HEAD returned no content-length`);
  return length;
}

/**
 * Stream a large match-up parquet with Range requests: read the footer for
 * metadata, then read only the leading row groups that fit the byte budget. The
 * returned array carries `.sampled = true` so the overlay can note it shows a
 * subsample rather than all rows.
 */
async function readClass4Sampled(url, byteLength) {
  const file = httpRangeAsyncBuffer(url, byteLength);
  const metadata = await parquetMetadataAsync(file);
  const rowGroups = metadata.row_groups || [];
  let remainingBudget = RANGE_BYTE_BUDGET;
  let rowEnd = 0;
  for (const group of rowGroups) {
    const groupBytes = Number(group.total_byte_size) || 0;
    if (rowEnd > 0 && groupBytes > remainingBudget) break;
    remainingBudget -= groupBytes;
    rowEnd += Number(group.num_rows);
  }
  const rows = await parquetReadObjects({ file, metadata, rowStart: 0, rowEnd, rowFormat: "object" });
  if (rowEnd < Number(metadata.num_rows)) rows.sampled = true;
  return rows;
}

/**
 * Load the Class-4 match-up parquet for the overlay. Small regional files are read
 * whole; large global files (multi-GB) are streamed with Range requests and
 * sampled (see `readClass4Sampled`). `byteLength` may be passed from the sibling
 * manifest's `bytes` field to skip a HEAD request.
 */
export async function loadClass4(url, { byteLength } = {}) {
  if (!url) return null;
  if (parquetCache.has(url)) return parquetCache.get(url);
  const promise = (async () => {
    const size = await resolveByteLength(url, byteLength);
    if (size > WHOLE_FILE_MAX_BYTES) return readClass4Sampled(url, size);
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
