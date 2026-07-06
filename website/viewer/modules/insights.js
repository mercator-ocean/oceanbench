// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Loaders for the insight artifacts the viewer overlays and context rail read
// (contracts.md §4): eddy census (JSON, snake_case schema), realism spectra
// (JSON), the aggregated score summary (mean ± CI per lead), and Class-4 obs
// match-ups (parquet, read with the vendored hyparquet). Everything is fetched
// lazily and memoised by URL — a panel only pays for the overlay it turns on.

import { resolveViewerDataUrl } from "../config.js";

const INDEX_URL = resolveViewerDataUrl("./data/insights.json");
const jsonCache = new Map();
let indexPromise = null;

const CLASS4_WORKER_URL = new URL("./class4-worker.js", import.meta.url);
let class4Worker = null;
let class4RequestId = 0;
const class4Pending = new Map();
const class4ModeCache = new Map(); // resolvedUrl -> Promise<boolean targeted>
const class4LegacyCache = new Map(); // resolvedUrl -> Promise<result>
const class4TargetedCache = new Map(); // `${resolvedUrl}|${start}|${lead}` -> Promise<result>

export function loadInsightIndex() {
  if (!indexPromise) indexPromise = fetchJSON(INDEX_URL).catch(() => null);
  return indexPromise;
}

async function fetchJSON(url) {
  const resolvedUrl = resolveViewerDataUrl(url);
  if (jsonCache.has(resolvedUrl)) return jsonCache.get(resolvedUrl);
  const promise = (async () => {
    const response = await fetch(resolvedUrl);
    if (!response.ok) throw new Error(`${resolvedUrl} -> HTTP ${response.status}`);
    return response.json();
  })();
  jsonCache.set(resolvedUrl, promise);
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

/**
 * Load the Class-4 match-up overlay data. When the parquet follows the match-up
 * contract (one (start_date, lead_day) pair per row group, with statistics), only
 * the row group(s) for the requested `startDate`/`leadDay` are fetched — complete
 * rows, no sampling. Legacy files fall back to whole-file or scattered sampling.
 *
 * Returns `{ targeted, rows, total, sampled }` (or null on failure). In targeted
 * mode `rows` is exactly the selected pair; in legacy mode it is the (possibly
 * sampled) whole set as before. `byteLength` may come from the sibling manifest's
 * `bytes` field to skip a HEAD request.
 */
export async function loadClass4(url, { byteLength, rowGroupIndex, sampleVariables, startDate, leadDay } = {}) {
  if (!url) return null;
  const resolvedUrl = resolveViewerDataUrl(url);
  try {
    const targeted = await probeClass4Mode(resolvedUrl, byteLength);
    if (!targeted) {
      if (!class4LegacyCache.has(resolvedUrl)) {
        class4LegacyCache.set(
          resolvedUrl,
          requestClass4Worker({ op: "legacy", url: resolvedUrl, byteLength, rowGroupIndex, sampleVariables }).then(
            (payload) => normalizeClass4Payload(payload),
          ),
        );
      }
      return await class4LegacyCache.get(resolvedUrl);
    }
    const key = `${resolvedUrl}|${startDate ?? ""}|${leadDay ?? ""}`;
    if (!class4TargetedCache.has(key)) {
      class4TargetedCache.set(
        key,
        requestClass4Worker({ op: "targeted", url: resolvedUrl, byteLength, startDate, leadDay }).then((payload) =>
          normalizeClass4Payload(payload),
        ),
      );
    }
    return await class4TargetedCache.get(key);
  } catch {
    return null;
  }
}

function normalizeClass4Payload(payload) {
  const rows = payload && payload.rows ? payload.rows : [];
  return {
    targeted: Boolean(payload && payload.targeted),
    rows,
    total: payload && Number.isFinite(payload.total) ? payload.total : rows.length,
    sampled: Boolean(payload && payload.sampled),
  };
}

function probeClass4Mode(resolvedUrl, byteLength) {
  if (!class4ModeCache.has(resolvedUrl)) {
    class4ModeCache.set(
      resolvedUrl,
      requestClass4Worker({ op: "probe", url: resolvedUrl, byteLength }).then((payload) => Boolean(payload.targeted)),
    );
  }
  return class4ModeCache.get(resolvedUrl);
}

function ensureClass4Worker() {
  if (class4Worker) return class4Worker;
  class4Worker = new Worker(CLASS4_WORKER_URL, { type: "module" });
  class4Worker.addEventListener("message", (event) => {
    const { id, error } = event.data;
    const pending = class4Pending.get(id);
    if (!pending) return;
    class4Pending.delete(id);
    if (error) pending.reject(new Error(error));
    else pending.resolve(event.data);
  });
  class4Worker.addEventListener("error", (event) => {
    for (const pending of class4Pending.values()) pending.reject(new Error(event.message || "Class-4 worker failed"));
    class4Pending.clear();
  });
  return class4Worker;
}

function requestClass4Worker(message) {
  const worker = ensureClass4Worker();
  const id = ++class4RequestId;
  const promise = new Promise((resolve, reject) => class4Pending.set(id, { resolve, reject }));
  worker.postMessage({ id, ...message });
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
  const requestedLead = leadDay == null ? null : Number(leadDay);
  const requestedStart = startDate || null;
  const matched = [];
  for (const row of rows) {
    if (row.variable !== variable) continue;
    if (depthBin && row.depth_bin !== depthBin) continue;
    if (requestedLead !== null && Number(row.lead_day) !== requestedLead) continue;
    if (requestedStart && formatClass4Date(row.start_date) !== requestedStart) continue;
    matched.push(row);
  }
  if (matched.length <= limit) return matched;
  const stride = Math.ceil(matched.length / limit);
  const thinned = [];
  for (let i = 0; i < matched.length; i += stride) thinned.push(matched[i]);
  return thinned;
}

function formatClass4Date(value) {
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  return String(value).slice(0, 10);
}
