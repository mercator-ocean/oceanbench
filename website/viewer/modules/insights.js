// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Loaders for the insight artifacts the viewer overlays and context rail read
// (contracts.md §4): eddy census (JSON, snake_case schema), realism spectra
// (JSON), the aggregated score summary (mean ± CI per lead), and Class-4 obs
// match-ups (parquet, read with the vendored hyparquet). Everything is fetched
// lazily and memoised by URL — a panel only pays for the overlay it turns on.

import { resolveViewerDataUrl } from "../config.js";
import { class4AbsoluteError } from "./overlays.js";

// Resolved lazily, not at import time: the data root can still be rewritten by the
// optional viewer-config.json that boot awaits before the first fetch.
const INDEX_PATH = "./data/insights.json";
const jsonCache = new Map();
let indexPromise = null;

const CLASS4_WORKER_URL = new URL("./class4-worker.js", import.meta.url);
let class4Worker = null;
// Set once the worker has died; every later request rejects with it rather than posting
// into the void (see the error listener in ensureClass4Worker).
let class4WorkerFailure = null;
let class4RequestId = 0;
const class4Pending = new Map();
const class4TargetedCache = new Map(); // `${resolvedUrl}|${start}|${lead}` -> Promise<result>
// Prefetches still in flight, `cache key -> worker request id`, so they can be called off.
const class4PrefetchRequests = new Map();

export function loadInsightIndex() {
  if (!indexPromise) indexPromise = fetchJSON(resolveViewerDataUrl(INDEX_PATH)).catch(() => null);
  return indexPromise;
}

async function fetchJSON(url) {
  const resolvedUrl = resolveViewerDataUrl(url);
  if (jsonCache.has(resolvedUrl)) return jsonCache.get(resolvedUrl);
  const promise = (async () => {
    const response = await fetch(resolvedUrl, { cache: "no-cache" });
    if (!response.ok) throw new Error(`${resolvedUrl} -> HTTP ${response.status}`);
    return response.json();
  })();
  jsonCache.set(resolvedUrl, promise);
  return promise;
}

// Where an eddy artifact was fetched from, so the per-lead sidecars of the index format can
// be resolved against the same directory. Symbol-keyed so it never lands in JSON consumers.
const EDDY_SOURCE_URL = Symbol("eddySourceUrl");
// `${indexUrl}|${lead}` -> Promise<frame|null>, so scrubbing back to an already-read lead
// (or the second panel showing the same dataset) never refetches a sidecar.
const eddyLeadFrameCache = new Map();

/**
 * Load a dataset's eddy-census artifact. Two published shapes are supported and the
 * difference is invisible to callers:
 *   - index (current): `{ ...metadata, leads: [{ lead_day, file }] }`, each `file` a sidecar
 *     next to the index holding that one lead's `frame`.
 *   - inline (legacy, still served by the carried datasets): `{ ...metadata, frames: [...] }`.
 */
export async function loadEddies(url) {
  if (!url) return null;
  const resolvedUrl = resolveViewerDataUrl(url);
  const data = await fetchJSON(resolvedUrl).catch(() => null);
  if (data && typeof data === "object" && !data[EDDY_SOURCE_URL]) data[EDDY_SOURCE_URL] = resolvedUrl;
  return data;
}

/** The index format's lead entries, or null when the artifact carries inline frames. */
function eddyLeadEntries(eddies) {
  if (!eddies || !Array.isArray(eddies.leads)) return null;
  return eddies.leads.filter((entry) => entry && entry.file && Number.isFinite(Number(entry.lead_day)));
}

/** Fetch (once per index + lead) the sidecar frame for the available lead nearest `leadDay`. */
function eddyLeadFrame(eddies, entries, leadDay) {
  let best = null;
  for (const entry of entries) {
    if (!best || Math.abs(Number(entry.lead_day) - leadDay) < Math.abs(Number(best.lead_day) - leadDay)) best = entry;
  }
  if (!best) return Promise.resolve(null);
  const indexUrl = eddies[EDDY_SOURCE_URL] || "";
  const key = `${indexUrl}|${best.lead_day}`;
  if (!eddyLeadFrameCache.has(key)) {
    const sidecarUrl = `${indexUrl.slice(0, indexUrl.lastIndexOf("/") + 1)}${best.file}`;
    eddyLeadFrameCache.set(
      key,
      fetchJSON(sidecarUrl)
        .then((payload) => (payload && payload.frame ? payload.frame : null))
        .catch(() => null),
    );
  }
  return eddyLeadFrameCache.get(key);
}

export async function loadSpectra(url) {
  if (!url) return null;
  return fetchJSON(url).catch(() => null);
}

export async function loadRmsdByDepth(url) {
  if (!url) return null;
  return fetchJSON(url).catch(() => null);
}

/**
 * Vertical RMSD profile for one variable at the lead day nearest `leadDay`, read from a
 * rmsd-by-depth artifact (schema_version 1). Returns { bins: [{ label, rmsd, bias, n }],
 * lead } ordered surface→deep, or null when the variable/lead carries no finite value.
 * `variableName` is the observation standard name (e.g. sea_water_potential_temperature).
 */
export function rmsdDepthProfile(data, variableName, leadDay) {
  const entry = data && data.variables && data.variables[variableName];
  if (!entry || !Array.isArray(entry.depth_bins) || !Array.isArray(entry.leads) || !entry.leads.length) return null;
  const leadIndex = nearestIndex(entry.leads, leadDay);
  if (leadIndex < 0) return null;
  const valueAt = (matrix, row) => {
    const cell = Array.isArray(matrix) && Array.isArray(matrix[row]) ? matrix[row][leadIndex] : null;
    return cell == null || !Number.isFinite(Number(cell)) ? null : Number(cell);
  };
  const bins = entry.depth_bins.map((label, row) => ({
    label,
    rmsd: valueAt(entry.rmsd, row),
    bias: valueAt(entry.bias, row),
    n: valueAt(entry.n, row),
  }));
  if (!bins.some((bin) => Number.isFinite(bin.rmsd))) return null;
  return { bins, lead: entry.leads[leadIndex] };
}

/**
 * Lead-independent RMSD extent for the depth-profile chart: the largest finite RMSD over
 * EVERY depth bin and EVERY lead of one variable. The profile axis is bounded by this, so
 * scrubbing the lead moves the profile within a constant frame instead of rescaling it.
 * Pure function of the loaded artifact, so it only changes with dataset/variable/region.
 */
export function rmsdDepthProfileMax(data, variableName) {
  const entry = data && data.variables && data.variables[variableName];
  if (!entry || !Array.isArray(entry.rmsd)) return 0;
  let maximum = 0;
  for (const row of entry.rmsd) {
    if (!Array.isArray(row)) continue;
    for (const cell of row) {
      const value = Number(cell);
      if (cell != null && Number.isFinite(value) && value > maximum) maximum = value;
    }
  }
  return maximum;
}

function nearestIndex(leads, leadDay) {
  if (!Array.isArray(leads) || !leads.length) return -1;
  let bestIndex = 0;
  for (let index = 1; index < leads.length; index += 1) {
    if (Math.abs(leads[index] - leadDay) < Math.abs(leads[bestIndex] - leadDay)) bestIndex = index;
  }
  return bestIndex;
}

export async function loadScoresSummary(index) {
  if (!index || !index.scores_summary) return [];
  return fetchJSON(index.scores_summary).catch(() => []);
}

/**
 * Load the Class-4 match-up overlay data. The served parquet follows the match-up
 * contract (one (start_date, lead_day) pair per row group, with statistics), so only
 * the row group(s) for the requested `startDate`/`leadDay` are fetched — complete
 * rows, no sampling.
 *
 * Returns `{ targeted: true, rows, total }` where `rows` is exactly the selected
 * pair. `byteLength` may come from the sibling manifest's `bytes` field to skip a
 * HEAD request, but callers should omit stale hints.
 */
export async function loadClass4(url, { byteLength, startDate, leadDay, variables, onProgress, quiet } = {}) {
  if (!url) throw new Error("Class-4 URL is missing");
  const key = class4CacheKey(url, { startDate, leadDay, variables });
  if (!class4TargetedCache.has(key)) {
    const resolvedUrl = resolveViewerDataUrl(url);
    const request = requestClass4Worker(
      { op: "targeted", url: resolvedUrl, byteLength, startDate, leadDay, variables, quiet: Boolean(quiet) },
      onProgress,
    );
    if (quiet) class4PrefetchRequests.set(key, request.id);
    const promise = request.promise.then((payload) => normalizeClass4Payload(payload));
    class4TargetedCache.set(key, promise);
    // A cancelled prefetch, or any failure, must leave no rejected promise behind for the
    // next request for that pair to inherit.
    promise
      .catch(() => {
        if (class4TargetedCache.get(key) === promise) class4TargetedCache.delete(key);
      })
      .finally(() => class4PrefetchRequests.delete(key));
  }
  return await class4TargetedCache.get(key);
}

// Row groups are variable-partitioned within a (start, lead) block, so the requested
// variables refine which groups are fetched, so they belong in the targeted cache key.
function class4CacheKey(url, { startDate, leadDay, variables }) {
  const variableKey = Array.isArray(variables) && variables.length ? [...variables].sort().join(",") : "";
  return `${resolveViewerDataUrl(url)}|${startDate ?? ""}|${leadDay ?? ""}|${variableKey}`;
}

/**
 * Read a (start, lead) pair into the same cache `loadClass4` reads from, without touching
 * the progress bar: the bar belongs to the load the user is waiting on. Failures (including
 * cancellation) resolve to null, since nothing is waiting on the result.
 */
export function prefetchClass4(url, options = {}) {
  return loadClass4(url, { ...options, quiet: true, onProgress: undefined }).catch(() => null);
}

/**
 * Call off every prefetch still in flight except the pair described by `keep` (the load the
 * user just asked for, if that is what was being prefetched), so prefetching never competes
 * with a user-initiated read for the origin's connections.
 */
export function cancelClass4Prefetches(keep) {
  const keepKey = keep && keep.url ? class4CacheKey(keep.url, keep) : null;
  const cancelled = [];
  for (const [key, id] of class4PrefetchRequests) {
    if (key === keepKey) continue;
    cancelled.push(id);
    class4PrefetchRequests.delete(key);
    class4TargetedCache.delete(key);
  }
  if (cancelled.length && class4Worker) class4Worker.postMessage({ op: "cancel", cancel: cancelled });
}

function normalizeClass4Payload(payload) {
  const rows = payload && payload.rows ? payload.rows : [];
  return {
    targeted: Boolean(payload && payload.targeted),
    rows,
    total: payload && Number.isFinite(payload.total) ? payload.total : rows.length,
  };
}

function ensureClass4Worker() {
  if (class4Worker) return class4Worker;
  class4Worker = new Worker(CLASS4_WORKER_URL, { type: "module" });
  class4Worker.addEventListener("message", (event) => {
    const { id, error, progress } = event.data;
    const pending = class4Pending.get(id);
    if (!pending) return;
    // Byte counters for the loading bar: the request is still running, so keep it pending.
    if (progress) {
      if (pending.onProgress) pending.onProgress(progress);
      return;
    }
    class4Pending.delete(id);
    if (error) pending.reject(new Error(error));
    else pending.resolve(event.data);
  });
  class4Worker.addEventListener("error", (event) => {
    // A worker that fails to start (bad MIME on its module import, syntax error) never
    // answers a postMessage, so every later request would hang on a promise that can
    // never settle and the panel would spin forever. Reject what is pending, drop the
    // dead worker and remember why, so subsequent requests fail fast with that reason
    // and the caller can render an error state instead of a spinner.
    const reason = event.message || "Class-4 worker failed to start";
    class4WorkerFailure = reason;
    class4Worker = null;
    for (const pending of class4Pending.values()) pending.reject(new Error(reason));
    class4Pending.clear();
  });
  return class4Worker;
}

// Returns the worker request id alongside its promise, so a prefetch can be cancelled by id.
function requestClass4Worker(message, onProgress) {
  if (class4WorkerFailure) return { id: 0, promise: Promise.reject(new Error(class4WorkerFailure)) };
  const worker = ensureClass4Worker();
  const id = ++class4RequestId;
  const promise = new Promise((resolve, reject) => class4Pending.set(id, { resolve, reject, onProgress }));
  worker.postMessage({ id, ...message });
  return { id, promise };
}

/** Insight URLs for a (dataset slug, region) pair, or empty object. */
export function insightsFor(index, slug, region) {
  return (index && index.datasets && index.datasets[slug] && index.datasets[slug][region]) || {};
}

function nearestLeadFrame(frames, leadDay) {
  if (!Array.isArray(frames) || !frames.length) return null;
  let best = frames[0];
  for (const frame of frames) {
    if (Math.abs(frame.lead_day - leadDay) < Math.abs(best.lead_day - leadDay)) best = frame;
  }
  return best;
}

/**
 * Per-lead census of a dataset's OWN eddy detections ("eddy-census" artifact:
 * frame.detections directly), at the available lead nearest `leadDay`. Returns
 * { detections, leadDay, parameters } or null. Nothing here privileges any dataset
 * as ground truth — a census is symmetric across forecasts and references.
 *
 * Async because the index format holds each lead in its own sidecar file; the legacy
 * inline-frames format resolves without a fetch.
 */
export async function eddyCensus(eddies, leadDay) {
  if (!eddies) return null;
  const entries = eddyLeadEntries(eddies);
  const frame = entries ? await eddyLeadFrame(eddies, entries, leadDay) : nearestLeadFrame(eddies.frames, leadDay);
  if (!frame) return null;
  return {
    detections: Array.isArray(frame.detections) ? frame.detections : [],
    leadDay: frame.lead_day,
    parameters: eddies.parameters || null,
  };
}

/** Sorted, de-duplicated lead days present in an eddy-census artifact (either format). */
export function eddyLeads(eddies) {
  const entries = eddyLeadEntries(eddies);
  const source = entries || (eddies && Array.isArray(eddies.frames) ? eddies.frames : null);
  if (!source) return [];
  const leads = new Set();
  for (const item of source) if (Number.isFinite(Number(item.lead_day))) leads.add(Number(item.lead_day));
  return [...leads].sort((a, b) => a - b);
}

function nearestLead(leads, leadDay) {
  let best = leads[0];
  for (const lead of leads) {
    if (Math.abs(lead - leadDay) < Math.abs(best - leadDay)) best = lead;
  }
  return best;
}

/**
 * Two forecasts' eddy censuses at a SINGLE shared lead day, so a cross-match compares like
 * with like. The requested `leadDay` is snapped to the nearest lead present in BOTH
 * forecasts' available leads (their intersection), and both censuses are read at that one
 * lead. When the two forecasts publish no lead in common the intersection is empty: each
 * census is read at its own nearest lead and `mismatch` is set so the caller suppresses the
 * cross-match and reports both leads. Returns `{ censuses: [censusA, censusB], lead, mismatch }`.
 */
export async function alignedEddyCensuses(eddiesA, eddiesB, leadDay) {
  const common = eddyLeads(eddiesA).filter((lead) => eddyLeads(eddiesB).includes(lead));
  if (common.length) {
    const lead = nearestLead(common, leadDay);
    return { censuses: await Promise.all([eddyCensus(eddiesA, lead), eddyCensus(eddiesB, lead)]), lead, mismatch: false };
  }
  return {
    censuses: await Promise.all([eddyCensus(eddiesA, leadDay), eddyCensus(eddiesB, leadDay)]),
    lead: null,
    mismatch: true,
  };
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
 * without display thinning. The viewer may thin the drawn subset, but statistics
 * and skill curves must keep using the complete loaded match-up set.
 */
export function class4Points(rows, { variable, depthBin, leadDay, startDate }) {
  if (!rows) return [];
  if (isClass4CurrentSpeedVariable(variable)) return class4SpeedPoints(rows, { variable, depthBin, leadDay, startDate });
  const parquetVariable = class4ParquetVariable(variable);
  const requestedLead = leadDay == null ? null : Number(leadDay);
  const requestedStart = startDate || null;
  const matched = [];
  for (const row of rows) {
    if (row.variable !== parquetVariable) continue;
    if (depthBin && row.depth_bin !== depthBin) continue;
    if (requestedLead !== null && Number(row.lead_day) !== requestedLead) continue;
    if (requestedStart && formatClass4Date(row.start_date) !== requestedStart) continue;
    // Masked-model rows (finite obs, NaN model → non-finite error) carry no comparison to
    // draw or count; drop them so they neither paint as false zero-error nor inflate "N obs".
    if (!Number.isFinite(class4AbsoluteError(row))) continue;
    matched.push(row);
  }
  return matched;
}

const CLASS4_CURRENT_COMPONENTS = {
  u: "eastward_sea_water_velocity",
  v: "northward_sea_water_velocity",
};

// Depth-suffixed velocity components (…_15m) are stored in the match-up parquet under
// their depth-less standard name (eastward/northward_sea_water_velocity); the 15 m depth
// is carried by depth_bin, not the variable name. Map the viewer field key back to the
// parquet variable name so the 15 m velocity components match their drifter obs rows.
export function class4ParquetVariable(variable) {
  if (/sea_water_velocity_15m$/.test(String(variable))) return String(variable).replace(/_15m$/, "");
  return variable;
}

function isClass4CurrentSpeedVariable(variable) {
  return variable === "current_speed" || variable === "current_speed_15m";
}

function class4SpeedPoints(rows, { variable, depthBin, leadDay, startDate }) {
  const requestedLead = leadDay == null ? null : Number(leadDay);
  const requestedStart = startDate || null;
  const byLocation = new Map();
  for (const row of rows) {
    if (row.variable !== CLASS4_CURRENT_COMPONENTS.u && row.variable !== CLASS4_CURRENT_COMPONENTS.v) continue;
    if (depthBin && row.depth_bin !== depthBin) continue;
    if (requestedLead !== null && Number(row.lead_day) !== requestedLead) continue;
    if (requestedStart && formatClass4Date(row.start_date) !== requestedStart) continue;
    const key = class4PairKey(row);
    const pair = byLocation.get(key) || {};
    if (row.variable === CLASS4_CURRENT_COMPONENTS.u) pair.u = row;
    else pair.v = row;
    byLocation.set(key, pair);
  }
  const matched = [];
  for (const pair of byLocation.values()) {
    const point = class4SpeedPoint(pair, variable);
    if (point) matched.push(point);
  }
  return matched;
}

function class4PairKey(row) {
  return [
    row.latitude,
    row.longitude,
    row.depth_bin || "",
    formatClass4Date(row.start_date),
    Number(row.lead_day),
  ].join("|");
}

function class4SpeedPoint(pair, variable) {
  if (!pair.u || !pair.v) return null;
  const obsU = numericOrNaN(pair.u.observation_value);
  const obsV = numericOrNaN(pair.v.observation_value);
  const modelU = numericOrNaN(pair.u.model_value);
  const modelV = numericOrNaN(pair.v.model_value);
  if (![obsU, obsV, modelU, modelV].every(Number.isFinite)) return null;
  const observation = Math.hypot(obsU, obsV);
  const model = Math.hypot(modelU, modelV);
  return {
    ...pair.u,
    variable,
    observation_value: observation,
    model_value: model,
    abs_error: Math.abs(observation - model),
  };
}

function numericOrNaN(value) {
  if (value == null || value === "") return NaN;
  const number = Number(value);
  return Number.isFinite(number) ? number : NaN;
}

function formatClass4Date(value) {
  if (value instanceof Date) return value.toISOString().slice(0, 10);
  return String(value).slice(0, 10);
}
