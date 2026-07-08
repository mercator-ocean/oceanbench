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
    const response = await fetch(resolvedUrl, { cache: "no-cache" });
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
 * Returns `{ targeted, rows, total, sampled }`. In targeted
 * mode `rows` is exactly the selected pair; in legacy mode it is the (possibly
 * sampled) whole set as before. `byteLength` may come from the sibling manifest's
 * `bytes` field to skip a HEAD request, but callers should omit stale hints.
 */
export async function loadClass4(url, { byteLength, rowGroupIndex, sampleVariables, startDate, leadDay, variables } = {}) {
  if (!url) throw new Error("Class-4 URL is missing");
  const resolvedUrl = resolveViewerDataUrl(url);
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
  // Row groups are variable-partitioned within a (start, lead) block, so the requested
  // variables refine which groups are fetched — they belong in the targeted cache key.
  const variableKey = Array.isArray(variables) && variables.length ? [...variables].sort().join(",") : "";
  const key = `${resolvedUrl}|${startDate ?? ""}|${leadDay ?? ""}|${variableKey}`;
  if (!class4TargetedCache.has(key)) {
    class4TargetedCache.set(
      key,
      requestClass4Worker({ op: "targeted", url: resolvedUrl, byteLength, startDate, leadDay, variables }).then((payload) =>
        normalizeClass4Payload(payload),
      ),
    );
  }
  return await class4TargetedCache.get(key);
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

function nearestLeadFrame(frames, leadDay) {
  if (!Array.isArray(frames) || !frames.length) return null;
  let best = frames[0];
  for (const frame of frames) {
    if (Math.abs(frame.lead_day - leadDay) < Math.abs(best.lead_day - leadDay)) best = frame;
  }
  return best;
}

/**
 * Per-lead census of a dataset's OWN eddy detections, read from EITHER artifact
 * schema, at the available lead nearest `leadDay`:
 *   - "eddy-census": frame.detections directly.
 *   - legacy "eddies": the dataset's own detections are the challenger side of every
 *     match plus its spurious eddies (matches[].challenger ∪ spurious); the truth-side
 *     detections are never treated as this dataset's own.
 * Returns { detections, leadDay, parameters } or null. Nothing here privileges any
 * dataset as ground truth — a census is symmetric across forecasts and references.
 */
export function eddyCensus(eddies, leadDay) {
  if (!eddies) return null;
  if (Array.isArray(eddies.frames)) {
    const frame = nearestLeadFrame(eddies.frames, leadDay);
    if (!frame) return null;
    return {
      detections: Array.isArray(frame.detections) ? frame.detections : [],
      leadDay: frame.lead_day,
      parameters: eddies.parameters || null,
    };
  }
  if (Array.isArray(eddies.references)) {
    const entry = eddies.references[0];
    const frame = entry ? nearestLeadFrame(entry.frames, leadDay) : null;
    if (!frame) return null;
    const detections = [
      ...(frame.matches || []).map((match) => match.challenger).filter(Boolean),
      ...(frame.spurious || []),
    ];
    return { detections, leadDay: frame.lead_day, parameters: eddies.parameters || null };
  }
  return null;
}

/** Sorted, de-duplicated lead days present in an eddy artifact (either schema). */
export function eddyLeads(eddies) {
  if (!eddies) return [];
  const frames = Array.isArray(eddies.frames)
    ? eddies.frames
    : Array.isArray(eddies.references) && eddies.references[0]
      ? eddies.references[0].frames || []
      : [];
  const leads = new Set();
  for (const frame of frames) if (Number.isFinite(frame.lead_day)) leads.add(frame.lead_day);
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
export function alignedEddyCensuses(eddiesA, eddiesB, leadDay) {
  const common = eddyLeads(eddiesA).filter((lead) => eddyLeads(eddiesB).includes(lead));
  if (common.length) {
    const lead = nearestLead(common, leadDay);
    return { censuses: [eddyCensus(eddiesA, lead), eddyCensus(eddiesB, lead)], lead, mismatch: false };
  }
  return { censuses: [eddyCensus(eddiesA, leadDay), eddyCensus(eddiesB, leadDay)], lead: null, mismatch: true };
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
