// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Class-4 match-up reader (runs in a Web Worker so parquet decoding never blocks
// the map). Two loading strategies, chosen from the footer metadata:
//
//   TARGETED — the served parquet follows the match-up contract: rows sorted by
//   (start_date, lead_day) with every row group holding exactly one such pair
//   (min == max in the group's start_date/lead_day statistics). We fetch only the
//   row group(s) for the selected forecast start + lead — complete rows, no cap.
//
//   SAMPLED (legacy) — files whose stats are absent or mixed keep the old
//   behaviour: small files read whole, multi-GB files streamed and sampled across
//   scattered row groups so an altimeter month never OOMs the worker.

import { parquetReadObjects, parquetMetadataAsync } from "../vendor/hyparquet/hyparquet.min.js";

const WHOLE_FILE_MAX_BYTES = 50 * 1024 * 1024;
const RANGE_BYTE_BUDGET = 48 * 1024 * 1024;
const MIN_SCATTERED_GROUPS = 8;
// Row groups in a global match-up parquet hold ~1M rows each; reading whole groups
// would pull millions of objects into the worker (OOM) and choke the postMessage.
// A few thousand rows per sampled group is plenty for a scattered overlay preview.
const ROWS_PER_SAMPLED_GROUP = 4000;

// Per-file footer + open handle (shared by mode detection and targeted reads).
const fileInfoCache = new Map();
// Legacy whole/sampled row arrays, keyed by url.
const legacyRowsCache = new Map();
// Targeted pair reads, keyed by `${url}|${startDate}|${leadDay}`.
const targetedCache = new Map();

self.addEventListener("message", (event) => {
  const { id, op, url, byteLength, rowGroupIndex, sampleVariables, startDate, leadDay } = event.data;
  handle(op, url, { byteLength, rowGroupIndex, sampleVariables, startDate, leadDay })
    .then((payload) => self.postMessage({ id, ...payload }))
    .catch((error) => self.postMessage({ id, error: String(error.message || error) }));
});

async function handle(op, url, options) {
  const info = await ensureFileInfo(url, options.byteLength);
  if (op === "probe") return { targeted: info.targeted };
  if (op === "targeted" && info.targeted) {
    const rows = await targetedPair(url, info, options.startDate, options.leadDay);
    // `sampled`/`total` are sent as explicit fields: array-attached properties do
    // not survive the structured clone across postMessage.
    return { targeted: true, rows, total: rows.length, sampled: false };
  }
  const rows = await legacyRows(url, info, options);
  return { targeted: false, rows, total: rows.length, sampled: Boolean(rows.sampled) };
}

function ensureFileInfo(url, byteLengthHint) {
  if (fileInfoCache.has(url)) return fileInfoCache.get(url);
  const promise = (async () => {
    const size = await resolveByteLength(url, byteLengthHint);
    const file = httpRangeAsyncBuffer(url, size);
    const metadata = await parquetMetadataAsync(file);
    const rowGroups = metadata.row_groups || [];
    const targeted = rowGroups.length > 0 && allRowGroupsSinglePair(rowGroups);
    return { size, file, metadata, rowGroups, offsets: rowGroupRowOffsets(rowGroups), targeted };
  })();
  fileInfoCache.set(url, promise);
  return promise;
}

// ---- targeted mode ----------------------------------------------------------

function allRowGroupsSinglePair(rowGroups) {
  for (const group of rowGroups) {
    const startStat = columnStatistics(group, "start_date");
    const leadStat = columnStatistics(group, "lead_day");
    if (!startStat || !leadStat) return false;
    if (!statIsSingular(startStat) || !statIsSingular(leadStat)) return false;
  }
  return true;
}

function statIsSingular(statistics) {
  const min = statMin(statistics);
  const max = statMax(statistics);
  if (min == null || max == null) return false;
  return String(min) === String(max);
}

function columnStatistics(rowGroup, name) {
  for (const column of rowGroup.columns || []) {
    const meta = column.meta_data;
    if (meta && Array.isArray(meta.path_in_schema) && meta.path_in_schema.join(".") === name) {
      return meta.statistics || null;
    }
  }
  return null;
}

function statMin(statistics) {
  return statistics.min_value != null ? statistics.min_value : statistics.min;
}
function statMax(statistics) {
  return statistics.max_value != null ? statistics.max_value : statistics.max;
}

async function targetedPair(url, info, startDate, leadDay) {
  const key = `${url}|${startDate}|${leadDay}`;
  if (targetedCache.has(key)) return targetedCache.get(key);
  const promise = (async () => {
    const indices = matchingRowGroups(info.rowGroups, startDate, leadDay);
    if (!indices.length) {
      const empty = [];
      empty.targeted = true;
      empty.total = 0;
      return empty;
    }
    const rowStart = info.offsets[indices[0]];
    const lastIndex = indices[indices.length - 1];
    const rowEnd = info.offsets[lastIndex] + Number(info.rowGroups[lastIndex].num_rows || 0);
    const rows = await parquetReadObjects({
      file: info.file,
      metadata: info.metadata,
      rowStart,
      rowEnd,
      rowFormat: "object",
    });
    rows.targeted = true;
    rows.total = rows.length;
    return rows;
  })();
  targetedCache.set(key, promise);
  return promise;
}

function matchingRowGroups(rowGroups, startDate, leadDay) {
  const requestedStart = startDate == null ? null : String(startDate).slice(0, 10);
  const requestedLead = leadDay == null ? null : Number(leadDay);
  const indices = [];
  for (let index = 0; index < rowGroups.length; index += 1) {
    const startStat = columnStatistics(rowGroups[index], "start_date");
    const leadStat = columnStatistics(rowGroups[index], "lead_day");
    const groupStart = String(statMin(startStat)).slice(0, 10);
    const groupLead = Number(statMin(leadStat));
    if (requestedStart !== null && groupStart !== requestedStart) continue;
    if (requestedLead !== null && groupLead !== requestedLead) continue;
    indices.push(index);
  }
  return indices;
}

// ---- legacy (whole / sampled) mode ------------------------------------------

function legacyRows(url, info, options) {
  if (legacyRowsCache.has(url)) return legacyRowsCache.get(url);
  const promise = (async () => {
    if (info.size <= WHOLE_FILE_MAX_BYTES) return readWhole(url);
    return readSampled(info, options);
  })();
  legacyRowsCache.set(url, promise);
  return promise;
}

async function readWhole(url) {
  const response = await fetch(url, { cache: "no-cache" });
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

async function readSampled(info, { rowGroupIndex, sampleVariables }) {
  const { file, metadata, rowGroups, offsets } = info;
  const indices = selectRowGroups(rowGroups, { rowGroupIndex, sampleVariables });
  const chunks = [];
  for (const index of indices) {
    const rowStart = offsets[index];
    const groupRows = Number(rowGroups[index].num_rows || 0);
    const rowEnd = rowStart + Math.min(groupRows, ROWS_PER_SAMPLED_GROUP);
    if (rowEnd <= rowStart) continue;
    const rows = await parquetReadObjects({ file, metadata, rowStart, rowEnd, rowFormat: "object" });
    chunks.push(...rows);
  }
  chunks.sampled = indices.length < rowGroups.length;
  chunks.rowGroupSample = indices;
  return chunks;
}

function selectRowGroups(rowGroups, { rowGroupIndex, sampleVariables }) {
  const indexed = indexedRowGroups(rowGroupIndex, sampleVariables).filter((index) => index >= 0 && index < rowGroups.length);
  const candidates = indexed.length ? indexed : rowGroups.map((_, index) => index);
  const targetCount = Math.min(candidates.length, Math.max(MIN_SCATTERED_GROUPS, budgetedGroupCount(rowGroups, candidates)));
  if (targetCount >= candidates.length) return candidates;
  const selected = new Set();
  for (let i = 0; i < targetCount; i += 1) {
    const position = Math.round((i * (candidates.length - 1)) / Math.max(1, targetCount - 1));
    selected.add(candidates[position]);
  }
  return [...selected].sort((a, b) => a - b);
}

function budgetedGroupCount(rowGroups, candidates) {
  let used = 0;
  let count = 0;
  for (const index of candidates) {
    const size = Number(rowGroups[index].total_byte_size) || 1;
    if (count > 0 && used + size > RANGE_BYTE_BUDGET) break;
    used += size;
    count += 1;
  }
  return Math.max(1, count);
}

function indexedRowGroups(rowGroupIndex, sampleVariables) {
  const byVariable = rowGroupIndex && (rowGroupIndex.by_variable || rowGroupIndex.variables);
  if (!byVariable) return [];
  const selected = new Set();
  const variables = sampleVariables && sampleVariables.length ? sampleVariables : Object.keys(byVariable);
  for (const variable of variables) {
    for (const index of byVariable[variable] || []) selected.add(Number(index));
  }
  return [...selected].sort((a, b) => a - b);
}

function rowGroupRowOffsets(rowGroups) {
  const offsets = [];
  let cursor = 0;
  for (const group of rowGroups) {
    offsets.push(cursor);
    cursor += Number(group.num_rows || 0);
  }
  return offsets;
}

function httpRangeAsyncBuffer(url, byteLength) {
  return {
    byteLength,
    async slice(start, end) {
      const last = (end ?? byteLength) - 1;
      const response = await fetch(url, { cache: "no-cache", headers: { Range: `bytes=${start}-${last}` } });
      if (response.status !== 206 && response.status !== 200) {
        throw new Error(`${url} range ${start}-${last} -> HTTP ${response.status}`);
      }
      return response.arrayBuffer();
    },
  };
}

async function resolveByteLength(url, hint) {
  if (Number.isFinite(hint) && hint > 0) return hint;
  const response = await fetch(url, { method: "HEAD", cache: "no-cache" });
  if (!response.ok) throw new Error(`${url} HEAD -> HTTP ${response.status}`);
  const length = Number(response.headers.get("content-length"));
  if (!Number.isFinite(length) || length <= 0) throw new Error(`${url} HEAD returned no content-length`);
  return length;
}
