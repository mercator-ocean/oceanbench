// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

import { parquetReadObjects, parquetMetadataAsync } from "../vendor/hyparquet/hyparquet.min.js";

const WHOLE_FILE_MAX_BYTES = 50 * 1024 * 1024;
const RANGE_BYTE_BUDGET = 48 * 1024 * 1024;
const MIN_SCATTERED_GROUPS = 8;
// Row groups in a global match-up parquet hold ~1M rows each; reading whole groups
// would pull millions of objects into the worker (OOM) and choke the postMessage.
// A few thousand rows per sampled group is plenty for a scattered overlay preview.
const ROWS_PER_SAMPLED_GROUP = 4000;

const parsedCache = new Map();

self.addEventListener("message", (event) => {
  const { id, url, byteLength, rowGroupIndex, sampleVariables } = event.data;
  loadClass4Rows(url, { byteLength, rowGroupIndex, sampleVariables })
    .then((rows) => self.postMessage({ id, rows }))
    .catch((error) => self.postMessage({ id, error: String(error.message || error) }));
});

async function loadClass4Rows(url, { byteLength, rowGroupIndex, sampleVariables }) {
  if (parsedCache.has(url)) return parsedCache.get(url);
  const promise = (async () => {
    const size = await resolveByteLength(url, byteLength);
    if (size <= WHOLE_FILE_MAX_BYTES) return readWhole(url);
    return readSampled(url, size, { rowGroupIndex, sampleVariables });
  })();
  parsedCache.set(url, promise);
  return promise;
}

async function readWhole(url) {
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
}

async function readSampled(url, byteLength, { rowGroupIndex, sampleVariables }) {
  const file = httpRangeAsyncBuffer(url, byteLength);
  const metadata = await parquetMetadataAsync(file);
  const rowGroups = metadata.row_groups || [];
  const offsets = rowGroupRowOffsets(rowGroups);
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
