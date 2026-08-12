// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Class-4 match-up reader (runs in a Web Worker so parquet decoding never blocks
// the map). The served parquet follows the match-up contract: rows sorted by
// (start_date, lead_day) with every row group holding exactly one such pair
// (min == max in the group's start_date/lead_day statistics), as enforced at
// publish time by oceanbench.publish.viewer_artifacts.verify_matchup_parquet. We
// fetch only the row group(s) for the selected forecast start + lead — complete
// rows, no cap.

import { parquetReadObjects, parquetMetadataAsync } from "../vendor/hyparquet/hyparquet.min.js";
import { decompress as zstdDecompress } from "../vendor/fzstd/fzstd.js";

const RANGE_BYTE_BUDGET = 48 * 1024 * 1024;

// hyparquet decodes UNCOMPRESSED and SNAPPY itself; the published parquet is ZSTD, so supply
// that codec from the vendored fzstd. Files of either vintage decode through the same path.
const PARQUET_COMPRESSORS = {
  ZSTD: (input, outputLength) => zstdDecompress(input, new Uint8Array(outputLength)),
};

// The published parquet drops `abs_error`: it is |model_value - observation_value| and every
// consumer can derive it. Files written before the change still carry the column, so derive it
// only when it is missing and both encodings behave identically downstream.
function withAbsoluteError(rows) {
  for (const row of rows) {
    if (row.abs_error == null) {
      row.abs_error = Math.abs(Number(row.model_value) - Number(row.observation_value));
    }
  }
  return rows;
}

// Per-file footer + open handle (shared by mode detection and targeted reads).
const fileInfoCache = new Map();
// Targeted pair reads, keyed by `${url}|${startDate}|${leadDay}`.
const targetedCache = new Map();

self.addEventListener("message", (event) => {
  const { id, op, url, byteLength, startDate, leadDay, variables } = event.data;
  handle(op, url, { byteLength, startDate, leadDay, variables }, id)
    .then((payload) => self.postMessage({ id, ...payload }))
    .catch((error) => self.postMessage({ id, error: String(error.message || error) }));
});

// The range reader is built once per file and cached, so it cannot hold the reporter of
// the request that created it: it reads whichever request is in flight instead.
let activeProgress = null;

// Byte counter behind the loading bar. `bytesTotal` is 0 while the size of the phase is
// unknown (the caller shows an indeterminate bar); the footer phase learns it from the
// content-length its requests announce, the row-group phase from its byte span.
function progressReporter(id) {
  let phase = "footer";
  let bytesDone = 0;
  let bytesTotal = 0;
  let bytesAnnounced = 0;
  let phaseDeclaredTotal = false;
  let lastPostedAt = 0;
  const post = () => {
    lastPostedAt = Date.now();
    self.postMessage({ id, progress: { phase, bytesDone, bytesTotal } });
  };
  return {
    beginPhase(name, total) {
      phase = name;
      bytesDone = 0;
      bytesAnnounced = 0;
      bytesTotal = Number(total) > 0 ? Number(total) : 0;
      phaseDeclaredTotal = bytesTotal > 0;
      post();
    },
    // A phase with no declared size sizes itself from what its requests announce. Summing
    // the content-lengths (rather than extending from the request in flight) keeps the
    // total right when hyparquet has several range reads open at once.
    expectRequest(bytes) {
      if (phaseDeclaredTotal || !(Number(bytes) > 0)) return;
      bytesAnnounced += Number(bytes);
      bytesTotal = bytesAnnounced;
      post();
    },
    addBytes(bytes) {
      bytesDone += bytes;
      if (Date.now() - lastPostedAt >= 250) post();
    },
  };
}

async function handle(op, url, options, id) {
  activeProgress = progressReporter(id);
  activeProgress.beginPhase("footer", 0);
  try {
    return await read(op, url, options);
  } finally {
    activeProgress = null;
  }
}

async function read(op, url, options) {
  const info = await ensureFileInfo(url, options.byteLength);
  if (!info.targeted) {
    throw new Error(`${url} is not in targeted layout (one (start_date, lead_day) pair per row group)`);
  }
  const rows = await targetedPair(url, info, options.startDate, options.leadDay, options.variables);
  // `total` is sent as an explicit field: array-attached properties do not survive
  // the structured clone across postMessage.
  return { targeted: true, rows, total: rows.length };
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

async function targetedPair(url, info, startDate, leadDay, variables) {
  const requestedVariables = Array.isArray(variables) && variables.length ? [...variables].sort() : null;
  const key = `${url}|${startDate}|${leadDay}|${requestedVariables ? requestedVariables.join(",") : ""}`;
  if (targetedCache.has(key)) return targetedCache.get(key);
  const promise = (async () => {
    const indices = matchingRowGroups(info.rowGroups, startDate, leadDay, requestedVariables);
    if (!indices.length) {
      const empty = [];
      empty.targeted = true;
      empty.total = 0;
      return empty;
    }
    const rowStart = info.offsets[indices[0]];
    const lastIndex = indices[indices.length - 1];
    const rowEnd = info.offsets[lastIndex] + Number(info.rowGroups[lastIndex].num_rows || 0);
    // hyparquet fetches each row group's column chunks with a separate serial range read.
    // For a targeted (start, lead) pair the selected groups occupy one contiguous byte span,
    // so coalesce them into a single range request served from memory — halving the network
    // phase of a lead change. Falls back to the range-backed file if the span is unknown or
    // exceeds the budget (keeps large legacy files streaming).
    const coalesced = await coalescedRangeFile(info, indices);
    // The coalesced path announces its own byte span; the fallback reads row group by row
    // group with no span to announce, so its phase stays indeterminate.
    if (!coalesced && activeProgress) activeProgress.beginPhase("rows", 0);
    const file = coalesced || info.file;
    const rows = withAbsoluteError(
      await parquetReadObjects({
        file,
        metadata: info.metadata,
        rowStart,
        rowEnd,
        rowFormat: "object",
        compressors: PARQUET_COMPRESSORS,
      })
    );
    rows.targeted = true;
    rows.total = rows.length;
    return rows;
  })();
  targetedCache.set(key, promise);
  return promise;
}

// Byte span [lowest column-chunk offset, highest chunk end) covering the selected row
// groups. Returns null if any offset/size is missing (fall back to per-range reads).
function rowGroupsByteSpan(rowGroups, indices) {
  let low = Infinity;
  let high = 0;
  for (const index of indices) {
    for (const column of rowGroups[index].columns || []) {
      const meta = column.meta_data;
      if (!meta) return null;
      const dataOffset = meta.data_page_offset != null ? Number(meta.data_page_offset) : null;
      const dictionaryOffset = meta.dictionary_page_offset != null ? Number(meta.dictionary_page_offset) : null;
      const chunkStart = dictionaryOffset != null ? Math.min(dictionaryOffset, dataOffset ?? dictionaryOffset) : dataOffset;
      const chunkSize = Number(meta.total_compressed_size) || 0;
      if (chunkStart == null || !Number.isFinite(chunkStart)) return null;
      low = Math.min(low, chunkStart);
      high = Math.max(high, chunkStart + chunkSize);
    }
  }
  if (!Number.isFinite(low) || high <= low) return null;
  return { low, high };
}

// Prefetch the selected row groups' contiguous byte span in one request and hand back a
// buffer-backed file so hyparquet's subsequent slices resolve from memory (no network).
async function coalescedRangeFile(info, indices) {
  const span = rowGroupsByteSpan(info.rowGroups, indices);
  if (!span) return null;
  if (span.high - span.low > RANGE_BYTE_BUDGET) return null;
  if (activeProgress) activeProgress.beginPhase("rows", span.high - span.low);
  const buffer = await info.file.slice(span.low, span.high);
  return {
    byteLength: info.size,
    async slice(start, end) {
      const stop = end ?? info.size;
      if (start >= span.low && stop <= span.high) return buffer.slice(start - span.low, stop - span.low);
      // Any read outside the prefetched window (should not happen for a targeted read)
      // falls back to a direct range request against the source.
      return info.file.slice(start, end);
    },
  };
}

function matchingRowGroups(rowGroups, startDate, leadDay, variables) {
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
    // Rows are sorted (start, lead, variable): within a matched (start, lead) block, skip
    // row groups whose `variable` min/max stats prove they hold no requested variable.
    // Groups straddling a variable boundary (min < v < max) still pass. Missing stats →
    // keep (cannot prove absence). Derived current speed asks for both u and v.
    if (variables && !rowGroupHasVariable(rowGroups[index], variables)) continue;
    indices.push(index);
  }
  return indices;
}

function rowGroupHasVariable(rowGroup, variables) {
  const stat = columnStatistics(rowGroup, "variable");
  if (!stat) return true;
  const min = statMin(stat);
  const max = statMax(stat);
  if (min == null || max == null) return true;
  const low = String(min);
  const high = String(max);
  return variables.some((variable) => String(variable) >= low && String(variable) <= high);
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
      return readCountingBytes(response);
    },
  };
}

// Drain the range response through its stream so the bar advances during the download
// rather than at the end of it. Only one read streams at a time: legacy files are read as
// dozens of concurrent range requests, and holding all of their bodies as JS chunks runs
// the browser out of resources (measured: net::ERR_INSUFFICIENT_RESOURCES on wenhai, where
// the buffered read completes). The rest are counted when they land, which still advances
// the bar, and the long serial reads (the footer, the coalesced row-group span) stream.
let streamingRead = false;

async function readCountingBytes(response) {
  const progress = activeProgress;
  if (!progress) return response.arrayBuffer();
  progress.expectRequest(Number(response.headers.get("content-length")));
  if (!response.body || streamingRead) {
    const buffer = await response.arrayBuffer();
    progress.addBytes(buffer.byteLength);
    return buffer;
  }
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;
  streamingRead = true;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.byteLength;
      progress.addBytes(value.byteLength);
    }
  } finally {
    streamingRead = false;
  }
  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return buffer.buffer;
}

async function resolveByteLength(url, hint) {
  if (Number.isFinite(hint) && hint > 0) return hint;
  const response = await fetch(url, { method: "HEAD" });
  if (!response.ok) throw new Error(`${url} HEAD -> HTTP ${response.status}`);
  const length = Number(response.headers.get("content-length"));
  if (!Number.isFinite(length) || length <= 0) throw new Error(`${url} HEAD returned no content-length`);
  return length;
}
