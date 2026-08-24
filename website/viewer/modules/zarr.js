// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Minimal zarr v2 reader for OceanBench viewer pyramids (contracts.md §6).
//
// The layout is fixed and known: a consolidated `.zmetadata`, groups `level/<k>`,
// data arrays of shape (start_date, lead_day, latitude, longitude) chunked as
// 256x256 spatial tiles with one (start_date, lead_day) per chunk, stored as
// uint16 with per-variable scale_factor/add_offset and an explicit _FillValue,
// DEFLATE-compressed. Decoding prefers the platform-native DecompressionStream (no
// wasm codec, which is why the builder writes zlib rather than blosc) and falls back
// to the small built-in inflater below on browsers without it.
//
// This is deliberately a few hundred lines for our own layout, not a general
// zarr client. It scales to multi-level pyramids because every read is driven by
// the array's own .zarray chunk grid, not by any 1-degree assumption.

async function inflate(compressed, codecId) {
  if (codecId === null || codecId === undefined) return new Uint8Array(compressed);
  if (codecId !== "zlib" && codecId !== "gzip") {
    throw new Error(`Unsupported compressor '${codecId}'. Pyramids must be zlib/gzip for browser decode.`);
  }
  // DecompressionStream is absent on older Safari and Firefox; those decode through
  // the software inflater below instead of failing every tile fetch.
  if (typeof DecompressionStream === "function") {
    const format = codecId === "gzip" ? "gzip" : "deflate";
    const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream(format));
    const buffer = await new Response(stream).arrayBuffer();
    return new Uint8Array(buffer);
  }
  return softwareInflate(new Uint8Array(compressed), codecId);
}

// ---- software DEFLATE decoder (RFC 1951), used only when DecompressionStream is missing ----

const LENGTH_BASE = [3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131, 163, 195, 227, 258];
const LENGTH_EXTRA = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0];
const DISTANCE_BASE = [1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577];
const DISTANCE_EXTRA = [0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13];
// Order in which the dynamic header stores its code-length alphabet (RFC 1951 §3.2.7).
const CODE_LENGTH_ORDER = [16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15];

function softwareInflate(wrapped, codecId) {
  const payloadStart = codecId === "gzip" ? gzipHeaderBytes(wrapped) : zlibHeaderBytes(wrapped);
  return inflateRaw(wrapped.subarray(payloadStart));
}

function zlibHeaderBytes(bytes) {
  if (bytes.length < 2 || (bytes[0] & 0x0f) !== 8) throw new Error("Data is not a zlib stream");
  // FDICT set (FLG bit 5): a four-byte dictionary id sits between the two-byte
  // header and the data.
  const start = bytes[1] & 0x20 ? 6 : 2;
  if (start > bytes.length) throw new Error("Truncated zlib stream");
  return start;
}

function gzipHeaderBytes(bytes) {
  if (bytes.length < 18 || bytes[0] !== 0x1f || bytes[1] !== 0x8b || bytes[2] !== 8) {
    throw new Error("Data is not a gzip stream");
  }
  const flags = bytes[3];
  let offset = 10;
  if (flags & 4) {
    if (offset + 2 > bytes.length) throw new Error("Truncated gzip stream");
    offset += 2 + bytes[offset] + (bytes[offset + 1] << 8);
  }
  for (const flag of [8, 16]) {
    if (!(flags & flag)) continue;
    while (offset < bytes.length && bytes[offset]) offset += 1;
    offset += 1;
  }
  if (flags & 2) offset += 2;
  if (offset > bytes.length) throw new Error("Truncated gzip stream");
  return offset;
}

function bitReader(bytes) {
  let position = 0;
  let bitBuffer = 0;
  let bitCount = 0;
  return {
    bits(count) {
      while (bitCount < count) {
        if (position >= bytes.length) throw new Error("Truncated DEFLATE data");
        bitBuffer |= bytes[position] << bitCount;
        position += 1;
        bitCount += 8;
      }
      const value = bitBuffer & ((1 << count) - 1);
      bitBuffer >>>= count;
      bitCount -= count;
      return value;
    },
    alignToByte() {
      const drop = bitCount & 7;
      bitBuffer >>>= drop;
      bitCount -= drop;
    },
  };
}

function huffmanTable(lengths) {
  const counts = new Uint16Array(16);
  for (let symbol = 0; symbol < lengths.length; symbol += 1) counts[lengths[symbol]] += 1;
  counts[0] = 0;
  // offsets holds one slot past the longest code (index 16 doubles as the total).
  const offsets = new Uint16Array(17);
  for (let length = 1; length < 16; length += 1) offsets[length + 1] = offsets[length] + counts[length];
  const symbols = new Uint16Array(offsets[16]);
  for (let symbol = 0; symbol < lengths.length; symbol += 1) {
    if (!lengths[symbol]) continue;
    symbols[offsets[lengths[symbol]]] = symbol;
    offsets[lengths[symbol]] += 1;
  }
  return { counts, symbols };
}

// Canonical Huffman lookup by walking one bit at a time: symbols are sorted per code
// length, so a code's rank within its length indexes straight into `symbols`.
function decodeSymbol(reader, table) {
  let code = 0;
  let first = 0;
  let index = 0;
  for (let length = 1; length <= 15; length += 1) {
    code |= reader.bits(1);
    const count = table.counts[length];
    if (code - first < count) return table.symbols[index + code - first];
    index += count;
    first = (first + count) << 1;
    code <<= 1;
  }
  throw new Error("Invalid DEFLATE data");
}

function inflateRaw(bytes) {
  const reader = bitReader(bytes);
  let output = new Uint8Array(Math.max(65536, bytes.length * 3));
  let outputLength = 0;
  const push = (byte) => {
    if (outputLength === output.length) {
      const grown = new Uint8Array(output.length * 2);
      grown.set(output);
      output = grown;
    }
    output[outputLength] = byte;
    outputLength += 1;
  };
  let isFinal;
  do {
    isFinal = reader.bits(1);
    const blockType = reader.bits(2);
    if (blockType === 0) {
      reader.alignToByte();
      const storedLength = reader.bits(16);
      const storedInverse = reader.bits(16);
      if ((storedLength ^ 0xffff) !== storedInverse) throw new Error("Corrupt stored DEFLATE block");
      for (let i = 0; i < storedLength; i += 1) push(reader.bits(8));
    } else if (blockType === 1 || blockType === 2) {
      let literals;
      let distances;
      if (blockType === 1) {
        const fixedLiterals = new Uint8Array(288);
        fixedLiterals.fill(8, 0, 144);
        fixedLiterals.fill(9, 144, 256);
        fixedLiterals.fill(7, 256, 280);
        fixedLiterals.fill(8, 280, 288);
        literals = huffmanTable(fixedLiterals);
        distances = huffmanTable(new Uint8Array(30).fill(5));
      } else {
        const tables = readDynamicTables(reader);
        literals = huffmanTable(tables.literals);
        distances = huffmanTable(tables.distances);
      }
      for (;;) {
        const symbol = decodeSymbol(reader, literals);
        if (symbol < 256) {
          push(symbol);
          continue;
        }
        if (symbol === 256) break;
        const lengthIndex = symbol - 257;
        if (lengthIndex >= LENGTH_BASE.length) throw new Error("Invalid DEFLATE length symbol");
        const matchLength = LENGTH_BASE[lengthIndex] + reader.bits(LENGTH_EXTRA[lengthIndex]);
        const distanceSymbol = decodeSymbol(reader, distances);
        if (distanceSymbol >= DISTANCE_BASE.length) throw new Error("Invalid DEFLATE distance symbol");
        const matchDistance = DISTANCE_BASE[distanceSymbol] + reader.bits(DISTANCE_EXTRA[distanceSymbol]);
        if (matchDistance > outputLength) throw new Error("DEFLATE match reaches before the stream start");
        // Byte-at-a-time copy: the source run may overlap the write position, and the
        // overlap must repeat the bytes being produced, not the original ones.
        let source = outputLength - matchDistance;
        for (let i = 0; i < matchLength; i += 1) push(output[source + i]);
      }
    } else {
      throw new Error("Invalid DEFLATE block type");
    }
  } while (!isFinal);
  return output.slice(0, outputLength);
}

function readDynamicTables(reader) {
  const literalCount = reader.bits(5) + 257;
  const distanceCount = reader.bits(5) + 1;
  const codeLengthCount = reader.bits(4) + 4;
  const codeLengths = new Uint8Array(19);
  for (let i = 0; i < codeLengthCount; i += 1) codeLengths[CODE_LENGTH_ORDER[i]] = reader.bits(3);
  const codeTable = huffmanTable(codeLengths);
  const lengths = new Uint8Array(literalCount + distanceCount);
  let index = 0;
  while (index < lengths.length) {
    const symbol = decodeSymbol(reader, codeTable);
    if (symbol < 16) {
      lengths[index] = symbol;
      index += 1;
      continue;
    }
    let repeats;
    let value = 0;
    if (symbol === 16) {
      if (index === 0) throw new Error("Invalid DEFLATE run with no previous length");
      value = lengths[index - 1];
      repeats = 3 + reader.bits(2);
    } else if (symbol === 17) {
      repeats = 3 + reader.bits(3);
    } else {
      repeats = 11 + reader.bits(7);
    }
    if (index + repeats > lengths.length) throw new Error("Invalid DEFLATE run past the table end");
    for (let r = 0; r < repeats; r += 1) {
      lengths[index] = value;
      index += 1;
    }
  }
  return { literals: lengths.slice(0, literalCount), distances: lengths.slice(literalCount) };
}

export async function loadStore(storeUrl) {
  const base = storeUrl.replace(/\/$/, "");
  const response = await fetch(`${base}/.zmetadata`);
  if (!response.ok) throw new Error(`Cannot load ${base}/.zmetadata (${response.status})`);
  const consolidated = await response.json();
  return {
    baseUrl: base,
    metadata: consolidated.metadata,
    chunkCache: new Map(),
    coordinateCache: new Map(),
  };
}

export async function loadManifest(manifestUrl) {
  const response = await fetch(manifestUrl);
  if (!response.ok) throw new Error(`Cannot load manifest ${manifestUrl} (${response.status})`);
  return response.json();
}

function arrayMetadata(store, path) {
  const zarray = store.metadata[`${path}/.zarray`];
  const zattrs = store.metadata[`${path}/.zattrs`] || {};
  if (!zarray) throw new Error(`No array metadata for ${path}`);
  return { zarray, zattrs };
}

function compressorId(zarray) {
  return zarray.compressor ? zarray.compressor.id : null;
}

function abortError() {
  return new DOMException("Aborted", "AbortError");
}

// The cache holds the in-flight PROMISE, not the settled record: two panels asking
// for the same tile in the same frame then share one download instead of racing two.
// A rejected request drops out of the cache so a later read can retry.
function fetchChunk(store, path, chunkKey, codecId, signal) {
  const cacheKey = `${path}/${chunkKey}`;
  // A caller arriving with an already-aborted signal wants nothing: it must not create a
  // request, and above all it must not walk through the waiter bookkeeping of a request it
  // never joined. Doing so used to abort a live download and drop a fully decoded tile.
  if (signal && signal.aborted) return Promise.reject(abortError());
  let entry = store.chunkCache.get(cacheKey);
  if (!entry) {
    const controller = new AbortController();
    entry = { controller, waiters: 0, promise: null, decoded: false, bytes: 0, used: 0 };
    entry.promise = requestChunk(store, path, chunkKey, codecId, controller.signal);
    entry.promise.then(
      (record) => {
        entry.decoded = true;
        entry.bytes = record.bytes.byteLength;
        evictChunks(store);
      },
      () => {
        if (store.chunkCache.get(cacheKey) === entry) store.chunkCache.delete(cacheKey);
      },
    );
    store.chunkCache.set(cacheKey, entry);
  }
  entry.used = (store.chunkClock = (store.chunkClock || 0) + 1);
  // Every consumer counts, signalled or not. The signal-less readers (windows, columns,
  // coordinates, prefetch) were invisible to the refcount, so a single signalled waiter
  // changing its mind aborted the download they were still waiting on.
  if (!signal) return joinUnsignalledRequest(entry);
  return joinChunkRequest(store, cacheKey, entry, signal);
}

function joinUnsignalledRequest(entry) {
  entry.waiters += 1;
  const release = () => {
    entry.waiters -= 1;
  };
  return entry.promise.then(
    (record) => {
      release();
      return record;
    },
    (error) => {
      release();
      throw error;
    },
  );
}

// A long session pans and scrubs through far more tiles than it can hold: at ~0.5 MB decoded
// each, an unbounded cache is a slow leak. Keep the most recently used decoded tiles up to a
// byte budget wide enough for every lead of two panels (the warm playback loop must still not
// touch the network), and drop the coldest beyond it. In-flight tiles and tiles anyone is
// still waiting on are never evicted.
const CHUNK_CACHE_BYTES = 512 * 1024 * 1024;

function evictChunks(store) {
  let total = 0;
  const evictable = [];
  for (const [key, entry] of store.chunkCache) {
    if (!entry.decoded) continue;
    total += entry.bytes;
    if (entry.waiters <= 0) evictable.push([key, entry]);
  }
  if (total <= CHUNK_CACHE_BYTES) return;
  evictable.sort((a, b) => a[1].used - b[1].used);
  for (const [key, entry] of evictable) {
    if (total <= CHUNK_CACHE_BYTES) return;
    if (store.chunkCache.get(key) !== entry) continue;
    store.chunkCache.delete(key);
    total -= entry.bytes;
  }
}

// Wait on a shared request under a caller's own abort signal. The underlying fetch is
// only really aborted once every waiter has walked away, so one panel changing its mind
// can never cancel the tile a sibling panel is still drawing.
function joinChunkRequest(store, cacheKey, entry, signal) {
  entry.waiters += 1;
  return new Promise((resolve, reject) => {
    let settled = false;
    const release = () => {
      if (settled) return false;
      settled = true;
      entry.waiters -= 1;
      return true;
    };
    const onAbort = () => {
      if (!release()) return;
      // A tile that has already arrived is not cancellable and is worth keeping: evicting it
      // here threw away decoded data and made the next read of the same tile go to the
      // network again, which is exactly what the cache exists to prevent.
      if (entry.waiters <= 0 && !entry.decoded) {
        entry.controller.abort();
        if (store.chunkCache.get(cacheKey) === entry) store.chunkCache.delete(cacheKey);
      }
      reject(abortError());
    };
    if (signal.aborted) {
      onAbort();
      return;
    }
    signal.addEventListener("abort", onAbort, { once: true });
    entry.promise.then(
      (record) => {
        signal.removeEventListener("abort", onAbort);
        if (release()) resolve(record);
      },
      (error) => {
        signal.removeEventListener("abort", onAbort);
        if (release()) reject(error);
      },
    );
  });
}

/**
 * True when every tile of one (start_date, lead_day) slice is already decoded in this
 * store's cache, so `readLayer` for that slice resolves without touching the network.
 * A still-pending request does not count: the caller uses this to decide whether it
 * can paint inside the current frame.
 */
export function isLayerCached(store, { variable, level, startIndex, leadIndex }) {
  const path = `level/${level}/${variable}`;
  const zarray = store.metadata[`${path}/.zarray`];
  if (!zarray) return false;
  const [, , latitudeSize, longitudeSize] = zarray.shape;
  const [, , latitudeChunk, longitudeChunk] = zarray.chunks;
  const latitudeTiles = Math.ceil(latitudeSize / latitudeChunk);
  const longitudeTiles = Math.ceil(longitudeSize / longitudeChunk);
  for (let ty = 0; ty < latitudeTiles; ty += 1) {
    for (let tx = 0; tx < longitudeTiles; tx += 1) {
      const entry = store.chunkCache.get(`${path}/${startIndex}.${leadIndex}.${ty}.${tx}`);
      if (!entry || !entry.decoded) return false;
    }
  }
  return true;
}

async function requestChunk(store, path, chunkKey, codecId, signal) {
  const url = `${store.baseUrl}/${path}/${chunkKey}`;
  const started = performance.now();
  // Name the resource (the variable/level path) rather than echoing the raw tile
  // URL, and turn a bare network failure ("Failed to fetch") into the same
  // friendly, resource-named message.
  let response;
  try {
    response = await fetch(url, { signal });
  } catch (error) {
    if (error && error.name === "AbortError") throw error;
    throw new Error(`Could not load a map tile for ${path} (network error, check your connection).`);
  }
  if (!response.ok) throw new Error(`Could not load a map tile for ${path} (HTTP ${response.status}).`);
  const compressed = await response.arrayBuffer();
  const bytes = await inflate(compressed, codecId);
  return { bytes, compressedBytes: compressed.byteLength, milliseconds: performance.now() - started };
}

/**
 * Read one 2D layer — a single (start_date, lead_day) slice of one variable at
 * one pyramid level — into a Float32Array in real units (NaN over land).
 * Returns { data, width, height, compressedBytes, fetchMilliseconds }.
 */
export async function readLayer(store, { variable, level, startIndex, leadIndex, signal }) {
  const path = `level/${level}/${variable}`;
  const { zarray, zattrs } = arrayMetadata(store, path);
  if (zarray.dtype !== "<u2" && zarray.dtype !== "|u2") {
    throw new Error(`Expected uint16 data, got ${zarray.dtype} for ${path}`);
  }
  const [, , latitudeSize, longitudeSize] = zarray.shape;
  const [, , latitudeChunk, longitudeChunk] = zarray.chunks;
  const scale = zattrs.scale_factor ?? 1;
  const offset = zattrs.add_offset ?? 0;
  const fill = zarray.fill_value ?? 65535;
  const codecId = compressorId(zarray);

  const data = new Float32Array(latitudeSize * longitudeSize);
  const latitudeTiles = Math.ceil(latitudeSize / latitudeChunk);
  const longitudeTiles = Math.ceil(longitudeSize / longitudeChunk);

  let compressedBytes = 0;
  let fetchMilliseconds = 0;
  const tileReads = [];
  for (let ty = 0; ty < latitudeTiles; ty += 1) {
    for (let tx = 0; tx < longitudeTiles; tx += 1) {
      tileReads.push({ ty, tx, record: fetchChunk(store, path, `${startIndex}.${leadIndex}.${ty}.${tx}`, codecId, signal) });
    }
  }
  // The tiles are fetched concurrently but consumed sequentially below. If an early
  // tile rejects, the loop throws before later (still-pending) tile promises are
  // awaited, so attach a benign handler now to keep a blocked tile from surfacing as
  // an uncaught promise rejection; the await still observes and rethrows the first error.
  for (const tile of tileReads) tile.record.catch(() => {});
  for (const tile of tileReads) {
    const record = await tile.record;
    compressedBytes += record.compressedBytes;
    fetchMilliseconds += record.milliseconds;
    const stored = new Uint16Array(record.bytes.buffer, record.bytes.byteOffset, record.bytes.byteLength / 2);
    const latitudeStart = tile.ty * latitudeChunk;
    const longitudeStart = tile.tx * longitudeChunk;
    const latitudeExtent = Math.min(latitudeChunk, latitudeSize - latitudeStart);
    const longitudeExtent = Math.min(longitudeChunk, longitudeSize - longitudeStart);
    for (let i = 0; i < latitudeExtent; i += 1) {
      const sourceRow = i * longitudeChunk;
      const destinationRow = (latitudeStart + i) * longitudeSize + longitudeStart;
      for (let j = 0; j < longitudeExtent; j += 1) {
        const raw = stored[sourceRow + j];
        data[destinationRow + j] = raw === fill ? NaN : raw * scale + offset;
      }
    }
  }
  return { data, width: longitudeSize, height: latitudeSize, compressedBytes, fetchMilliseconds };
}

/**
 * Read a geographic WINDOW of one 2D layer — only the 256×256 tiles intersecting the
 * requested lat/lon box are fetched, so a finest-level regional read (e.g. the seed
 * neighbourhood for trajectory advection) stays a handful of tiles instead of the
 * whole global grid. `latitudes`/`longitudes` are the level's coordinate axes (from
 * readCoordinate); the box is `{ latMin, latMax, lonMin, lonMax }` in degrees. On a
 * periodic (global) grid the longitude range may cross the dateline: the returned
 * axis is CONTINUOUS (unwrapped, may exceed ±180) so windows never carry a seam.
 * Returns { data, width, height, lat0, latStep, lon0, lonStep } or null when the box
 * misses the grid entirely.
 */
export async function readLayerWindow(store, { variable, level, startIndex, leadIndex }, latitudes, longitudes, box) {
  const path = `level/${level}/${variable}`;
  const { zarray, zattrs } = arrayMetadata(store, path);
  if (zarray.dtype !== "<u2" && zarray.dtype !== "|u2") {
    throw new Error(`Expected uint16 data, got ${zarray.dtype} for ${path}`);
  }
  const [, , latitudeSize, longitudeSize] = zarray.shape;
  const [, , latitudeChunk, longitudeChunk] = zarray.chunks;
  const scale = zattrs.scale_factor ?? 1;
  const offset = zattrs.add_offset ?? 0;
  const fill = zarray.fill_value ?? 65535;
  const codecId = compressorId(zarray);

  const latStep = latitudes.length > 1 ? latitudes[1] - latitudes[0] : 1;
  const lonStep = longitudes.length > 1 ? longitudes[1] - longitudes[0] : 1;
  const periodic = Math.abs(lonStep) * longitudeSize >= 359;

  // Row (latitude) range — clamped to the grid.
  const rowOf = (lat) => (lat - latitudes[0]) / latStep;
  let rowMin = Math.floor(Math.min(rowOf(box.latMin), rowOf(box.latMax)));
  let rowMax = Math.ceil(Math.max(rowOf(box.latMin), rowOf(box.latMax)));
  rowMin = Math.max(0, rowMin);
  rowMax = Math.min(latitudeSize - 1, rowMax);
  if (rowMax < rowMin) return null;

  // Column (longitude) range — continuous/unwrapped; wrapped onto source columns when
  // periodic, clamped otherwise. Capped at one full revolution.
  const colOf = (lon) => (lon - longitudes[0]) / lonStep;
  let colMin = Math.floor(Math.min(colOf(box.lonMin), colOf(box.lonMax)));
  let colMax = Math.ceil(Math.max(colOf(box.lonMin), colOf(box.lonMax)));
  if (!periodic) {
    colMin = Math.max(0, colMin);
    colMax = Math.min(longitudeSize - 1, colMax);
    if (colMax < colMin) return null;
  } else if (colMax - colMin + 1 > longitudeSize) {
    colMax = colMin + longitudeSize - 1;
  }

  const height = rowMax - rowMin + 1;
  const width = colMax - colMin + 1;
  const wrap = (column) => ((column % longitudeSize) + longitudeSize) % longitudeSize;

  // Distinct source tiles covering the window.
  const tileYs = new Set();
  for (let row = rowMin; row <= rowMax; row += 1) tileYs.add(Math.floor(row / latitudeChunk));
  const tileXs = new Set();
  for (let column = colMin; column <= colMax; column += 1) tileXs.add(Math.floor(wrap(column) / longitudeChunk));

  const tiles = new Map();
  await Promise.all(
    [...tileYs].flatMap((ty) =>
      [...tileXs].map(async (tx) => {
        const record = await fetchChunk(store, path, `${startIndex}.${leadIndex}.${ty}.${tx}`, codecId);
        tiles.set(`${ty}.${tx}`, new Uint16Array(record.bytes.buffer, record.bytes.byteOffset, record.bytes.byteLength / 2));
      }),
    ),
  );

  const data = new Float32Array(height * width);
  for (let row = 0; row < height; row += 1) {
    const sourceRow = rowMin + row;
    const ty = Math.floor(sourceRow / latitudeChunk);
    const tileRow = sourceRow - ty * latitudeChunk;
    for (let column = 0; column < width; column += 1) {
      const sourceColumn = wrap(colMin + column);
      const tx = Math.floor(sourceColumn / longitudeChunk);
      const stored = tiles.get(`${ty}.${tx}`);
      const raw = stored[tileRow * longitudeChunk + (sourceColumn - tx * longitudeChunk)];
      data[row * width + column] = raw === fill ? NaN : raw * scale + offset;
    }
  }
  return {
    data,
    width,
    height,
    lat0: latitudes[0] + rowMin * latStep,
    latStep,
    lon0: longitudes[0] + colMin * lonStep, // continuous (unwrapped) axis origin
    lonStep,
  };
}

export async function readCoordinate(store, level, name) {
  const cacheKey = `level/${level}/${name}`;
  const cached = store.coordinateCache.get(cacheKey);
  if (cached) return cached;
  const path = `level/${level}/${name}`;
  const { zarray } = arrayMetadata(store, path);
  const size = zarray.shape[0];
  const chunk = zarray.chunks[0];
  const codecId = compressorId(zarray);
  const values = new Float64Array(size);
  const tiles = Math.ceil(size / chunk);
  for (let t = 0; t < tiles; t += 1) {
    const record = await fetchChunk(store, path, `${t}`, codecId);
    const view =
      zarray.dtype === "<f4" || zarray.dtype === "|f4"
        ? new Float32Array(record.bytes.buffer, record.bytes.byteOffset, record.bytes.byteLength / 4)
        : new Float64Array(record.bytes.buffer, record.bytes.byteOffset, record.bytes.byteLength / 8);
    const start = t * chunk;
    const extent = Math.min(chunk, size - start);
    for (let i = 0; i < extent; i += 1) values[start + i] = view[i];
  }
  store.coordinateCache.set(cacheKey, values);
  return values;
}

/**
 * Read a 1-D coordinate array stored at the ROOT of a store (name is the array path
 * itself, not under `level/<k>/`) — used by the water-column store whose `depth`,
 * `latitude` and `longitude` axes live at the root. Handles f4/f8 (the coordinate
 * dtypes the column store writes). Cached like readCoordinate.
 */
export async function readRootCoordinate(store, name) {
  const cacheKey = `root/${name}`;
  const cached = store.coordinateCache.get(cacheKey);
  if (cached) return cached;
  const { zarray } = arrayMetadata(store, name);
  const size = zarray.shape[0];
  const chunk = zarray.chunks[0];
  const codecId = compressorId(zarray);
  const values = new Float64Array(size);
  const tiles = Math.ceil(size / chunk);
  for (let t = 0; t < tiles; t += 1) {
    const record = await fetchChunk(store, name, `${t}`, codecId);
    const view =
      zarray.dtype === "<f4" || zarray.dtype === "|f4"
        ? new Float32Array(record.bytes.buffer, record.bytes.byteOffset, record.bytes.byteLength / 4)
        : new Float64Array(record.bytes.buffer, record.bytes.byteOffset, record.bytes.byteLength / 8);
    const start = t * chunk;
    const extent = Math.min(chunk, size - start);
    for (let i = 0; i < extent; i += 1) values[start + i] = view[i];
  }
  store.coordinateCache.set(cacheKey, values);
  return values;
}

/**
 * Read one point's full water column from a `.columns.zarr` store: ALL lead days and
 * ALL depths of a single (start, latitude, longitude) point of one variable, decoded to
 * real units (NaN over land/missing). The store packs every lead and every depth of a
 * point into ONE chunk (dims start_date, lead_day, depth, latitude, longitude; chunk
 * [1, all-leads, all-depths, latTile, lonTile]), so a click is a single chunk fetch — and
 * because that chunk is cached, scrubbing the lead slider re-reads it with no new request.
 * Returns { values: Float32Array indexed [leadIndex * depths + depthIndex], leads, depths,
 * compressedBytes }.
 */
export async function readColumn(store, { variable, startIndex, latIndex, lonIndex }) {
  const { zarray, zattrs } = arrayMetadata(store, variable);
  if (zarray.dtype !== "<u2" && zarray.dtype !== "|u2") {
    throw new Error(`Expected uint16 column data, got ${zarray.dtype} for ${variable}`);
  }
  const [, leadSize, depthSize] = zarray.shape;
  const [, , depthChunk, latChunk, lonChunk] = zarray.chunks;
  const scale = zattrs.scale_factor ?? 1;
  const offset = zattrs.add_offset ?? 0;
  const fill = zarray.fill_value ?? 65535;
  const codecId = compressorId(zarray);
  const ty = Math.floor(latIndex / latChunk);
  const tx = Math.floor(lonIndex / lonChunk);
  // Leads and depths are packed into a single chunk (index 0 on both axes).
  const record = await fetchChunk(store, variable, `${startIndex}.0.0.${ty}.${tx}`, codecId);
  const stored = new Uint16Array(record.bytes.buffer, record.bytes.byteOffset, record.bytes.byteLength / 2);
  const localLat = latIndex - ty * latChunk;
  const localLon = lonIndex - tx * lonChunk;
  const values = new Float32Array(leadSize * depthSize);
  for (let lead = 0; lead < leadSize; lead += 1) {
    for (let depth = 0; depth < depthSize; depth += 1) {
      const flat = ((lead * depthChunk + depth) * latChunk + localLat) * lonChunk + localLon;
      const raw = stored[flat];
      values[lead * depthSize + depth] = raw === fill ? NaN : raw * scale + offset;
    }
  }
  return { values, leads: leadSize, depths: depthSize, compressedBytes: record.compressedBytes };
}

/** Warm the chunk cache for a layer without decoding — used for lead-day prefetch. */
export function prefetchLayer(store, { variable, level, startIndex, leadIndex }) {
  const path = `level/${level}/${variable}`;
  const { zarray } = arrayMetadata(store, path);
  const [, , latitudeSize, longitudeSize] = zarray.shape;
  const [, , latitudeChunk, longitudeChunk] = zarray.chunks;
  const codecId = compressorId(zarray);
  const latitudeTiles = Math.ceil(latitudeSize / latitudeChunk);
  const longitudeTiles = Math.ceil(longitudeSize / longitudeChunk);
  for (let ty = 0; ty < latitudeTiles; ty += 1) {
    for (let tx = 0; tx < longitudeTiles; tx += 1) {
      fetchChunk(store, path, `${startIndex}.${leadIndex}.${ty}.${tx}`, codecId).catch(() => {});
    }
  }
}
