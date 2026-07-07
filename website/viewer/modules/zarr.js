// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Minimal zarr v2 reader for OceanBench viewer pyramids (contracts.md §6).
//
// The layout is fixed and known: a consolidated `.zmetadata`, groups `level/<k>`,
// data arrays of shape (start_date, lead_day, latitude, longitude) chunked as
// 256x256 spatial tiles with one (start_date, lead_day) per chunk, stored as
// uint16 with per-variable scale_factor/add_offset and an explicit _FillValue,
// DEFLATE-compressed. Decompression is the platform-native DecompressionStream —
// no wasm codec — which is exactly why the builder writes zlib rather than blosc.
//
// This is deliberately a few hundred lines for our own layout, not a general
// zarr client. It scales to multi-level pyramids because every read is driven by
// the array's own .zarray chunk grid, not by any 1-degree assumption.

async function inflate(compressed, codecId) {
  if (codecId === null || codecId === undefined) return new Uint8Array(compressed);
  if (codecId !== "zlib" && codecId !== "gzip") {
    throw new Error(`Unsupported compressor '${codecId}'. Pyramids must be zlib/gzip for browser decode.`);
  }
  const format = codecId === "gzip" ? "gzip" : "deflate";
  const stream = new Blob([compressed]).stream().pipeThrough(new DecompressionStream(format));
  const buffer = await new Response(stream).arrayBuffer();
  return new Uint8Array(buffer);
}

export async function loadStore(storeUrl) {
  const base = storeUrl.replace(/\/$/, "");
  const response = await fetch(`${base}/.zmetadata`, { cache: "no-cache" });
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
  const response = await fetch(manifestUrl, { cache: "no-cache" });
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

async function fetchChunk(store, path, chunkKey, codecId) {
  const cacheKey = `${path}/${chunkKey}`;
  const cached = store.chunkCache.get(cacheKey);
  if (cached) return cached;
  const url = `${store.baseUrl}/${path}/${chunkKey}`;
  const started = performance.now();
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Cannot fetch chunk ${url} (${response.status})`);
  const compressed = await response.arrayBuffer();
  const bytes = await inflate(compressed, codecId);
  const record = { bytes, compressedBytes: compressed.byteLength, milliseconds: performance.now() - started };
  store.chunkCache.set(cacheKey, record);
  return record;
}

/**
 * Read one 2D layer — a single (start_date, lead_day) slice of one variable at
 * one pyramid level — into a Float32Array in real units (NaN over land).
 * Returns { data, width, height, compressedBytes, fetchMilliseconds }.
 */
export async function readLayer(store, { variable, level, startIndex, leadIndex }) {
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
      tileReads.push({ ty, tx, record: fetchChunk(store, path, `${startIndex}.${leadIndex}.${ty}.${tx}`, codecId) });
    }
  }
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
