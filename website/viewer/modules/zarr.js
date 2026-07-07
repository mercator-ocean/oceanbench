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
