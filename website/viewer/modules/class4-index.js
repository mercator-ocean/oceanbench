// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// The Class-4 observation point set, prepared once and drawn many times: a coarse
// lon/lat bucket grid over the points, the frame draw that walks only the buckets the
// viewport touches, the match count the legend reports, and the depth bin an observation
// belongs to. No view state of its own; the caller passes the zoom the thinning decision
// needs, because the display budget is a property of the frame, not of this index.

import { drawClass4Screen, class4AbsoluteError } from "./overlays.js";
import { class4Points, class4ParquetVariable } from "./insights.js";
import { isCurrentsVariable } from "./variables.js";

const CLASS4_INDEX_COLS = 360;
const CLASS4_INDEX_ROWS = 180;
// Above this zoom every visible observation is drawn: the reader is looking at a small
// enough patch that thinning would hide real structure rather than declutter it.
const CLASS4_FULL_DENSITY_ZOOM = 12;
const CLASS4_DISPLAY_POINT_BUDGET = 18000;

// Build parallel typed arrays (normalized lon/lat + abs error) and a coarse lon/lat
// bucket grid (CSR layout: `start` offsets + `order` point ids) once per point set.
// The grid lets pan/zoom frames project only the buckets overlapping the viewport and
// lets the hover hit-test scan the cursor's bucket neighbourhood instead of every point.
export function buildClass4Index(points) {
  const count = points.length;
  const normalizedLongitude = new Float32Array(count);
  const normalizedLatitude = new Float32Array(count);
  const absoluteError = new Float32Array(count);
  const cols = CLASS4_INDEX_COLS;
  const rows = CLASS4_INDEX_ROWS;
  const bucketCounts = new Int32Array(cols * rows);
  const bucketOfPoint = new Int32Array(count);
  for (let i = 0; i < count; i += 1) {
    const point = points[i];
    const nx = (point.longitude + 180) / 360;
    const ny = (90 - point.latitude) / 180;
    normalizedLongitude[i] = nx;
    normalizedLatitude[i] = ny;
    absoluteError[i] = class4AbsoluteError(point);
    let column = Math.floor(nx * cols);
    if (column < 0) column = 0;
    else if (column >= cols) column = cols - 1;
    let row = Math.floor(ny * rows);
    if (row < 0) row = 0;
    else if (row >= rows) row = rows - 1;
    const bucket = row * cols + column;
    bucketOfPoint[i] = bucket;
    bucketCounts[bucket] += 1;
  }
  const start = new Int32Array(cols * rows + 1);
  for (let bucket = 0; bucket < cols * rows; bucket += 1) start[bucket + 1] = start[bucket] + bucketCounts[bucket];
  const order = new Int32Array(count);
  const cursor = Int32Array.from(start.subarray(0, cols * rows));
  for (let i = 0; i < count; i += 1) {
    const bucket = bucketOfPoint[i];
    order[cursor[bucket]] = i;
    cursor[bucket] += 1;
  }
  return {
    normalizedLongitude,
    normalizedLatitude,
    absoluteError,
    spatialIndex: { cols, rows, start, order },
    // Frame-stamp visibility marker: avoids clearing an N-length mask every frame.
    visibilityStamp: new Int32Array(count),
    frameStamp: 0,
    candidateScratch: new Int32Array(count),
    columnScratch: new Uint8Array(cols),
  };
}

// Project the viewport-overlapping buckets once, decide stride-thinning, and draw — no
// second projection pass. Returns the exact same `visibleTotal` / `drawnVisible` / `stride`
// / `thinned` semantics as the previous per-point scan, plus the display selection mask the
// hover hit-test reuses. `selectedMask` is null when nothing is thinned (whole set drawn).
export function drawClass4Frame(context, projection, prepared, copyOffsets, canvas, options) {
  const normalizedLongitude = prepared.normalizedLongitude;
  const normalizedLatitude = prepared.normalizedLatitude;
  const absoluteError = prepared.absoluteError;
  const count = normalizedLongitude.length;
  if (!count) return { visibleTotal: 0, drawnVisible: 0, stride: 1, thinned: false, selectedMask: null };
  const { cols, rows, start, order } = prepared.spatialIndex;
  const { originX, originY, displayWidth, displayHeight } = projection;
  const width = canvas.width;
  const height = canvas.height;
  // Viewport bounds in base normalized coordinates (before world-copy offset).
  const nyTop = (0 - originY) / displayHeight;
  const nyBottom = (height - originY) / displayHeight;
  const nxLeft = (0 - originX) / displayWidth;
  const nxRight = (width - originX) / displayWidth;
  let rowLow = Math.floor(Math.min(nyTop, nyBottom) * rows) - 1;
  let rowHigh = Math.floor(Math.max(nyTop, nyBottom) * rows) + 1;
  rowLow = Math.max(0, rowLow);
  rowHigh = Math.min(rows - 1, rowHigh);
  const columnMask = prepared.columnScratch;
  columnMask.fill(0);
  for (const offset of copyOffsets) {
    let low = Math.min(nxLeft - offset, nxRight - offset);
    let high = Math.max(nxLeft - offset, nxRight - offset);
    low = Math.max(0, low);
    high = Math.min(0.9999999, high);
    if (low > high) continue;
    let columnLow = Math.floor(low * cols) - 1;
    let columnHigh = Math.floor(high * cols) + 1;
    columnLow = Math.max(0, columnLow);
    columnHigh = Math.min(cols - 1, columnHigh);
    for (let column = columnLow; column <= columnHigh; column += 1) columnMask[column] = 1;
  }
  // Gather candidate point ids from the overlapping buckets.
  const candidates = prepared.candidateScratch;
  let candidateCount = 0;
  if (rowLow <= rowHigh) {
    for (let row = rowLow; row <= rowHigh; row += 1) {
      const rowBase = row * cols;
      for (let column = 0; column < cols; column += 1) {
        if (!columnMask[column]) continue;
        const bucket = rowBase + column;
        for (let s = start[bucket]; s < start[bucket + 1]; s += 1) candidates[candidateCount++] = order[s];
      }
    }
  }
  const stamp = prepared.visibilityStamp;
  const frameStamp = (prepared.frameStamp += 1);
  let visibleTotal = 0;
  // Per world-copy: project each candidate once, cull off-canvas, keep the device-pixel
  // coordinates for the draw pass (single projection). Mark visibility across copies.
  const perCopy = [];
  for (const offset of copyOffsets) {
    const screenX = new Float32Array(candidateCount);
    const screenY = new Float32Array(candidateCount);
    const pointIds = new Int32Array(candidateCount);
    let onCanvas = 0;
    for (let t = 0; t < candidateCount; t += 1) {
      const id = candidates[t];
      const x = originX + (normalizedLongitude[id] + offset) * displayWidth;
      if (x < 0 || x > width) continue;
      const y = originY + normalizedLatitude[id] * displayHeight;
      if (y < 0 || y > height) continue;
      screenX[onCanvas] = x;
      screenY[onCanvas] = y;
      pointIds[onCanvas] = id;
      onCanvas += 1;
      if (stamp[id] !== frameStamp) {
        stamp[id] = frameStamp;
        visibleTotal += 1;
      }
    }
    perCopy.push({ screenX, screenY, pointIds, onCanvas });
  }
  let thinned = false;
  let stride = 1;
  let drawnVisible = visibleTotal;
  let selectedMask = null;
  if (!(options.zoom >= CLASS4_FULL_DENSITY_ZOOM) && visibleTotal > CLASS4_DISPLAY_POINT_BUDGET) {
    thinned = true;
    stride = Math.ceil(visibleTotal / CLASS4_DISPLAY_POINT_BUDGET);
    selectedMask = new Uint8Array(count);
    // Walk point ids in ascending order (identical to the old point-order scan) so the
    // stride selection picks exactly the same points, then count the drawn subset.
    let visiblePosition = 0;
    let selectedCount = 0;
    for (let id = 0; id < count; id += 1) {
      if (stamp[id] !== frameStamp) continue;
      if (visiblePosition % stride === 0) {
        selectedMask[id] = 1;
        selectedCount += 1;
      }
      visiblePosition += 1;
    }
    drawnVisible = selectedCount;
  }
  for (const copy of perCopy) {
    drawClass4Screen(context, copy.screenX, copy.screenY, copy.pointIds, copy.onCanvas, absoluteError, selectedMask, options);
  }
  return { visibleTotal, drawnVisible, stride, thinned, selectedMask };
}

// Number of Class-4 rows matching the active selector before spatial thinning — the
// "of M sampled" denominator the legend reports so low counts read as weak (item 5).
export function countClass4Matches(rows, { variable, depthBin, leadDay, startDate }) {
  if (isCurrentsVariable(variable)) return class4Points(rows, { variable, depthBin, leadDay, startDate }).length;
  if (!rows) return 0;
  const parquetVariable = class4ParquetVariable(variable);
  const requestedLead = leadDay == null ? null : Number(leadDay);
  let total = 0;
  for (const row of rows) {
    if (row.variable !== parquetVariable) continue;
    if (depthBin && row.depth_bin !== depthBin) continue;
    if (requestedLead !== null && Number(row.lead_day) !== requestedLead) continue;
    if (startDate && String(row.start_date).slice(0, 10) !== startDate) continue;
    // Match the drawn set: masked-model rows (non-finite error) are not drawn, so they
    // must not inflate the "N obs" / "of Y" denominator either.
    if (!Number.isFinite(class4AbsoluteError(row))) continue;
    total += 1;
  }
  return total;
}

export function class4DepthBin(entry) {
  if (!entry) return null;
  if (entry.depth === "15m") return "15m";
  if (entry.standard_name.includes("velocity")) return "15m";
  if (entry.standard_name === "sea_surface_height_above_geoid") return "surface";
  return "0-5m"; // temperature / salinity near-surface bin matching the surface viewer field
}

export function class4DepthLabel(entry, depthBin) {
  if (!entry) return depthBin || "selected depth";
  if (entry.depth && entry.depth !== "surface") return entry.depth;
  return depthBin || entry.depth || "selected depth";
}
