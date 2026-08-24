// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// INVARIANT: drawClass4Points and drawClass4Screen must bucket, colour and place points
// identically. They differ only in where the device coordinates come from: the first
// projects them, the second is handed a precomputed pair of arrays. Any change to one is
// a change to both, and the render fingerprints in qa/ compare only the projecting path.
//
// Insight overlays drawn on a panel's overlay canvas (contracts.md §6: overlays
// with purpose-modes, never all at once). Two live modes here — eddy census
// (matched / spurious / missed contours vs a reference) and Class-4 obs points
// coloured by |obs − model|. All drawing is in
// normalized-world → device-pixel space via the panel's `project`, so overlays
// stay registered to the field under pan, zoom and difference views.

import { sample as sampleColormap } from "../vendor/cmocean/colormaps.js";

// Category palette (luminous on the dark canvas; still legible on light).
export const CLASS4_COLORMAP = "thermal";

// The obs points paint the upper part of the colormap only: the darkest sliver is not
// distinguishable from the muted field underneath. The colorbar has to draw the SAME
// segment, or the key names colours the points never use and a reader matches a dot to
// the wrong error. One definition, used by the points and by the bar.
export const CLASS4_RAMP_START = 0.12;
export const CLASS4_RAMP_END = 1;

function class4RampPosition(normalized) {
  return CLASS4_RAMP_START + normalized * (CLASS4_RAMP_END - CLASS4_RAMP_START);
}

function toNorm(longitude, latitude) {
  return { nx: (longitude + 180) / 360, ny: (90 - latitude) / 180 };
}

function tracePolygon(drawing, project, longitudes, latitudes) {
  drawing.beginPath();
  let previousNx = null;
  for (let i = 0; i < longitudes.length; i += 1) {
    const { nx, ny } = toNorm(longitudes[i], latitudes[i]);
    const point = project(nx, ny);
    if (previousNx !== null && Math.abs(nx - previousNx) > 0.5) {
      drawing.moveTo(point.x, point.y); // antimeridian wrap, break the stroke
    } else if (i === 0) {
      drawing.moveTo(point.x, point.y);
    } else {
      drawing.lineTo(point.x, point.y);
    }
    previousNx = nx;
  }
  drawing.closePath();
}

function centerDot(drawing, project, eddy, color, radius) {
  const { nx, ny } = toNorm(eddy.longitude, eddy.latitude);
  const point = project(nx, ny);
  drawing.fillStyle = color;
  drawing.beginPath();
  drawing.arc(point.x, point.y, radius, 0, Math.PI * 2);
  drawing.fill();
}

// Neutral colour for eddies both forecasts agree on. Only-in-F1 / only-in-F2 use the
// canonical forecast colours supplied by the caller — no category implies truth.
export const EDDY_MATCHED_COLOR = "#3ddc97";

const EARTH_RADIUS_KM = 6371.0088;

function haversineDistanceKm(a, b) {
  const toRadians = Math.PI / 180;
  const latitude1 = a.latitude * toRadians;
  const latitude2 = b.latitude * toRadians;
  const deltaLatitude = (b.latitude - a.latitude) * toRadians;
  const deltaLongitude = (b.longitude - a.longitude) * toRadians;
  const h =
    Math.sin(deltaLatitude / 2) ** 2 +
    Math.cos(latitude1) * Math.cos(latitude2) * Math.sin(deltaLongitude / 2) ** 2;
  return 2 * EARTH_RADIUS_KM * Math.asin(Math.min(1, Math.sqrt(h)));
}

/**
 * Symmetric in-browser pairing of two forecast eddy censuses. Mirrors the offline
 * matcher's rules (oceanbench/core/eddies.match_mesoscale_eddies): pairing is per
 * polarity only, on great-circle centre distance, capped at `maxDistanceKm` (200 km,
 * DEFAULT_MATCH_DISTANCE_KM). The offline matcher solves an optimal assignment; here a
 * greedy nearest-first pass (shortest candidate pairs consumed first) approximates it,
 * which is what the design calls for. Returns matched pairs plus the eddies only one
 * forecast produced — neither side is a reference.
 */
export function matchCensuses(detectionsA, detectionsB, maxDistanceKm = 200) {
  const candidates = [];
  for (const a of detectionsA) {
    for (const b of detectionsB) {
      if (a.polarity !== b.polarity) continue;
      const distanceKm = haversineDistanceKm(a, b);
      if (distanceKm <= maxDistanceKm) candidates.push({ a, b, distanceKm });
    }
  }
  candidates.sort((first, second) => first.distanceKm - second.distanceKm);
  const usedA = new Set();
  const usedB = new Set();
  const matched = [];
  for (const candidate of candidates) {
    if (usedA.has(candidate.a) || usedB.has(candidate.b)) continue;
    usedA.add(candidate.a);
    usedB.add(candidate.b);
    matched.push(candidate);
  }
  const onlyA = detectionsA.filter((eddy) => !usedA.has(eddy));
  const onlyB = detectionsB.filter((eddy) => !usedB.has(eddy));
  const meanDisplacementKm = matched.length
    ? matched.reduce((sum, pair) => sum + pair.distanceKm, 0) / matched.length
    : NaN;
  return { matched, onlyA, onlyB, meanDisplacementKm };
}

/** Draw a set of eddy contours + centre dots in a single colour. */
export function drawEddyDetections(drawing, project, detections, color, options = {}) {
  const ratio = options.devicePixelRatio || 1;
  const dotRadius = 2.5 * ratio;
  drawing.lineWidth = 1.5 * ratio;
  drawing.lineJoin = "round";
  drawing.setLineDash([]);
  drawing.strokeStyle = color;
  for (const eddy of detections || []) {
    tracePolygon(drawing, project, eddy.contour_longitude, eddy.contour_latitude);
    drawing.stroke();
    centerDot(drawing, project, eddy, color, dotRadius);
  }
}

// The two Class-4 draws below reach the canvas through this one painter. They differ only
// in where the screen coordinates come from (projected here, or projected once per frame by
// the caller); the colour bucketing, alpha and square geometry are the same picture, and a
// change made to one of them but not the other silently draws the same observations two
// different ways.
const CLASS4_BUCKET_COUNT = 18;

function makeClass4Buckets() {
  return Array.from({ length: CLASS4_BUCKET_COUNT }, () => []);
}

function class4BucketIndex(error, scale) {
  const normalized = Math.min(1, error / scale);
  return Math.min(CLASS4_BUCKET_COUNT - 1, Math.max(0, Math.floor(normalized * (CLASS4_BUCKET_COUNT - 1))));
}

function paintClass4Buckets(drawing, buckets, radius) {
  const diameter = radius * 2;
  for (let bucketIndex = 0; bucketIndex < buckets.length; bucketIndex += 1) {
    const coordinates = buckets[bucketIndex];
    if (!coordinates.length) continue;
    const normalized = CLASS4_BUCKET_COUNT <= 1 ? 0 : bucketIndex / (CLASS4_BUCKET_COUNT - 1);
    const [r, g, b] = sampleColormap(CLASS4_COLORMAP, class4RampPosition(normalized));
    drawing.fillStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
    for (let i = 0; i < coordinates.length; i += 2) {
      drawing.fillRect(coordinates[i] - radius, coordinates[i + 1] - radius, diameter, diameter);
    }
  }
}

/**
 * Scatter Class-4 obs points coloured by |obs − model|. `errorScale` is the error
 * that maps to the top of the colormap (robust p95 from the caller). Points outside
 * the view are cheaply skipped by the projection returning off-canvas coordinates.
 */
export function drawClass4Points(drawing, project, points, options = {}) {
  const ratio = options.devicePixelRatio || 1;
  const radius = (options.radius || 2.2) * ratio;
  const scale = options.errorScale || 1;
  const width = options.canvasWidth || Infinity;
  const height = options.canvasHeight || Infinity;
  const buckets = makeClass4Buckets();
  for (const point of points) {
    const { nx, ny } = toNorm(point.longitude, point.latitude);
    const screen = project(nx, ny);
    if (screen.x < -8 || screen.y < -8 || screen.x > width + 8 || screen.y > height + 8) continue;
    const error = class4AbsoluteError(point);
    // A masked model gives model_value = NaN and a non-finite error. The obs is real, but
    // there is no comparison to colour, so skip it rather than paint it as bucket 0 (the
    // darkest, lowest-error colour — a phantom "perfect match" near coastlines).
    if (!Number.isFinite(error)) continue;
    buckets[class4BucketIndex(error, scale)].push(screen.x, screen.y);
  }
  paintClass4Buckets(drawing, buckets, radius);
}

/**
 * Draw Class-4 points from precomputed device-pixel coordinates — the projection has
 * already happened once for the frame (viewport-culled) so nothing is re-projected here.
 * `screenX`/`screenY` hold the coordinates of `count` candidate points at one world-copy;
 * `pointIds[t]` is the index into the frame's parallel `error` array. `selectedMask`, when
 * present, restricts drawing to the stride-thinned display subset (mask indexed by point id).
 * Colour bucketing, alpha and fillRect geometry are identical to `drawClass4Points`.
 */
export function drawClass4Screen(drawing, screenX, screenY, pointIds, count, error, selectedMask, options = {}) {
  const ratio = options.devicePixelRatio || 1;
  const radius = (options.radius || 2.2) * ratio;
  const scale = options.errorScale || 1;
  const buckets = makeClass4Buckets();
  for (let t = 0; t < count; t += 1) {
    const id = pointIds[t];
    if (selectedMask && !selectedMask[id]) continue;
    const value = error[id];
    // Prepared points already dropped masked (non-finite error) rows; keep the guard so the
    // colour bucketing stays identical to drawClass4Points even if a NaN ever slips through.
    if (!Number.isFinite(value)) continue;
    buckets[class4BucketIndex(value, scale)].push(screenX[t], screenY[t]);
  }
  paintClass4Buckets(drawing, buckets, radius);
}

/**
 * Ring the obs point under the cursor so the hover readout is visibly attached to a dot.
 * Drawn on every visible world copy, in the caller's note colour, as a stroked circle
 * just outside the dot: nothing is filled, so the point keeps its own error colour.
 */
export function drawClass4HoverRing(drawing, project, point, options = {}) {
  const ratio = options.devicePixelRatio || 1;
  const radius = (options.radius || 2.2) * ratio + 3 * ratio;
  const { nx, ny } = toNorm(point.longitude, point.latitude);
  const screen = project(nx, ny);
  drawing.save();
  drawing.setLineDash([]);
  drawing.lineWidth = 1.6 * ratio;
  drawing.strokeStyle = options.color || "#e5edf5";
  drawing.beginPath();
  drawing.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
  drawing.stroke();
  drawing.restore();
}

// Absolute obs−model error for a match-up row. Prefers a precomputed `abs_error`
// column, otherwise derives it from `observation_value`/`model_value`.
export function class4AbsoluteError(point) {
  const abs = numericOrNaN(point.abs_error);
  if (Number.isFinite(abs)) return Math.abs(abs);
  const obs = numericOrNaN(point.observation_value);
  const model = numericOrNaN(point.model_value);
  if (Number.isFinite(obs) && Number.isFinite(model)) return Math.abs(model - obs);
  return NaN;
}

// Number() maps null/"" to 0, which would fake a real value; treat those as missing.
export function numericOrNaN(value) {
  if (value == null || value === "") return NaN;
  const number = Number(value);
  return Number.isFinite(number) ? number : NaN;
}

/**
 * Robust upper error bound (~p90) for the Class-4 colour scale, or NaN when the point
 * set carries nothing to measure. A set with no points (or none with a finite error)
 * used to answer 1, and that placeholder is a plausible error in every unit the viewer
 * plots: fed into the grow-only ramp it pinned the scale at 1 for the rest of the
 * selection and painted real SLA errors (p90 ≈ 0.09 m) as near-zero. NaN cannot be
 * mistaken for a measurement, so the caller keeps the last real scale instead.
 */
export function class4ErrorScale(points) {
  if (!points.length) return NaN;
  const errors = points.map((point) => class4AbsoluteError(point)).filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!errors.length) return NaN;
  const index = Math.min(errors.length - 1, Math.floor(errors.length * 0.9));
  return errors[index] || errors[errors.length - 1] || NaN;
}
