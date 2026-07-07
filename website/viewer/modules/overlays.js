// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Insight overlays drawn on a panel's overlay canvas (contracts.md §6: overlays
// with purpose-modes, never all at once). Two live modes here — eddy census
// (matched / spurious / missed contours vs a reference) and Class-4 obs points
// coloured by |obs − model|. All drawing is in
// normalized-world → device-pixel space via the panel's `project`, so overlays
// stay registered to the field under pan, zoom and difference views.

import { sample as sampleColormap } from "../vendor/cmocean/colormaps.js";

// Category palette (luminous on the dark canvas; still legible on light).
export const EDDY_COLORS = {
  matched: "#3ddc97", // model eddy that pairs with a reference eddy
  spurious: "#ff9f45", // model eddy with no reference counterpart
  missed: "#c77dff", // reference eddy the model failed to produce
};
export const CLASS4_COLORMAP = "thermal";

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
      drawing.moveTo(point.x, point.y); // antimeridian wrap — break the stroke
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

/**
 * Draw one eddy frame. Matched pairs show the model contour (solid) and its
 * reference contour (dashed) joined by a displacement connector; spurious and
 * missed show a single contour each. Returns a legend summary for the rail.
 */
export function drawEddyFrame(drawing, project, frame, options = {}) {
  const ratio = options.devicePixelRatio || 1;
  const dotRadius = 2.5 * ratio;
  drawing.lineWidth = 1.5 * ratio;
  drawing.lineJoin = "round";

  for (const eddy of frame.missed || []) {
    drawing.setLineDash([]);
    drawing.strokeStyle = EDDY_COLORS.missed;
    tracePolygon(drawing, project, eddy.contour_longitude, eddy.contour_latitude);
    drawing.stroke();
    centerDot(drawing, project, eddy, EDDY_COLORS.missed, dotRadius);
  }
  for (const eddy of frame.spurious || []) {
    drawing.setLineDash([]);
    drawing.strokeStyle = EDDY_COLORS.spurious;
    tracePolygon(drawing, project, eddy.contour_longitude, eddy.contour_latitude);
    drawing.stroke();
    centerDot(drawing, project, eddy, EDDY_COLORS.spurious, dotRadius);
  }
  for (const match of frame.matches || []) {
    const model = match.challenger;
    const reference = match.reference;
    drawing.strokeStyle = EDDY_COLORS.matched;
    drawing.setLineDash([]);
    tracePolygon(drawing, project, model.contour_longitude, model.contour_latitude);
    drawing.stroke();
    drawing.setLineDash([4 * ratio, 3 * ratio]);
    tracePolygon(drawing, project, reference.contour_longitude, reference.contour_latitude);
    drawing.stroke();
    drawing.setLineDash([]);
    const a = project(...normPair(model.longitude, model.latitude));
    const b = project(...normPair(reference.longitude, reference.latitude));
    drawing.strokeStyle = "rgba(61, 220, 151, 0.5)";
    drawing.beginPath();
    drawing.moveTo(a.x, a.y);
    drawing.lineTo(b.x, b.y);
    drawing.stroke();
    centerDot(drawing, project, model, EDDY_COLORS.matched, dotRadius);
  }
  drawing.setLineDash([]);
  return {
    matched: (frame.matches || []).length,
    spurious: (frame.spurious || []).length,
    missed: (frame.missed || []).length,
    lead_day: frame.lead_day,
  };
}

// Neutral colour for eddies both forecasts agree on. Only-in-F1 / only-in-F2 use the
// canonical forecast colours supplied by the caller — no category implies truth.
export const EDDY_MATCHED_COLOR = EDDY_COLORS.matched;

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

function normPair(longitude, latitude) {
  const { nx, ny } = toNorm(longitude, latitude);
  return [nx, ny];
}

/**
 * Scatter Class-4 obs points coloured by |obs − model|. `errorScale` is the error
 * that maps to the top of the colormap (robust p95 from the caller). Points outside
 * the view are cheaply skipped by the projection returning off-canvas coordinates.
 */
export function drawClass4Points(drawing, project, points, options = {}) {
  const ratio = options.devicePixelRatio || 1;
  const radius = (options.radius || 2.2) * ratio;
  const diameter = radius * 2;
  const scale = options.errorScale || 1;
  const width = options.canvasWidth || Infinity;
  const height = options.canvasHeight || Infinity;
  const bucketCount = 18;
  const buckets = Array.from({ length: bucketCount }, () => []);
  for (const point of points) {
    const { nx, ny } = toNorm(point.longitude, point.latitude);
    const screen = project(nx, ny);
    if (screen.x < -8 || screen.y < -8 || screen.x > width + 8 || screen.y > height + 8) continue;
    const error = class4AbsoluteError(point);
    const normalized = Number.isFinite(error) ? Math.min(1, error / scale) : 0;
    const bucketIndex = Math.min(bucketCount - 1, Math.max(0, Math.floor(normalized * (bucketCount - 1))));
    buckets[bucketIndex].push(screen.x, screen.y);
  }
  for (let bucketIndex = 0; bucketIndex < buckets.length; bucketIndex += 1) {
    const coordinates = buckets[bucketIndex];
    if (!coordinates.length) continue;
    const normalized = bucketCount <= 1 ? 0 : bucketIndex / (bucketCount - 1);
    const [r, g, b] = sampleColormap(CLASS4_COLORMAP, 0.12 + normalized * 0.88);
    drawing.fillStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
    for (let i = 0; i < coordinates.length; i += 2) {
      drawing.fillRect(coordinates[i] - radius, coordinates[i + 1] - radius, diameter, diameter);
    }
  }
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

/** Robust upper error bound (~p90) for the Class-4 colour scale. */
export function class4ErrorScale(points) {
  if (!points.length) return 1;
  const errors = points.map((point) => class4AbsoluteError(point)).filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!errors.length) return 1;
  const index = Math.min(errors.length - 1, Math.floor(errors.length * 0.9));
  return errors[index] || errors[errors.length - 1] || 1;
}
