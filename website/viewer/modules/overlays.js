// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Insight overlays drawn on a panel's overlay canvas (contracts.md §6: overlays
// with purpose-modes, never all at once). Two live modes here — eddy census
// (matched / spurious / missed contours vs a reference) and Class-4 obs points
// coloured by |obs − model| — plus a stubbed trajectory mode. All drawing is in
// normalized-world → device-pixel space via the panel's `project`, so overlays
// stay registered to the field under pan, zoom and difference views.

import { sample as sampleColormap } from "../vendor/cmocean/colormaps.js";

// Category palette (luminous on the dark canvas; still legible on light).
export const EDDY_COLORS = {
  matched: "#3ddc97", // model eddy that pairs with a reference eddy
  spurious: "#ff9f45", // model eddy with no reference counterpart
  missed: "#c77dff", // reference eddy the model failed to produce
};
const CLASS4_COLORMAP = "thermal";

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
  const scale = options.errorScale || 1;
  const width = options.canvasWidth || Infinity;
  const height = options.canvasHeight || Infinity;
  for (const point of points) {
    const { nx, ny } = toNorm(point.longitude, point.latitude);
    const screen = project(nx, ny);
    if (screen.x < -8 || screen.y < -8 || screen.x > width + 8 || screen.y > height + 8) continue;
    const normalized = Math.min(1, Math.abs(point.abs_error) / scale);
    const [r, g, b] = sampleColormap(CLASS4_COLORMAP, 0.12 + normalized * 0.88);
    drawing.fillStyle = `rgba(${r}, ${g}, ${b}, 0.9)`;
    drawing.beginPath();
    drawing.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
    drawing.fill();
  }
}

/** Robust upper error bound (~p90) for the Class-4 colour scale. */
export function class4ErrorScale(points) {
  if (!points.length) return 1;
  const errors = points.map((point) => Math.abs(point.abs_error)).sort((a, b) => a - b);
  const index = Math.min(errors.length - 1, Math.floor(errors.length * 0.9));
  return errors[index] || errors[errors.length - 1] || 1;
}
