// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// How numbers and coordinates are written for the reader. Kept apart from the code that
// computes them so a change to a readout's wording never touches a computation.

export function formatLatLon(lon, lat) {
  const latText = `${Math.abs(lat).toFixed(2)}°${lat >= 0 ? "N" : "S"}`;
  const lonText = `${Math.abs(lon).toFixed(2)}°${lon >= 0 ? "E" : "W"}`;
  return `${latText}, ${lonText}`;
}

export function megabytes(bytes) {
  return (Number(bytes) / (1024 * 1024)).toFixed(1);
}

export function formatCount(value) {
  return Math.round(Number(value) || 0).toLocaleString("en-US");
}

export function escapeHtml(value) {
  return value.replace(/[&<>"']/g, (character) => {
    return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[character];
  });
}

// Human label for a grid cell size: 1/12° for fractional-degree grids, 0.5° otherwise.
export function cellDegreesLabel(cellDeg) {
  const inverse = 1 / cellDeg;
  if (inverse > 1.01 && Math.abs(inverse - Math.round(inverse)) < 0.05) return `1/${Math.round(inverse)}°`;
  return `${Number(cellDeg.toFixed(2))}°`;
}

export function rgbCss(rgb) {
  return `rgb(${rgb[0]}, ${rgb[1]}, ${rgb[2]})`;
}
