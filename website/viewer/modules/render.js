// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Field colorization and colorbar drawing for the viewer. Rendering stays on the
// 2D canvas: a decoded Float32 field becomes an ImageData at native grid
// resolution (pixel = grid cell), which the map view blits with a pan/zoom
// transform (nearest-neighbour, so the native zoom is pixel-faithful — §6).

import { lookupTable } from "../vendor/cmocean/colormaps.js";

const LAND_LIGHT = [214, 219, 226];
const LAND_DARK = [22, 27, 35];

/**
 * Colorize a field into an ImageData sized to the grid.
 * `range` is [minimum, maximum] in real units; values are clamped.
 * `flipVertical` flips rows so ascending-latitude data renders north-up.
 */
export function fieldToImageData({ data, width, height }, colormapName, range, options = {}) {
  const { flipVertical = false, theme = "dark" } = options;
  const lut = lookupTable(colormapName);
  const [minimum, maximum] = range;
  const span = maximum - minimum || 1;
  const land = theme === "light" ? LAND_LIGHT : LAND_DARK;
  const image = new ImageData(width, height);
  const pixels = image.data;
  for (let row = 0; row < height; row += 1) {
    const sourceRow = flipVertical ? height - 1 - row : row;
    for (let column = 0; column < width; column += 1) {
      const value = data[sourceRow * width + column];
      const destination = (row * width + column) * 4;
      if (Number.isNaN(value)) {
        pixels[destination] = land[0];
        pixels[destination + 1] = land[1];
        pixels[destination + 2] = land[2];
        pixels[destination + 3] = 255;
        continue;
      }
      let normalized = (value - minimum) / span;
      normalized = normalized <= 0 ? 0 : normalized >= 1 ? 1 : normalized;
      const lutIndex = Math.round(normalized * 255) * 3;
      pixels[destination] = lut[lutIndex];
      pixels[destination + 1] = lut[lutIndex + 1];
      pixels[destination + 2] = lut[lutIndex + 2];
      pixels[destination + 3] = 255;
    }
  }
  return image;
}

/** Statistics over the finite entries of a field. */
export function fieldStatistics({ data }) {
  let minimum = Infinity;
  let maximum = -Infinity;
  let sum = 0;
  let count = 0;
  for (let i = 0; i < data.length; i += 1) {
    const value = data[i];
    if (Number.isNaN(value)) continue;
    if (value < minimum) minimum = value;
    if (value > maximum) maximum = value;
    sum += value;
    count += 1;
  }
  if (count === 0) return { minimum: NaN, maximum: NaN, mean: NaN, count: 0 };
  return { minimum, maximum, mean: sum / count, count };
}

/** Symmetric range about zero covering a difference field's magnitude. */
export function symmetricRange({ data }) {
  let magnitude = 0;
  for (let i = 0; i < data.length; i += 1) {
    const value = data[i];
    if (Number.isNaN(value)) continue;
    const absolute = Math.abs(value);
    if (absolute > magnitude) magnitude = absolute;
  }
  const bound = magnitude || 1;
  return [-bound, bound];
}

/**
 * Resample a field onto a target regular lat/lon grid by nearest cell, leaving
 * NaN where the target lies outside the source. The viewer's datasets share the
 * 1° spacing but not the same origin (GLONET spans 168 rows, GLORYS/GLO12 170),
 * so differencing must register on coordinates, never on raw array index.
 */
export function resampleOntoGrid(field, sourceLatitudes, sourceLongitudes, targetLatitudes, targetLongitudes) {
  const targetHeight = targetLatitudes.length;
  const targetWidth = targetLongitudes.length;
  const data = new Float32Array(targetHeight * targetWidth).fill(NaN);
  const latitudeStep = sourceLatitudes.length > 1 ? sourceLatitudes[1] - sourceLatitudes[0] : 1;
  const longitudeStep = sourceLongitudes.length > 1 ? sourceLongitudes[1] - sourceLongitudes[0] : 1;
  const latitudeOrigin = sourceLatitudes[0];
  const longitudeOrigin = sourceLongitudes[0];
  for (let row = 0; row < targetHeight; row += 1) {
    const sourceRow = Math.round((targetLatitudes[row] - latitudeOrigin) / latitudeStep);
    if (sourceRow < 0 || sourceRow >= sourceLatitudes.length) continue;
    if (Math.abs(sourceLatitudes[sourceRow] - targetLatitudes[row]) > Math.abs(latitudeStep) * 0.5) continue;
    for (let column = 0; column < targetWidth; column += 1) {
      const sourceColumn = Math.round((targetLongitudes[column] - longitudeOrigin) / longitudeStep);
      if (sourceColumn < 0 || sourceColumn >= sourceLongitudes.length) continue;
      if (Math.abs(sourceLongitudes[sourceColumn] - targetLongitudes[column]) > Math.abs(longitudeStep) * 0.5) continue;
      data[row * targetWidth + column] = field.data[sourceRow * field.width + sourceColumn];
    }
  }
  return { data, width: targetWidth, height: targetHeight };
}

/** Elementwise A − B of two aligned fields (NaN where either is land). */
export function differenceField(fieldA, fieldB) {
  const data = new Float32Array(fieldA.data.length);
  for (let i = 0; i < data.length; i += 1) data[i] = fieldA.data[i] - fieldB.data[i];
  return { data, width: fieldA.width, height: fieldA.height };
}

/** Draw a horizontal colorbar with min/max labels into a canvas. */
export function drawColorbar(canvas, colormapName, range, { label = "", textColor = "#e6edf3" } = {}) {
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  context.clearRect(0, 0, width, height);
  const lut = lookupTable(colormapName);
  const labelHeight = label ? 18 : 0;
  const barTop = labelHeight;
  const barHeight = Math.min(14, height - labelHeight - 18);
  const gradient = context.createImageData(width, barHeight);
  for (let column = 0; column < width; column += 1) {
    const lutIndex = Math.round((column / (width - 1)) * 255) * 3;
    for (let row = 0; row < barHeight; row += 1) {
      const destination = (row * width + column) * 4;
      gradient.data[destination] = lut[lutIndex];
      gradient.data[destination + 1] = lut[lutIndex + 1];
      gradient.data[destination + 2] = lut[lutIndex + 2];
      gradient.data[destination + 3] = 255;
    }
  }
  context.putImageData(gradient, 0, barTop);
  context.fillStyle = textColor;
  context.font = "11px system-ui, sans-serif";
  context.textBaseline = "top";
  if (label) {
    context.textAlign = "center";
    context.fillText(label, width / 2, 1);
  }
  context.textAlign = "left";
  context.fillText(formatTick(range[0]), 0, barTop + barHeight + 3);
  context.textAlign = "right";
  context.fillText(formatTick(range[1]), width, barTop + barHeight + 3);
}

function formatTick(value) {
  if (!Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  if (absolute !== 0 && (absolute < 0.01 || absolute >= 10000)) return value.toExponential(1);
  return value.toFixed(absolute < 1 ? 3 : 2);
}
