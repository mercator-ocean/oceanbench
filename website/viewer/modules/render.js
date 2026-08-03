// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Field colorization and colorbar drawing for the viewer. Rendering stays on the
// 2D canvas: a decoded Float32 field becomes an ImageData at native grid
// resolution (pixel = grid cell), which the map view blits with a pan/zoom
// transform (nearest-neighbour, so the native zoom is pixel-faithful — §6).

import { lookupTable } from "../vendor/cmocean/colormaps.js";

// Land is a flat unsaturated grey, the oceanographic convention: it sits far from both
// ends of every field palette in both themes, so a continent never reads as an extreme
// value. "Sea but unobserved" in the entire-year error raster keeps a faint blue tint,
// clearly bluer than land, so land, no-obs ocean and scored ocean each read distinctly
// (year scope only; single-forecast fields keep land ↔ ocean as before).
// The live values come from the shared design tokens (tokens.css, --ob-viewer-land and
// --ob-viewer-no-obs, themed by data-theme on <html>); the per-theme literals below are
// the fallback kept in sync with that file.
const LAND_LIGHT = [190, 184, 174];
const LAND_DARK = [132, 130, 126];
export const NO_OBS_LIGHT = [222, 231, 243];
export const NO_OBS_DARK = [42, 54, 72];

// getComputedStyle flushes style, so resolve each token once per theme rather than on
// every colorize call (the lead scrub recolorizes a field per frame).
const tokenCache = new Map();

function themeColor(name, fallback) {
  // Key on the theme the document is actually wearing, since that is what
  // getComputedStyle resolves against.
  const key = `${name}|${document.documentElement.dataset.theme}`;
  const cached = tokenCache.get(key);
  if (cached) return cached;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const parsed = parseHexColor(value) || fallback;
  tokenCache.set(key, parsed);
  return parsed;
}

function parseHexColor(text) {
  const match = /^#([0-9a-f]{6})$/i.exec(text);
  if (!match) return null;
  const value = parseInt(match[1], 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

export function landColor(theme) {
  return themeColor("--ob-viewer-land", theme === "light" ? LAND_LIGHT : LAND_DARK);
}

export function noObsColor(theme) {
  return themeColor("--ob-viewer-no-obs", theme === "light" ? NO_OBS_LIGHT : NO_OBS_DARK);
}

/**
 * Colorize a field into an ImageData sized to the grid.
 * `range` is [minimum, maximum] in real units; values are clamped.
 * `flipVertical` flips rows so ascending-latitude data renders north-up.
 */
/**
 * Opaque where the field has no data, transparent everywhere else: the same cells
 * fieldToImageData paints with the land colour, in the same layout, so blitting this
 * through the field's own world path lands exactly on the drawn coastline. Used to erase
 * the particle layer over land.
 */
export function landStencilImageData({ data, width, height }, options = {}) {
  const { flipVertical = false } = options;
  const image = new ImageData(width, height);
  const pixels = image.data;
  for (let row = 0; row < height; row += 1) {
    const sourceRow = flipVertical ? height - 1 - row : row;
    for (let column = 0; column < width; column += 1) {
      if (!Number.isNaN(data[sourceRow * width + column])) continue;
      pixels[(row * width + column) * 4 + 3] = 255;
    }
  }
  return image;
}

export function fieldToImageData({ data, width, height }, colormapName, range, options = {}) {
  const { flipVertical = false, theme = "dark", transparentNaN = false, landMask = null } = options;
  const lut = lookupTable(colormapName);
  const [minimum, maximum] = range;
  const span = maximum - minimum || 1;
  const land = landColor(theme);
  const noObs = noObsColor(theme);
  const image = new ImageData(width, height);
  const pixels = image.data;
  for (let row = 0; row < height; row += 1) {
    const sourceRow = flipVertical ? height - 1 - row : row;
    for (let column = 0; column < width; column += 1) {
      const sourceIndex = sourceRow * width + column;
      const value = data[sourceIndex];
      const destination = (row * width + column) * 4;
      if (Number.isNaN(value)) {
        // With a land mask (year scope): land renders opaque; NaN over ocean is
        // "sea but unobserved" and gets the faint no-obs tint. Without a mask,
        // transparentNaN drops the cell; otherwise NaN is land as before.
        if (landMask) {
          const fill = landMask[sourceIndex] ? land : noObs;
          pixels[destination] = fill[0];
          pixels[destination + 1] = fill[1];
          pixels[destination + 2] = fill[2];
          pixels[destination + 3] = 255;
          continue;
        }
        if (transparentNaN) {
          pixels[destination + 3] = 0;
          continue;
        }
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

/**
 * cos(latitude)-weighted mean over the finite cells of a field — a true area-weighted
 * spatial mean, since equal-area on a lat/lon grid shrinks with cos(latitude) toward the
 * poles. `latitudes` are the field's row coordinates (length === height). Falls back to
 * the unweighted mean when no latitudes are supplied.
 */
export function areaWeightedMean({ data, width, height }, latitudes) {
  if (!latitudes || latitudes.length !== height) return fieldStatistics({ data }).mean;
  let weightedSum = 0;
  let weightTotal = 0;
  for (let row = 0; row < height; row += 1) {
    const weight = Math.max(0, Math.cos((latitudes[row] * Math.PI) / 180));
    for (let column = 0; column < width; column += 1) {
      const value = data[row * width + column];
      if (Number.isNaN(value)) continue;
      weightedSum += value * weight;
      weightTotal += weight;
    }
  }
  return weightTotal === 0 ? NaN : weightedSum / weightTotal;
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
    // Left-align and ellipsize on the right so the leading, most meaningful part of
    // a long label ("mean |obs − model| …") stays readable instead of overflowing
    // and being clipped on the left by centre alignment.
    context.textAlign = "left";
    context.fillText(fitLabel(context, label, width), 0, 1);
  }
  context.textAlign = "left";
  context.fillText(formatTick(range[0]), 0, barTop + barHeight + 3);
  context.textAlign = "right";
  context.fillText(formatTick(range[1]), width, barTop + barHeight + 3);
}

function fitLabel(context, label, maxWidth) {
  if (context.measureText(label).width <= maxWidth) return label;
  const ellipsis = "…";
  let truncated = label;
  while (truncated.length > 1 && context.measureText(truncated + ellipsis).width > maxWidth) {
    truncated = truncated.slice(0, -1);
  }
  return truncated + ellipsis;
}

/** The one minus sign the viewer prints, matching the "model − obs" captions. */
export const MINUS_SIGN = "−";

/**
 * The one numeric convention every readout uses: a fixed number of decimals, no
 * signed zero ("−0.000" is just "0.000"), one minus sign character, and "n/a" for
 * anything that is not a finite number.
 */
export function formatFixed(value, decimals) {
  if (!Number.isFinite(value)) return "n/a";
  const rounded = Number(value.toFixed(decimals));
  return (rounded === 0 ? 0 : rounded).toFixed(decimals).replace("-", MINUS_SIGN);
}

function formatTick(value) {
  if (!Number.isFinite(value)) return "n/a";
  const absolute = Math.abs(value);
  if (absolute !== 0 && (absolute < 0.01 || absolute >= 10000)) return value.toExponential(1).replace("-", MINUS_SIGN);
  return formatFixed(value, absolute < 1 ? 3 : 2);
}
