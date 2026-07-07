// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Live client-side power spectral density of the currently visible viewport box.
// The viewer already holds the decoded field for the selected variable/lead/level
// as a Float32 grid; this module crops it to the visible geographic box, fills land
// (NaN) with the box mean, removes the mean, applies a separable Hann window, runs a
// radix-2 2D FFT, and radially averages |F|² into a power-vs-wavenumber curve. The
// wavenumber axis is converted to physical wavelength in kilometres using a
// latitude-aware cell size at the box centre, so the same routine works for every
// variable, model and region — spectra from a 1° model simply stop at a coarser
// wavelength than a 1/12° one, which is honest and expected.
//
// Method (surfaced in the chart caption/tooltip): Hann window + mean-fill of land,
// mean removed. This is a pragmatic estimate for exploration, not a calibrated
// realism metric (the precomputed spectra battery on the scores page remains the
// reference for that).

const MAX_SIDE = 256; // resample the box to a square power-of-two grid of at most this side
const EARTH_KM_PER_DEGREE = 111.32;

/**
 * Radially-averaged PSD of the visible box of `field` for the given normalized-world
 * viewport. Returns { wavelength: number[] (metres), power: number[], samples,
 * cellKm } sorted by ascending wavelength, or null when the box is too small/empty.
 */
export function boxPowerSpectrum(field, latitudes, longitudes, viewport) {
  if (!field || !latitudes || !longitudes) return null;
  const lonMin = viewport.minX * 360 - 180;
  const lonMax = viewport.maxX * 360 - 180;
  // ny grows southward, so maxY maps to the smaller latitude.
  const latHigh = 90 - viewport.minY * 180;
  const latLow = 90 - viewport.maxY * 180;

  const columns = longitudeRange(longitudes, Math.min(lonMin, lonMax), Math.max(lonMin, lonMax));
  const rows = coordinateRange(latitudes, Math.min(latLow, latHigh), Math.max(latLow, latHigh));
  if (!columns || !rows) return null;
  if (columns.count < 8 || rows.count < 8) return null;

  const side = powerOfTwoAtMost(Math.min(MAX_SIDE, columns.count, rows.count));
  if (side < 8) return null;

  const box = resampleBox(field, rows, columns, side);
  if (!box) return null;

  const centreLatitude = (latLow + latHigh) / 2;
  const boxWidthKm = Math.abs(lonMax - lonMin) * EARTH_KM_PER_DEGREE * Math.cos((centreLatitude * Math.PI) / 180);
  const boxHeightKm = Math.abs(latHigh - latLow) * EARTH_KM_PER_DEGREE;
  const cellKm = (boxWidthKm + boxHeightKm) / (2 * side);

  const radial = radialAveragedPower(box.data, side);
  const wavelength = [];
  const power = [];
  for (let r = 1; r < radial.length; r += 1) {
    if (radial[r].count === 0) continue;
    const wavelengthSamples = side / r; // one radial ring = r cycles across the box
    wavelength.push(wavelengthSamples * cellKm * 1000); // metres, chart converts to km
    power.push(radial[r].sum / radial[r].count);
  }
  if (!wavelength.length) return null;
  return { wavelength, power, samples: side * side, cellKm, oceanFraction: box.oceanFraction };
}

function coordinateRange(coordinates, lowValue, highValue) {
  const size = coordinates.length;
  if (size < 2) return null;
  const step = coordinates[1] - coordinates[0];
  const ascending = step > 0;
  let low = Infinity;
  let high = -Infinity;
  for (let i = 0; i < size; i += 1) {
    const value = coordinates[i];
    if (value < lowValue || value > highValue) continue;
    if (i < low) low = i;
    if (i > high) high = i;
  }
  if (!Number.isFinite(low) || high <= low) return null;
  return { start: low, end: high, count: high - low + 1, ascending };
}

function longitudeRange(longitudes, lowValue, highValue) {
  const size = longitudes.length;
  if (size < 2) return null;
  const step = longitudes[1] - longitudes[0];
  const periodic = Math.abs(step) * size >= 359;
  if (!periodic) return coordinateRange(longitudes, lowValue, highValue);

  const count = Math.min(size, Math.floor(Math.abs(highValue - lowValue) / Math.abs(step)) + 1);
  if (count < 2) return null;
  const origin = longitudes[0];
  return {
    count,
    indexAt(fraction) {
      const longitude = lowValue + fraction * (highValue - lowValue);
      const unwrappedIndex = (longitude - origin) / step;
      return Math.round(((unwrappedIndex % size) + size) % size) % size;
    },
  };
}

function powerOfTwoAtMost(value) {
  let power = 1;
  while (power * 2 <= value) power *= 2;
  return power;
}

// Nearest-sample resample of the field sub-box into a square `side`×`side` grid,
// mean-filling land (NaN), removing the mean, then applying a separable Hann window.
function resampleBox(field, rows, columns, side) {
  const raw = new Float64Array(side * side);
  const filled = new Uint8Array(side * side);
  let sum = 0;
  let finiteCount = 0;
  for (let y = 0; y < side; y += 1) {
    const sourceRow = rows.start + Math.round((y / (side - 1)) * (rows.count - 1));
    for (let x = 0; x < side; x += 1) {
      const fraction = x / (side - 1);
      const sourceColumn = columns.indexAt
        ? columns.indexAt(fraction)
        : columns.start + Math.round(fraction * (columns.count - 1));
      const value = field.data[sourceRow * field.width + sourceColumn];
      const target = y * side + x;
      if (Number.isNaN(value)) {
        filled[target] = 0;
      } else {
        raw[target] = value;
        filled[target] = 1;
        sum += value;
        finiteCount += 1;
      }
    }
  }
  const oceanFraction = finiteCount / (side * side);
  if (oceanFraction < 0.25) return null; // mostly land — no meaningful spectrum
  const mean = sum / finiteCount;
  const hann = new Float64Array(side);
  for (let i = 0; i < side; i += 1) hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (side - 1));
  for (let y = 0; y < side; y += 1) {
    for (let x = 0; x < side; x += 1) {
      const target = y * side + x;
      const detrended = (filled[target] ? raw[target] : mean) - mean;
      raw[target] = detrended * hann[y] * hann[x];
    }
  }
  return { data: raw, oceanFraction };
}

// 2D FFT (rows then columns) of a real box, radially averaging |F|² into bins by the
// integer wavenumber magnitude. Uses a shared in-place radix-2 FFT over rows/columns.
function radialAveragedPower(box, side) {
  const real = Float64Array.from(box);
  const imaginary = new Float64Array(side * side);
  const rowReal = new Float64Array(side);
  const rowImaginary = new Float64Array(side);

  for (let y = 0; y < side; y += 1) {
    const base = y * side;
    for (let x = 0; x < side; x += 1) {
      rowReal[x] = real[base + x];
      rowImaginary[x] = 0;
    }
    fastFourierTransform(rowReal, rowImaginary);
    for (let x = 0; x < side; x += 1) {
      real[base + x] = rowReal[x];
      imaginary[base + x] = rowImaginary[x];
    }
  }
  const columnReal = new Float64Array(side);
  const columnImaginary = new Float64Array(side);
  for (let x = 0; x < side; x += 1) {
    for (let y = 0; y < side; y += 1) {
      columnReal[y] = real[y * side + x];
      columnImaginary[y] = imaginary[y * side + x];
    }
    fastFourierTransform(columnReal, columnImaginary);
    for (let y = 0; y < side; y += 1) {
      real[y * side + x] = columnReal[y];
      imaginary[y * side + x] = columnImaginary[y];
    }
  }

  const maxRadius = Math.floor(side / 2);
  const bins = Array.from({ length: maxRadius + 1 }, () => ({ sum: 0, count: 0 }));
  for (let y = 0; y < side; y += 1) {
    const ky = y <= side / 2 ? y : y - side;
    for (let x = 0; x < side; x += 1) {
      const kx = x <= side / 2 ? x : x - side;
      const radius = Math.round(Math.hypot(kx, ky));
      if (radius > maxRadius) continue;
      const index = y * side + x;
      const magnitude = real[index] * real[index] + imaginary[index] * imaginary[index];
      bins[radius].sum += magnitude;
      bins[radius].count += 1;
    }
  }
  return bins;
}

// In-place iterative radix-2 Cooley–Tukey FFT (length must be a power of two).
function fastFourierTransform(real, imaginary) {
  const n = real.length;
  for (let i = 1, j = 0; i < n; i += 1) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const tempReal = real[i];
      real[i] = real[j];
      real[j] = tempReal;
      const tempImaginary = imaginary[i];
      imaginary[i] = imaginary[j];
      imaginary[j] = tempImaginary;
    }
  }
  for (let length = 2; length <= n; length <<= 1) {
    const angle = (-2 * Math.PI) / length;
    const wReal = Math.cos(angle);
    const wImaginary = Math.sin(angle);
    for (let start = 0; start < n; start += length) {
      let curReal = 1;
      let curImaginary = 0;
      for (let k = 0; k < length / 2; k += 1) {
        const evenIndex = start + k;
        const oddIndex = start + k + length / 2;
        const oddReal = real[oddIndex] * curReal - imaginary[oddIndex] * curImaginary;
        const oddImaginary = real[oddIndex] * curImaginary + imaginary[oddIndex] * curReal;
        real[oddIndex] = real[evenIndex] - oddReal;
        imaginary[oddIndex] = imaginary[evenIndex] - oddImaginary;
        real[evenIndex] += oddReal;
        imaginary[evenIndex] += oddImaginary;
        const nextReal = curReal * wReal - curImaginary * wImaginary;
        curImaginary = curReal * wImaginary + curImaginary * wReal;
        curReal = nextReal;
      }
    }
  }
}

/** Difference (error) spectrum of two boxes: PSD of (fieldA − fieldB) over the box. */
export function differenceBoxSpectrum(fieldA, latitudesA, longitudesA, fieldB, viewport, resample) {
  // fieldA/fieldB are already aligned onto the same grid by the caller (resample=true
  // means B was resampled onto A's grid), so their difference is well defined.
  void resample;
  const difference = { data: new Float32Array(fieldA.data.length), width: fieldA.width, height: fieldA.height };
  for (let i = 0; i < difference.data.length; i += 1) difference.data[i] = fieldA.data[i] - fieldB.data[i];
  return boxPowerSpectrum(difference, latitudesA, longitudesA, viewport);
}
