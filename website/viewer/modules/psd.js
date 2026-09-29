// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Live client-side power spectral density of a geographic box of a field.
// The viewer already holds the decoded field for the selected variable/lead/level
// as a Float32 grid; this module crops it to the box ON ITS NATIVE GRID, fills land
// (NaN) with the box mean, removes the mean, applies a separable Hann window over the
// native cells, zero-pads to a power of two per axis, runs a radix-2 2D FFT, and sums
// |F|² over annular wavenumber rings into an isotropic power-vs-wavenumber curve.
// No cell is resampled: the transform sees exactly the cells the model published.
//
// Normalization: the ring sum is divided by the true cell count (nx·ny), the padded
// transform size (Nx·Ny, the DFT's own Parseval factor), the window's mean square and
// the ring width in wavenumber. The curve is then a one-dimensional isotropic spectral
// density (field units² per cycle/km) whose integral over wavenumber is the variance of
// the field in the box, less the modes beyond the last ring.
//
// Rings are one cycle per SHORTER box side apart and stop at 1/(2·max(dx, dy)), the
// Nyquist wavenumber of the coarser axis: past it a ring would hold modes along one
// axis only and read low. dx and dy are kept apart and rings are binned by PHYSICAL
// wavenumber, so a box that is square in degrees but not in kilometres (high latitude)
// is still binned correctly.

const MAX_CELLS = 512; // at most this many native cells per axis enter the transform
const EARTH_KM_PER_DEGREE = 111.32;

/**
 * Isotropic (ring-summed) PSD of the box of `field` given as a normalized-world
 * viewport. Returns { wavelength: number[] (metres), power: number[] (field-units² per
 * cycle/km), samples, cellKm, oceanFraction }, ordered from the longest wavelength down,
 * or null when the box is too small or mostly land.
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
  const nx = Math.min(MAX_CELLS, columns.count);
  const ny = Math.min(MAX_CELLS, rows.count);

  const box = nativeBox(field, rows, columns, nx, ny);
  if (!box) return null;

  const centreLatitude = (latLow + latHigh) / 2;
  const lonStep = Math.abs(longitudes[1] - longitudes[0]);
  const latStep = Math.abs(latitudes[1] - latitudes[0]);
  const dxKm = lonStep * EARTH_KM_PER_DEGREE * Math.cos((centreLatitude * Math.PI) / 180);
  const dyKm = latStep * EARTH_KM_PER_DEGREE;
  if (!(dxKm > 0) || !(dyKm > 0)) return null;

  const ringWidth = 1 / Math.min(nx * dxKm, ny * dyKm); // cycles per km
  const nyquist = 1 / (2 * Math.max(dxKm, dyKm));
  const rings = ringSummedPower(box.data, box.paddedX, box.paddedY, dxKm, dyKm, ringWidth, nyquist);
  const scale = 1 / (nx * ny * box.paddedX * box.paddedY * box.windowPower * ringWidth);
  const wavelength = [];
  const power = [];
  for (let r = 1; r < rings.length; r += 1) {
    if (rings[r].count === 0) continue;
    wavelength.push((1 / (r * ringWidth)) * 1000); // metres, chart converts to km
    power.push(rings[r].sum * scale);
  }
  if (!wavelength.length) return null;
  return { wavelength, power, samples: nx * ny, cellKm: Math.sqrt(dxKm * dyKm), oceanFraction: box.oceanFraction };
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

// The native cells of the box, land (NaN) filled with the box mean, the mean removed,
// a separable Hann window applied over the nx×ny native cells, then zero-padded to the
// next power of two along each axis. A box with more than MAX_CELLS cells along an axis
// keeps its first MAX_CELLS.
function nativeBox(field, rows, columns, nx, ny) {
  const paddedX = powerOfTwoAtLeast(nx);
  const paddedY = powerOfTwoAtLeast(ny);
  const sourceColumnAt = columns.indexAt
    ? (ordinal) => columns.indexAt(ordinal / (columns.count - 1))
    : (ordinal) => columns.start + ordinal;
  const raw = new Float64Array(nx * ny).fill(NaN);
  let sum = 0;
  let finiteCount = 0;
  for (let y = 0; y < ny; y += 1) {
    const base = (rows.start + y) * field.width;
    for (let x = 0; x < nx; x += 1) {
      const value = field.data[base + sourceColumnAt(x)];
      if (Number.isNaN(value)) continue;
      raw[y * nx + x] = value;
      sum += value;
      finiteCount += 1;
    }
  }
  const oceanFraction = finiteCount / (nx * ny);
  if (oceanFraction < 0.25) return null; // mostly land, no meaningful spectrum
  const mean = sum / finiteCount;
  const hannX = hannWindow(nx);
  const hannY = hannWindow(ny);
  // Mean square of the separable window over the true cells, the factor by which the
  // taper lowers the variance of whatever it multiplies; dividing it out restores the
  // untapered variance.
  let windowSquares = 0;
  const data = new Float64Array(paddedX * paddedY);
  for (let y = 0; y < ny; y += 1) {
    for (let x = 0; x < nx; x += 1) {
      const value = raw[y * nx + x];
      const detrended = Number.isNaN(value) ? 0 : value - mean;
      const weight = hannY[y] * hannX[x];
      data[y * paddedX + x] = detrended * weight;
      windowSquares += weight * weight;
    }
  }
  return { data, paddedX, paddedY, oceanFraction, windowPower: windowSquares / (nx * ny) };
}

function hannWindow(length) {
  const window = new Float64Array(length);
  for (let i = 0; i < length; i += 1) window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (length - 1));
  return window;
}

function powerOfTwoAtLeast(value) {
  let power = 1;
  while (power < value) power *= 2;
  return power;
}

// 2D FFT (rows then columns) of the padded box, summing |F|² into annular bins of the
// physical wavenumber magnitude, `ringWidth` cycles/km apart, up to `nyquist`.
function ringSummedPower(box, paddedX, paddedY, dxKm, dyKm, ringWidth, nyquist) {
  const real = Float64Array.from(box);
  const imaginary = new Float64Array(paddedX * paddedY);
  const rowReal = new Float64Array(paddedX);
  const rowImaginary = new Float64Array(paddedX);
  for (let y = 0; y < paddedY; y += 1) {
    const base = y * paddedX;
    for (let x = 0; x < paddedX; x += 1) {
      rowReal[x] = real[base + x];
      rowImaginary[x] = 0;
    }
    fastFourierTransform(rowReal, rowImaginary);
    for (let x = 0; x < paddedX; x += 1) {
      real[base + x] = rowReal[x];
      imaginary[base + x] = rowImaginary[x];
    }
  }
  const columnReal = new Float64Array(paddedY);
  const columnImaginary = new Float64Array(paddedY);
  for (let x = 0; x < paddedX; x += 1) {
    for (let y = 0; y < paddedY; y += 1) {
      columnReal[y] = real[y * paddedX + x];
      columnImaginary[y] = imaginary[y * paddedX + x];
    }
    fastFourierTransform(columnReal, columnImaginary);
    for (let y = 0; y < paddedY; y += 1) {
      real[y * paddedX + x] = columnReal[y];
      imaginary[y * paddedX + x] = columnImaginary[y];
    }
  }

  const maxRing = Math.floor(nyquist / ringWidth);
  const bins = Array.from({ length: maxRing + 1 }, () => ({ sum: 0, count: 0 }));
  for (let y = 0; y < paddedY; y += 1) {
    const ky = (y <= paddedY / 2 ? y : y - paddedY) / (paddedY * dyKm);
    for (let x = 0; x < paddedX; x += 1) {
      const kx = (x <= paddedX / 2 ? x : x - paddedX) / (paddedX * dxKm);
      const ring = Math.round(Math.hypot(kx, ky) / ringWidth);
      if (ring > maxRing) continue;
      const index = y * paddedX + x;
      bins[ring].sum += real[index] * real[index] + imaginary[index] * imaginary[index];
      bins[ring].count += 1;
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
  // fieldA/fieldB are already aligned onto the same grid by the caller (B was
  // block-averaged or nearest-sampled onto A's grid), so their difference is defined.
  void resample;
  const difference = { data: new Float32Array(fieldA.data.length), width: fieldA.width, height: fieldA.height };
  for (let i = 0; i < difference.data.length; i += 1) difference.data[i] = fieldA.data[i] - fieldB.data[i];
  return boxPowerSpectrum(difference, latitudesA, longitudesA, viewport);
}
