// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Period scores for the rail curves, re-aggregated in the browser from the per-start file
// insights/<slug>/<region>/scores-by-start.json (values[lead_index][start_index]) over the
// starts that fall in a period:
// - gridded RMSE (vs GLORYS, vs GLO12): mean over starts of the per-start RMSE, as the year;
// - Class 4 RMSE (vs observations): pooled sqrt(sum n_i rmsd_i^2 / sum n_i); bias pooled by n;
// - the 95% band is a percentile bootstrap over the period's starts (1000 draws, seeded, so
//   the band is the same on every load). The generator differs from numpy's, so a band is
//   close to, not equal to, what the publish step would give.
// The whole year never comes through here: it keeps the published rows and band.
//
// Pure module with no DOM access, so it runs under Node too.

export const BOOTSTRAP_DRAWS = 1000;
export const BOOTSTRAP_SEED = 20240703;

// mulberry32: a small seeded generator.
function seededRandom(seed) {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let mixed = state;
    mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
    mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
    return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
  };
}

const multiplicityCache = new Map();

// One row per draw: how many times each start of the period was drawn.
function bootstrapMultiplicities(startCount, draws = BOOTSTRAP_DRAWS, seed = BOOTSTRAP_SEED) {
  const key = `${startCount}:${draws}:${seed}`;
  if (multiplicityCache.has(key)) return multiplicityCache.get(key);
  const random = seededRandom(seed);
  const multiplicities = new Float64Array(draws * startCount);
  for (let draw = 0; draw < draws; draw += 1) {
    for (let pick = 0; pick < startCount; pick += 1) multiplicities[draw * startCount + Math.floor(random() * startCount)] += 1;
  }
  multiplicityCache.set(key, multiplicities);
  return multiplicities;
}

// numpy.percentile with linear interpolation, over the finite values.
function percentile(values, percent) {
  const finite = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!finite.length) return NaN;
  const position = (percent / 100) * (finite.length - 1);
  const low = Math.floor(position);
  const high = Math.min(low + 1, finite.length - 1);
  return finite[low] + (finite[high] - finite[low]) * (position - low);
}

const finite = (value) => value != null && Number.isFinite(Number(value));

// Indices (into data.dates) of the starts inside [from, to] (ISO dates, inclusive).
export function periodStartIndices(data, range) {
  const indices = [];
  (data?.dates || []).forEach((date, index) => {
    const day = String(date).slice(0, 10);
    if (day >= range.from && day <= range.to) indices.push(index);
  });
  return indices;
}

// Point value and band of one lead over the chosen starts. `values` and `counts` are the
// per-start arrays of that lead; counts null means a plain mean (gridded).
function aggregateLead(values, counts, starts) {
  const value = [];
  const weight = [];
  for (const index of starts) {
    const v = values?.[index];
    const n = counts ? counts[index] : 1;
    if (!finite(v) || !finite(n) || Number(n) <= 0) {
      value.push(NaN);
      weight.push(0);
    } else {
      value.push(counts ? Number(v) * Number(v) * Number(n) : Number(v));
      weight.push(Number(n));
    }
  }
  const used = weight.filter((w) => w > 0).length;
  if (!used) return null;
  const finish = (total, totalWeight) => (totalWeight > 0 ? (counts ? Math.sqrt(total / totalWeight) : total / totalWeight) : NaN);
  let total = 0;
  let totalWeight = 0;
  value.forEach((v, position) => {
    if (weight[position] > 0) {
      total += v;
      totalWeight += weight[position];
    }
  });
  const mean = finish(total, totalWeight);
  const startCount = starts.length;
  const multiplicities = bootstrapMultiplicities(startCount);
  const draws = new Array(BOOTSTRAP_DRAWS);
  for (let draw = 0; draw < BOOTSTRAP_DRAWS; draw += 1) {
    let drawTotal = 0;
    let drawWeight = 0;
    const offset = draw * startCount;
    for (let position = 0; position < startCount; position += 1) {
      const m = multiplicities[offset + position];
      if (!m || !(weight[position] > 0)) continue;
      drawTotal += m * value[position];
      drawWeight += m * weight[position];
    }
    draws[draw] = finish(drawTotal, drawWeight);
  }
  return { mean, ci_low: percentile(draws, 2.5), ci_high: percentile(draws, 97.5), starts: used };
}

const sameDepth = (a, b) => String(a ?? "") === String(b ?? "");

// The per-start series matching a published summary row key, or null.
export function findSeries(data, { metric, reference, variable, depth }) {
  const class4 = metric === "class4_rmsd";
  const block = class4 ? data?.class4 : data?.gridded;
  if (!block || block.available === false || !Array.isArray(block.series)) return null;
  return (
    block.series.find(
      (series) =>
        (series.metric ?? metric) === metric &&
        (class4 || series.reference === reference) &&
        series.variable === variable &&
        sameDepth(series.depth, depth),
    ) || null
  );
}

// Period rows for one summary key: [{ lead_day, mean, ci_low, ci_high, n_starts }] or null
// when the file has no per-start values for that key.
export function periodLeadRows(data, key, range) {
  const series = findSeries(data, key);
  if (!series) return null;
  const starts = periodStartIndices(data, range);
  if (!starts.length) return null;
  const class4 = key.metric === "class4_rmsd";
  const rows = [];
  (data.lead_days || []).forEach((leadDay, leadIndex) => {
    const result = aggregateLead(series.rmsd?.[leadIndex], class4 ? series.n?.[leadIndex] : null, starts);
    if (result) rows.push({ lead_day: Number(leadDay), mean: result.mean, ci_low: result.ci_low, ci_high: result.ci_high, n_starts: result.starts });
  });
  return rows.length ? rows : null;
}

// Class 4 RMSE, bias and n of one variable and depth bin at one lead over a period's starts.
export function periodClass4Bin(data, variable, depth, leadDay, range) {
  const series = findSeries(data, { metric: "class4_rmsd", reference: "observations", variable, depth });
  if (!series) return null;
  const leads = (data.lead_days || []).map(Number);
  if (!leads.length) return null;
  let leadIndex = 0;
  leads.forEach((lead, index) => {
    if (Math.abs(lead - leadDay) < Math.abs(leads[leadIndex] - leadDay)) leadIndex = index;
  });
  let squares = 0;
  let biasTotal = 0;
  let count = 0;
  let used = 0;
  for (const index of periodStartIndices(data, range)) {
    const rmsd = series.rmsd?.[leadIndex]?.[index];
    const bias = series.bias?.[leadIndex]?.[index];
    const n = series.n?.[leadIndex]?.[index];
    if (!finite(rmsd) || !finite(n) || Number(n) <= 0) continue;
    squares += Number(rmsd) * Number(rmsd) * Number(n);
    if (finite(bias)) biasTotal += Number(bias) * Number(n);
    count += Number(n);
    used += 1;
  }
  if (!count) return null;
  return { rmsd: Math.sqrt(squares / count), bias: biasTotal / count, n: count, starts: used, lead: leads[leadIndex] };
}
