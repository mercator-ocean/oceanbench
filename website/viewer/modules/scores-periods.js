// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Period scores for the Scores page: the year aggregation of oceanbench/publish/aggregate.py
// (aggregate_scores) run in the browser on the forecasts whose start date falls in a chosen
// range, read from the per-start rows of scores.parquet.
//
// The published summary decides which rows exist and which baseline each skill is paired
// with (its skill_baseline), so a period row is always the same row as the year one, just
// averaged over fewer starts:
// - mean over the present starts is a plain mean, except class4_rmsd, pooled as
//   sqrt(sum(v^2 n) / sum(n));
// - skill is 1 - model / baseline, both aggregated over the starts common to the two;
// - the 95% interval is a percentile bootstrap over the period's starts, one resample of the
//   starts shared by every row and by the model and its baseline. The generator differs from
//   numpy's, so the bounds are close to, not equal to, the publish step's.
//
// Pure module with no DOM access, so the parity check in qa/ runs it under Node.

import { parquetRead } from "../vendor/hyparquet/hyparquet.min.js";

const COLUMNS = ["challenger", "year", "region", "metric", "reference", "variable", "depth", "lead_day", "start_date", "band", "polarity", "value", "n"];
const DIAGNOSTIC_METRICS = new Set(["grid_coverage"]);
const CLASS4_METRIC = "class4_rmsd";
const SKILL_FIELD = "skill_vs_persistence_1_degree";
const NULL_KEY = "\u0000";
const DAY_MS = 86400000;

export const BOOTSTRAP_DRAWS = 1000;
export const BOOTSTRAP_SEED = 20240703;
const CONFIDENCE = 0.95;

function keyPart(value) {
  return value === null || value === undefined ? NULL_KEY : String(value);
}

/** Identity of a row: challenger, year, region and the metric key, nulls equal to nulls. */
export function identityKey(row, challenger = row.challenger) {
  return [challenger, row.year, row.region, row.metric, row.reference, row.variable, row.depth, row.lead_day, row.band, row.polarity]
    .map(keyPart)
    .join("|");
}

/**
 * Read the per-start rows into groups keyed by identity. Rows without a start date
 * (year-level metrics) and diagnostic rows are dropped, as aggregate_scores drops them.
 */
export async function readPerStartScores(buffer) {
  const file = { byteLength: buffer.byteLength, slice: (start, end) => buffer.slice(start, end) };
  const columns = {};
  await parquetRead({
    file,
    columns: COLUMNS,
    onChunk: ({ columnName, columnData, rowStart }) => {
      const target = (columns[columnName] ??= []);
      for (let index = 0; index < columnData.length; index += 1) target[rowStart + index] = columnData[index];
    },
  });

  const startSet = new Set();
  for (const start of columns.start_date) if (start) startSet.add(start.getTime());
  const starts = [...startSet].sort((first, second) => first - second);
  const startPosition = new Map(starts.map((time, position) => [time, position]));

  const groups = new Map();
  const rowCount = columns.value.length;
  for (let index = 0; index < rowCount; index += 1) {
    const start = columns.start_date[index];
    if (!start || DIAGNOSTIC_METRICS.has(columns.metric[index])) continue;
    const key = identityKey({
      challenger: columns.challenger[index],
      year: columns.year[index],
      region: columns.region[index],
      metric: columns.metric[index],
      reference: columns.reference[index],
      variable: columns.variable[index],
      depth: columns.depth[index],
      lead_day: columns.lead_day[index],
      band: columns.band[index],
      polarity: columns.polarity[index],
    });
    let group = groups.get(key);
    if (!group) {
      group = { starts: [], values: [], counts: [], isClass4: columns.metric[index] === CLASS4_METRIC };
      groups.set(key, group);
    }
    group.starts.push(startPosition.get(start.getTime()));
    group.values.push(columns.value[index]);
    group.counts.push(columns.n[index] ?? 0);
  }
  return { starts, groups };
}

/** Positions on the start axis of the forecasts starting inside [from, to], both "YYYY-MM-DD". */
export function periodStartIndices(perStart, from, to) {
  const low = Date.parse(`${from}T00:00:00Z`);
  const high = Date.parse(`${to}T00:00:00Z`) + DAY_MS - 1;
  const indices = [];
  perStart.starts.forEach((time, position) => {
    if (time >= low && time <= high) indices.push(position);
  });
  return indices;
}

// mulberry32: a small seeded generator, so a period's interval is the same on every load.
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

// One row per bootstrap draw: how many times each start of the period was drawn. The draws
// only enter through these multiplicities, so each aggregate is a dot product per draw.
function bootstrapMultiplicities(startCount, draws, seed) {
  const random = seededRandom(seed);
  const multiplicities = new Float64Array(draws * startCount);
  for (let draw = 0; draw < draws; draw += 1) {
    for (let pick = 0; pick < startCount; pick += 1) multiplicities[draw * startCount + Math.floor(random() * startCount)] += 1;
  }
  return multiplicities;
}

// A group's values on the period's start axis; NaN marks a start the group lacks.
function align(group, positionInPeriod, startCount) {
  const values = new Float64Array(startCount).fill(NaN);
  const counts = new Float64Array(startCount);
  for (let index = 0; index < group.starts.length; index += 1) {
    const position = positionInPeriod[group.starts[index]];
    if (position < 0) continue;
    values[position] = group.values[index];
    counts[position] = group.counts[index];
  }
  return { values, counts, isClass4: group.isClass4 };
}

// Point estimate over the present (non-NaN) starts.
function aggregate(aligned) {
  let total = 0;
  let weight = 0;
  aligned.values.forEach((value, position) => {
    if (Number.isNaN(value)) return;
    const count = aligned.isClass4 ? aligned.counts[position] : 1;
    total += aligned.isClass4 ? value * value * count : value;
    weight += count;
  });
  if (weight <= 0) return NaN;
  return aligned.isClass4 ? Math.sqrt(total / weight) : total / weight;
}

// The same aggregate for every bootstrap draw: sum(m v) / sum(m) over the present starts,
// or sqrt(sum(m v^2 n) / sum(m n)) for Class 4, with m the draw's multiplicities.
function bootstrapAggregates(aligned, multiplicities, draws) {
  const startCount = aligned.values.length;
  const numerator = new Float64Array(startCount);
  const weight = new Float64Array(startCount);
  aligned.values.forEach((value, position) => {
    if (Number.isNaN(value)) return;
    const count = aligned.isClass4 ? aligned.counts[position] : 1;
    numerator[position] = aligned.isClass4 ? value * value * count : value;
    weight[position] = count;
  });
  const results = new Float64Array(draws);
  // Every draw picks startCount starts, so a plain mean over a complete group needs no weights.
  if (!aligned.isClass4 && !aligned.values.some(Number.isNaN)) {
    for (let draw = 0; draw < draws; draw += 1) {
      const offset = draw * startCount;
      let total = 0;
      for (let position = 0; position < startCount; position += 1) total += multiplicities[offset + position] * numerator[position];
      results[draw] = total / startCount;
    }
    return results;
  }
  for (let draw = 0; draw < draws; draw += 1) {
    const offset = draw * startCount;
    let total = 0;
    let totalWeight = 0;
    for (let position = 0; position < startCount; position += 1) {
      const multiplicity = multiplicities[offset + position];
      total += multiplicity * numerator[position];
      totalWeight += multiplicity * weight[position];
    }
    results[draw] = totalWeight > 0 ? (aligned.isClass4 ? Math.sqrt(total / totalWeight) : total / totalWeight) : NaN;
  }
  return results;
}

// Rearrange values[from..to) so that position k holds the value it would hold once sorted,
// with nothing larger before it and nothing smaller after it (Hoare quickselect).
function selectInPlace(values, from, to, k) {
  let low = from;
  let high = to - 1;
  while (low < high) {
    const pivot = values[(low + high) >> 1];
    let left = low;
    let right = high;
    while (left <= right) {
      while (values[left] < pivot) left += 1;
      while (values[right] > pivot) right -= 1;
      if (left <= right) {
        const swap = values[left];
        values[left] = values[right];
        values[right] = swap;
        left += 1;
        right -= 1;
      }
    }
    if (k <= right) high = right;
    else if (k >= left) low = left;
    else return;
  }
}

function minimum(values, from, to) {
  let smallest = Infinity;
  for (let index = from; index < to; index += 1) if (values[index] < smallest) smallest = values[index];
  return smallest;
}

// numpy.percentile with its default linear interpolation, over the finite draws. Only two
// order statistics are read, so they are selected rather than sorted for.
function percentileInterval(draws) {
  const finite = new Float64Array(draws.length);
  let count = 0;
  for (const value of draws) if (Number.isFinite(value)) finite[count++] = value;
  if (!count) return [NaN, NaN];
  const at = (percent) => {
    const position = (percent / 100) * (count - 1);
    const low = Math.floor(position);
    selectInPlace(finite, 0, count, low);
    const next = low + 1 < count ? minimum(finite, low + 1, count) : finite[low];
    return finite[low] + (next - finite[low]) * (position - low);
  };
  return [at(((1 - CONFIDENCE) / 2) * 100), at(((1 + CONFIDENCE) / 2) * 100)];
}

function masked(aligned, keep) {
  const values = aligned.values.map((value, position) => (keep[position] ? value : NaN));
  return { values, counts: aligned.counts, isClass4: aligned.isClass4 };
}

function presentMask(first, second) {
  return Array.from(first.values, (value, position) => !Number.isNaN(value) && !Number.isNaN(second.values[position]));
}

/**
 * Aggregate every published summary row over the starts in `startIndices`.
 *
 * Returns rows shaped like the summary's, in its order, with mean, n_starts, n (Class 4),
 * the skill against the row's own skill_baseline, n_starts_paired and the 95% bootstrap
 * intervals of both.
 */
export function aggregatePeriod(perStart, summaryRows, startIndices, { draws = BOOTSTRAP_DRAWS, seed = BOOTSTRAP_SEED } = {}) {
  const startCount = startIndices.length;
  const positionInPeriod = new Int32Array(perStart.starts.length).fill(-1);
  startIndices.forEach((startIndex, position) => {
    positionInPeriod[startIndex] = position;
  });
  const multiplicities = bootstrapMultiplicities(startCount, draws, seed);
  const alignedCache = new Map();
  const alignedFor = (key) => {
    if (!alignedCache.has(key)) {
      const group = perStart.groups.get(key);
      alignedCache.set(key, group ? align(group, positionInPeriod, startCount) : null);
    }
    return alignedCache.get(key);
  };
  // A baseline serves every challenger of its key, so its draws are computed once.
  const drawsOf = (aligned) => (aligned.draws ??= bootstrapAggregates(aligned, multiplicities, draws));

  const rows = [];
  for (const summaryRow of summaryRows) {
    const model = alignedFor(identityKey(summaryRow));
    if (!model) continue;
    const present = model.values.filter((value) => !Number.isNaN(value)).length;
    if (!present) continue;
    const row = { ...summaryRow, mean: aggregate(model), n_starts: present };
    [row.ci_low, row.ci_high] = percentileInterval(drawsOf(model));
    if (model.isClass4) row.n = model.counts.reduce((sum, count, position) => sum + (Number.isNaN(model.values[position]) ? 0 : count), 0);

    const baseline = summaryRow.skill_baseline ? alignedFor(identityKey(summaryRow, summaryRow.skill_baseline)) : null;
    if (baseline) {
      const common = presentMask(model, baseline);
      const pairedCount = common.filter(Boolean).length;
      const allPresent = (aligned) => aligned.values.every((value, position) => !common[position] === Number.isNaN(value));
      const pairedModel = allPresent(model) ? model : masked(model, common);
      const pairedBaseline = allPresent(baseline) ? baseline : masked(baseline, common);
      row[SKILL_FIELD] = 1 - aggregate(pairedModel) / aggregate(pairedBaseline);
      const modelDraws = drawsOf(pairedModel);
      const baselineDraws = drawsOf(pairedBaseline);
      [row.skill_ci_low, row.skill_ci_high] = percentileInterval(modelDraws.map((value, draw) => 1 - value / baselineDraws[draw]));
      row.n_starts_paired = pairedCount;
    } else {
      row[SKILL_FIELD] = row.skill_ci_low = row.skill_ci_high = row.n_starts_paired = null;
    }
    rows.push(row);
  }
  return rows;
}
