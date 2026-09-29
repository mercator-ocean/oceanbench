// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Parity check for the Scores page period aggregation (modules/scores-periods.js).
//
// Runs the browser aggregation over every start of the published scores.parquet and
// requires each mean and skill to equal the published scores-summary.json (the Python
// aggregate_scores output) to 1e-9 relative. Skill is 1 - model / baseline, so its rounding
// error is relative to the larger of |skill| and |1 - skill|: a model equal to its baseline
// has a skill of 0 +- 1e-16 whose own relative difference means nothing. Also prints the
// bootstrap cost.
//
//   node qa/period-parity.mjs [scores.parquet] [scores-summary.json]
//
// Arguments are local paths or URLs; the default is the published preview.

import { readFile } from "node:fs/promises";
import { aggregatePeriod, periodStartIndices, readPerStartScores } from "../modules/scores-periods.js";

const PREVIEW = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/dev/benchmark/rebuild-preview";
const [parquetSource = `${PREVIEW}/scores.parquet`, summarySource = `${PREVIEW}/viewer/data/scores-summary.json`] = process.argv.slice(2);
const TOLERANCE = 1e-9;
const FIELDS = ["mean", "skill_vs_persistence_1_degree"];

async function bytes(source) {
  if (/^https?:/.test(source)) {
    const response = await fetch(source);
    if (!response.ok) throw new Error(`${source} -> HTTP ${response.status}`);
    return response.arrayBuffer();
  }
  const buffer = await readFile(source);
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

const summary = JSON.parse(new TextDecoder().decode(await bytes(summarySource))).filter((row) => Number.isFinite(row.mean));
let started = performance.now();
const perStart = await readPerStartScores(await bytes(parquetSource));
console.log(`read ${perStart.groups.size} groups, ${perStart.starts.length} starts in ${(performance.now() - started).toFixed(0)} ms`);

const allStarts = perStart.starts.map((_, position) => position);
started = performance.now();
const rows = aggregatePeriod(perStart, summary, allStarts);
console.log(`means, skill and intervals for ${rows.length} rows in ${(performance.now() - started).toFixed(0)} ms`);

let compared = 0;
let worst = 0;
let worstRaw = 0;
const mismatches = [];
if (rows.length !== summary.length) mismatches.push(`row count ${rows.length} != ${summary.length}`);
rows.forEach((row, index) => {
  const published = summary[index];
  for (const field of FIELDS) {
    const expected = published[field] ?? null;
    const actual = Number.isFinite(row[field]) ? row[field] : null;
    if (expected === null && actual === null) continue;
    compared += 1;
    const absolute = expected === null || actual === null ? Infinity : Math.abs(actual - expected);
    const scale = field === "mean" ? Math.abs(expected) : Math.max(Math.abs(expected), Math.abs(1 - expected));
    if (expected !== 0) worstRaw = Math.max(worstRaw, absolute / Math.abs(expected));
    const difference = absolute / scale;
    worst = Math.max(worst, difference);
    if (!(difference <= TOLERANCE)) mismatches.push(`${field} ${JSON.stringify(published)} got ${actual}`);
  }
  if (row.n_starts !== published.n_starts) mismatches.push(`n_starts ${JSON.stringify(published)} got ${row.n_starts}`);
});
console.log(
  `compared ${compared} values over ${rows.length} rows: max relative difference ${worst.toExponential(2)}, ` +
    `mismatches ${mismatches.length} (plain relative difference, largest on near-zero skills: ${worstRaw.toExponential(2)})`,
);
for (const line of mismatches.slice(0, 10)) console.log(`  ${line}`);

for (const [label, from, to] of [["Jul-Sep", "2024-07-01", "2024-09-30"], ["whole year", "2024-01-01", "2024-12-31"]]) {
  const starts = periodStartIndices(perStart, from, to);
  started = performance.now();
  aggregatePeriod(perStart, summary, starts);
  console.log(`${label} (${starts.length} starts): ${summary.length} rows with bootstrap intervals in ${(performance.now() - started).toFixed(0)} ms`);
}
process.exit(mismatches.length ? 1 : 0);
