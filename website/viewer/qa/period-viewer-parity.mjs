// SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
//
// SPDX-License-Identifier: EUPL-1.2

// Parity check for the viewer's Period control (modules/period-view.js).
//
// Runs the viewer's period path over every start of the published scores.parquet, the
// whole year as a range, and requires it to reproduce what the rail draws for the whole
// year from the published artifacts:
// - the RMSE vs lead day curves: each Class 4 observation row of scores-summary.json, mean
//   to 1e-9 relative and n_starts exact (the band is a bootstrap with another generator,
//   so its bounds are printed, not compared);
// - the RMSE vs depth profiles: every cell of every rmsd-by-depth.json the insight index
//   lists, RMSE to 1e-6 relative, bias to 1e-6 of the cell's RMSE, n exact. The year
//   artifacts pool the raw match-ups and the period pools the per-start values of
//   scores.parquet; the two roundings differ by 1e-9 to a few 1e-7, far below what the
//   chart can show.
//
//   node qa/period-viewer-parity.mjs [scores.parquet] [viewer data root]
//
// Arguments are a local path or URL for the parquet and a URL or local directory for the
// viewer data root; the default is the published preview.

import { readFile } from "node:fs/promises";
import path from "node:path";
import { periodRange } from "../modules/period-range.js";
import { periodDepthArtifact, periodLeadRows, periodStartIndices, readPerStartScores } from "../modules/period-view.js";

const PREVIEW = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/dev/benchmark/rebuild-preview";
const [parquetSource = `${PREVIEW}/scores.parquet`, dataRoot = `${PREVIEW}/viewer`] = process.argv.slice(2);
const LEAD_TOLERANCE = 1e-9;
const DEPTH_TOLERANCE = 1e-6;

async function bytes(source) {
  if (/^https?:/.test(source)) {
    const response = await fetch(source);
    if (!response.ok) throw new Error(`${source} -> HTTP ${response.status}`);
    return response.arrayBuffer();
  }
  const buffer = await readFile(source);
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}

// Artifact paths in the index are relative to the viewer directory ("./data/...").
const dataSource = (relative) => (/^https?:/.test(dataRoot) ? `${dataRoot}/${relative.replace(/^\.\//, "")}` : path.join(dataRoot, relative));
const json = async (relative) => JSON.parse(new TextDecoder().decode(await bytes(dataSource(relative))));

const index = await json("./data/insights.json");
const summary = await json(index.scores_summary);
const perStart = await readPerStartScores(await bytes(parquetSource));
const year = summary[0].year;
// The whole year as a period range, through the same helpers the viewer calls.
const range = periodRange(`${year}-01-01_${year}-12-31`, year);
const starts = periodStartIndices(perStart, range.from, range.to);
console.log(`${starts.length} starts in ${range.label}, ${perStart.groups.size} per-start groups`);

const mismatches = [];
const relative = (actual, expected) => Math.abs(actual - expected) / Math.abs(expected);

// (a) Lead curves: one series per challenger, region, variable and depth bin, as the rail
// selects them (Class 4 RMSE against observations).
const series = new Map();
for (const row of summary) {
  if (row.metric !== "class4_rmsd" || row.reference !== "observations" || !Number.isFinite(row.mean)) continue;
  const key = [row.challenger, row.region, row.variable, row.depth].join("|");
  if (!series.has(key)) series.set(key, []);
  series.get(key).push(row);
}
let leadValues = 0;
let leadWorst = 0;
let bandWorst = 0;
const leadChecked = new Set();
for (const [key, rows] of series) {
  const periodRows = periodLeadRows(perStart, rows, starts);
  if (periodRows.length !== rows.length) mismatches.push(`lead ${key}: ${periodRows.length} rows != ${rows.length}`);
  periodRows.forEach((row, position) => {
    const published = rows[position];
    const difference = relative(row.mean, published.mean);
    leadWorst = Math.max(leadWorst, difference);
    leadValues += 1;
    if (!(difference <= LEAD_TOLERANCE)) mismatches.push(`lead ${key} lead ${published.lead_day}: mean ${row.mean} != ${published.mean}`);
    if (row.n_starts !== published.n_starts) mismatches.push(`lead ${key} lead ${published.lead_day}: n_starts ${row.n_starts} != ${published.n_starts}`);
    if (Number.isFinite(published.ci_low)) {
      bandWorst = Math.max(bandWorst, relative(row.ci_low, published.ci_low), relative(row.ci_high, published.ci_high));
    }
  });
  leadChecked.add(key.split("|").slice(0, 2).join("|"));
}
console.log(
  `lead curves: ${series.size} series, ${leadValues} values, max relative difference ${leadWorst.toExponential(2)} ` +
    `(band bounds, not compared: ${bandWorst.toExponential(2)})`,
);
for (const required of ["glo12|global", "glo12|ibi"]) {
  if (!leadChecked.has(required)) mismatches.push(`lead curves: no ${required} series`);
}

// (b) Depth profiles: every published rmsd-by-depth.json.
let depthFiles = 0;
let depthCells = 0;
let rmsdWorst = 0;
let biasWorst = 0;
const depthChecked = new Set();
for (const [slug, regions] of Object.entries(index.datasets)) {
  for (const [region, entry] of Object.entries(regions)) {
    if (!entry.rmsd_by_depth) continue;
    const published = await json(entry.rmsd_by_depth);
    const rebuilt = periodDepthArtifact(perStart, published, { challenger: published.challenger, year, region: published.region }, starts);
    depthFiles += 1;
    let fileRmsdWorst = 0;
    let fileBiasWorst = 0;
    depthChecked.add(`${published.challenger}|${published.region}`);
    for (const [variable, expected] of Object.entries(published.variables)) {
      const actual = rebuilt.variables[variable];
      expected.depth_bins.forEach((bin, row) => {
        expected.leads.forEach((lead, column) => {
          const where = `${slug}/${region} ${variable} ${bin} lead ${lead}`;
          const want = { rmsd: expected.rmsd[row][column], bias: expected.bias[row][column], n: expected.n[row][column] };
          const got = { rmsd: actual.rmsd[row][column], bias: actual.bias[row][column], n: actual.n[row][column] };
          if (want.rmsd == null || want.n == null || want.n === 0) {
            if (got.rmsd != null) mismatches.push(`depth ${where}: published empty, rebuilt ${got.rmsd}`);
            return;
          }
          depthCells += 1;
          const rmsdDifference = relative(got.rmsd, want.rmsd);
          const biasDifference = Math.abs(got.bias - want.bias) / want.rmsd;
          fileRmsdWorst = Math.max(fileRmsdWorst, rmsdDifference);
          fileBiasWorst = Math.max(fileBiasWorst, biasDifference);
          if (!(rmsdDifference <= DEPTH_TOLERANCE)) mismatches.push(`depth ${where}: rmsd ${got.rmsd} != ${want.rmsd}`);
          if (!(biasDifference <= DEPTH_TOLERANCE)) mismatches.push(`depth ${where}: bias ${got.bias} != ${want.bias}`);
          if (got.n !== want.n) mismatches.push(`depth ${where}: n ${got.n} != ${want.n}`);
        });
      });
    }
    console.log(`  ${slug}/${region}: RMSE ${fileRmsdWorst.toExponential(2)}, bias / RMSE ${fileBiasWorst.toExponential(2)}`);
    rmsdWorst = Math.max(rmsdWorst, fileRmsdWorst);
    biasWorst = Math.max(biasWorst, fileBiasWorst);
  }
}
console.log(
  `depth profiles: ${depthFiles} artifacts, ${depthCells} cells, max RMSE relative difference ${rmsdWorst.toExponential(2)}, ` +
    `max bias difference / RMSE ${biasWorst.toExponential(2)}, n exact unless listed`,
);
for (const required of ["glo12|global", "glo12|ibi"]) {
  if (!depthChecked.has(required)) mismatches.push(`depth profiles: no ${required} artifact`);
}

console.log(`mismatches ${mismatches.length}`);
for (const line of mismatches.slice(0, 20)) console.log(`  ${line}`);
process.exit(mismatches.length ? 1 : 0);
