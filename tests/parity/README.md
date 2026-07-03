<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Parity golden dataset

Golden copy of every score published by OceanBench v1 for the 2024 benchmark:
the score tables of all published evaluation-report notebooks (report version
**0.2.1**), parsed into long-format records. 9,810 rows — 10 challengers ×
≤2 regions (global, ibi) × 9 metric keys.

## Files

- `golden_scores.parquet` — the golden records. One row per (challenger,
  region, metric_key, variable, depth_label, lead_day); `value` is null where
  the published table shows NaN.
- `golden_metadata.json` — provenance: source version 0.2.1, retrieval
  timestamp (2026-07-03), the reports root URL, and sha256 + byte size of
  every source notebook.
- `extract_golden.py` — regenerates the golden from the published notebooks
  (parsing logic adapted from `website/helpers/notebook_score_parser.py`).
  The notebooks themselves are not vendored here; they remain on S3 under the
  reports root recorded in the metadata, and their sha256s pin exactly what
  was parsed.

## Phase 1 gate

The next-release score runner must reproduce these values within numerical tolerance
(see `docs/contracts.md` §10, Phase 1 gate) before any published number is
replaced.

## Caveat: #298 provenance (RESOLVED)

The published 0.2.1 reports predate the current main-tip scoring code. Julien
confirmed (authoritative) that the golden was generated **before #298**:
gridded RMSD is **unweighted** and the Lagrangian metric uses the old seeding.
Class-4 observation metrics were **not** affected by #298 (known-identical).

`provenance_check.py` confirms this empirically. It recomputes surface
sea-surface-height RMSD for `glonet_1_degree` vs the 1-degree GLORYS reference
over the full 52 weekly starts of 2024, both with and without cos-lat area
weighting, and compares to the golden
(`rmsd_variables_glorys / sea_surface_height_above_geoid / surface`):

| variant | max abs diff vs golden |
|---|---|
| **unweighted** | **4.2e-07** (match) |
| area-weighted (#298) | 1.4e-03 (no match) |

Run: `python tests/parity/provenance_check.py --starts 52` (needs network).

### Consequence for the parity gate

- `class4_rmsd` rows compare **directly** against `golden_scores.parquet`.
- Gridded RMSD compares against the golden only in **unweighted** mode
  (test-only toggle; the production default stays area-weighted per next-release science).
- Lagrangian is excluded from golden comparison (legacy pre-#298 seeding);
  it is covered by the internal mean-equivalence check instead.
- `golden_scores_main_1degree.parquet` is the **go-forward** golden: the runner's
  per-start output in production (area-weighted, main-tip) mode on
  `glonet_1_degree`/`global`, which future changes must reproduce. This pass
  covers surface SSH + geostrophic currents vs both references (3,120 rows, 52
  start dates); see the scope note below.

## Phase 1 local parity run (glonet_1_degree / global / 2024)

Ran the runner end-to-end against public data (glonet forecasts on CloudFerro,
1-degree GLORYS/GLO12 references on EDITO MinIO). Three like-for-like checks:

**(a) Mean-equivalence (internal).** The mean over start dates of the per-start
records equals the aggregate from the untouched metric function
(`rmsd_of_variables_compared_to_glorys_reanalysis`), surface SSH:
`matched=10, max|diff| = 1.4e-17` — exact, as expected (the per-start path and
the aggregate share `_root_mean_squared_error_per_start`).

**(b) Unweighted runner vs published golden.** Gridded RMSD computed with
cos-lat weighting disabled (test-only), compared to `golden_scores.parquet`:

| golden metric key | matched | max abs diff | max rel diff |
|---|---|---|---|
| `rmsd_variables_glorys` (SSH surface) | 10 | 4.2e-07 | 5.3e-06 |
| `rmsd_variables_glo12` (SSH surface) | 10 | 4.8e-07 | 7.8e-06 |
| `rmsd_geostrophic_glorys` | 20 | 5.0e-07 | 7.4e-06 |
| `rmsd_geostrophic_glo12` | 20 | 4.6e-07 | 1.3e-05 |

Gate **PASSES** at `atol=1e-4` (and `1e-3`) for every computed key. This
confirms the golden is pre-#298 (unweighted) and that the runner reproduces it.

**(c) Go-forward golden.** Production (area-weighted) output written to
`golden_scores_main_1degree.parquet`.

### Scope of this run (honest flags)

This pass computed **surface SSH + geostrophic currents** vs both references
(weighted + unweighted). The following are **wired and unit-tested** in
`oceanbench/runner/` but were **not** run here — the 1/4-degree -> 1-degree
download of every variable at every depth over the local network exceeds the
compute budget (a single full `rmsd_variables/glorys` pass ran > 9 min without
finishing):

- deep T/S/U/V rows of `rmsd_variables_*` (the 240 golden rows shown as
  `gold_only` in the table above — the SSH rows that were computed all match),
- `rmsd_mld_*` (mixed layer depth — full-depth density download),
- `class4_rmsd` (observation match-ups — now emitted **per forecast start**:
  each row is the RMSD over that start's observations per variable × depth_bin ×
  lead_day with `n` = that observation count. The published pooled-over-obs value
  is recovered **exactly** by the n-weighted recombination
  `sqrt(sum(value² · n) / sum(n))` — proven in `tests/test_classiv_per_start.py`
  and applied by `oceanbench.runner.parity.aggregate_runner_scores` for the
  `class4_rmsd` metric),
- `lagrangian_deviation_km` (Parcels advection — hours for 52 global starts;
  excluded from golden comparison anyway per the pre-#298 seeding note above).

A full-scope run reproduces every key with the committed API:
`oceanbench.runner.run.run_challenger_scores("glonet_1_degree", "global", 2024)`
(area-weighted, all metrics) and the parity harness in
`oceanbench.runner.parity`.
