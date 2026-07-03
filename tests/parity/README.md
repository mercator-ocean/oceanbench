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

- `class4_rmsd` is emitted **per forecast start**: each row is the RMSD over
  that start's observations per variable × depth_bin × lead_day with `n` = that
  observation count. The published pooled-over-obs value is recovered
  **exactly** by the n-weighted recombination `sqrt(sum(value² · n) / sum(n))`
  — proven in `tests/test_classiv_per_start.py` and applied by
  `oceanbench.runner.parity.aggregate_runner_scores` for the `class4_rmsd`
  metric — so class4 rows compare directly against `golden_scores.parquet`
  after recombination.
- Gridded RMSD compares against the golden only in **unweighted** mode
  (test-only toggle; the production default stays area-weighted per next-release science).
- Lagrangian is excluded from golden comparison (legacy pre-#298 seeding);
  it is covered by the internal mean-equivalence check instead.
- `golden_scores_main_1degree.parquet` is the **go-forward** golden: the runner's
  per-start output in production (area-weighted, main-tip) mode on
  `glonet_1_degree`/`global` — full scope, all metrics except lagrangian (see
  the full-run section below), which future changes must reproduce.

## Phase 1 full local parity run (glonet_1_degree / global / 2024)

Ran the runner end-to-end for **every metric except lagrangian** against public
data (glonet forecasts on CloudFerro, 1-degree GLORYS/GLO12 references and
observations on EDITO MinIO), with local staging enabled
(`OCEANBENCH_STAGE=all`, `OCEANBENCH_REMOTE_RETRIES=5`) so all inputs were
materialised once (~100 GB: 52 challenger weeks at 1/4 degree, 52 reference
weeks per reference at 1 degree, the 2024 observation match-up store, the
MDTs). Wall time ~2 h 15 min end-to-end, dominated by the challenger download
(~70 min) and the Class-4 velocity 15 m interpolation; gridded compute from
the warm stage takes minutes. Internal mean-equivalence (per-start mean ==
untouched aggregate function) holds at 1.4e-17 (shared
`_root_mean_squared_error_per_start` path).

**Like-for-like parity vs the published golden** (gridded in unweighted
test-only mode per the #298 note; class4 recombined per-start → pooled; both
references; all variables, depths and bins — 660 golden rows compared, none
missing on either side):

| golden metric key | matched | max abs diff | max rel diff | verdict |
|---|---|---|---|---|
| `rmsd_variables_glorys` (5 vars × 6 depths) | 250 | 5.0e-07 | 5.4e-06 | PASS |
| `rmsd_variables_glo12` (5 vars × 6 depths) | 250 | 5.0e-07 | 1.7e-05 | PASS |
| `rmsd_geostrophic_glorys` | 20 | 5.0e-07 | 7.4e-06 | PASS |
| `rmsd_geostrophic_glo12` | 20 | 4.6e-07 | 1.3e-05 | PASS |
| `rmsd_mld_glorys` | 10 | 7.5e-06 | 1.4e-07 | PASS |
| `rmsd_mld_glo12` | 10 | 5.7e-06 | 1.0e-07 | PASS |
| `rmsd_variables_observations` — T/S bins + U/V 15m | 110 | 4.9e-07 | — | PASS |
| `rmsd_variables_observations` — SLA surface | 10 | 5.5e-02 | — | #295 provenance (below) |
| `lagrangian_glorys` / `lagrangian_glo12` | excluded | — | — | pre-#298 seeding (below) |

All PASS differences are at the parse precision of the published HTML tables
(the golden carries ~6 significant digits). The gate passes at `atol=1e-4` for
every compared key except the 10 SLA rows, which are a **verified
code-evolution difference, not a runner bug**:

### Caveat: class4 SLA rows and #295 (RESOLVED, same pattern as #298)

The published 0.2.1 reports (2026-06-17) predate #295 "Use GLO12 MDT for global
SSH→SLA" (2026-06-30). The main-tip runner computes SLA against the GLO12
(001_024) MDT while the golden was produced with the GLORYS12 (001_030) MDT.
Recomputing the 10 SLA class4 rows with the pre-#295 MDT reproduces the golden
exactly:

| variant | max abs diff vs golden (10 SLA rows) |
|---|---|
| main-tip MDT (GLO12, #295) | 5.5e-02 (no match — expected) |
| **pre-#295 MDT (GLORYS12 001_030)** | **4.5e-07 (match)** |

(The main-tip SLA RMSD ~0.055 vs published ~0.110 also reflects that #295 was
an accuracy fix.) T/S/U/V class4 rows are MDT-independent and match at parse
precision, which also validates the per-start emission + n-weighted
recombination end-to-end on real data.

### Lagrangian (excluded, honest statement)

`lagrangian_deviation_km` was **not** computed in this run: Parcels advection
costs hours for 52 global starts and the golden values use the legacy pre-#298
seeding, so a golden comparison would be moot regardless. It remains covered by
the internal mean-equivalence tests; its runner wiring is unchanged.

### Go-forward golden

`golden_scores_main_1degree.parquet` is the runner's **production** output of
this run (area-weighted gridded RMSD + per-start class4 with `n`): 35,360
per-start rows — 29,120 gridded (2 references × {5 variables × depths, MLD,
2 geostrophic} × 10 leads × 52 starts) + 6,240 class4 (5 variables × bins ×
10 leads × 52 starts, `n` from 735 to 315,147 obs per start-cell). Future
changes must reproduce it. Reproduce with
`oceanbench.runner.run.run_challenger_scores("glonet_1_degree", "global", 2024, include_lagrangian=False)`
and compare via `oceanbench.runner.parity`.
