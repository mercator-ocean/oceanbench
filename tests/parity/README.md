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
  output in production (area-weighted, main-tip) mode on
  `glonet_1_degree`/`global`, which future changes must reproduce.
