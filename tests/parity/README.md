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

The v2 score runner must reproduce these values within numerical tolerance
(see `docs/contracts.md` §10, Phase 1 gate) before any published number is
replaced.

## Caveat: #298 provenance

The published 0.2.1 reports may predate the current main-tip scoring code —
in particular the area-weighted RMSD reapply (#298). Verifying which code
produced the published reports, and therefore what the golden actually
encodes, is Phase 1's opening task. Until that check is done, treat the gate
as provisional.
