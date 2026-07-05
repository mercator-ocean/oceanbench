<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Rebuild wave 1 parity verdict

Like-for-like comparison against `tests/parity/golden_scores.parquet`: Class-4 metrics use production Class-4 rows; gridded RMSD uses `runs-unweighted/`. Lagrangian rows are excluded because the published golden predates the seeding change.

Gate rule: non-SLA rows must be <= 5e-7; SLA rows must be a consistent known #295 MDT-provenance offset around 5e-2.

| challenger | metric family | count | max abs diff | mean abs diff | spread | verdict |
|---|---:|---:|---:|---:|---:|---|
| glonet | class4_non_sla | 110 | 4.82434e-07 | 2.38048e-07 | 1.44186e-07 | PASS |
| glonet | class4_sla | 10 | 0.0569819 | 0.0508832 | 0.00510221 | PASS |
| glonet | rmsd_geostrophic_glo12 | 20 | 4.99222e-07 | 2.46024e-07 | 1.52615e-07 | PASS |
| glonet | rmsd_geostrophic_glorys | 20 | 4.91407e-07 | 2.75965e-07 | 1.42908e-07 | PASS |
| glonet | rmsd_mld_glo12 | 10 | 5.14505e-06 | 2.35739e-06 | 1.41633e-06 | FAIL |
| glonet | rmsd_mld_glorys | 10 | 5.43094e-06 | 1.68689e-06 | 1.68787e-06 | FAIL |
| glonet | rmsd_variables_glo12 | 250 | 4.98786e-07 | 2.66165e-07 | 1.41117e-07 | PASS |
| glonet | rmsd_variables_glorys | 250 | 4.98675e-07 | 2.61304e-07 | 1.40476e-07 | PASS |
| glonet | OVERALL_GATE | 680 | 0.0569819 | 0.0508832 | 0.00510221 | FAIL |
| xihe | class4_non_sla | 110 | 4.94144e-07 | 2.43383e-07 | 1.46289e-07 | PASS |
| xihe | class4_sla | 10 | 0.05371 | 0.0508113 | 0.00217918 | PASS |
| xihe | rmsd_geostrophic_glo12 | 20 | 5.26009e-07 | 2.50558e-07 | 1.46845e-07 | FAIL |
| xihe | rmsd_geostrophic_glorys | 20 | 4.20896e-07 | 1.96271e-07 | 1.18191e-07 | PASS |
| xihe | rmsd_mld_glo12 | 10 | 4.41656e-06 | 1.96709e-06 | 1.24684e-06 | FAIL |
| xihe | rmsd_mld_glorys | 10 | 1.01715e-05 | 3.60332e-06 | 3.04141e-06 | FAIL |
| xihe | rmsd_variables_glo12 | 250 | 5.08339e-07 | 2.62226e-07 | 1.39585e-07 | FAIL |
| xihe | rmsd_variables_glorys | 250 | 4.99781e-07 | 2.50084e-07 | 1.45595e-07 | PASS |
| xihe | OVERALL_GATE | 680 | 0.05371 | 0.0508113 | 0.00217918 | FAIL |

Anomaly: mixed-layer-depth gridded rows compare at 4e-6 to 1e-5, above the strict 5e-7 non-SLA threshold. The SLA offset is consistent with the known #295 MDT provenance change.
