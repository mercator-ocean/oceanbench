<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Store rewrite to 2024-v2.1.0 and promotion-grade rescore

Working dir: `/scratch/jseillade/obs-rebuild/rescore/full-rescore-v21`
Date: 2026-08-06

## Adopted policy

Default currents scoring columns become

```
uo = EWCT_FILTR - uo_ws   where uo_ws is finite, EWCT_FILTR otherwise
vo = NSCT_FILTR - vo_ws   where vo_ws is finite, NSCT_FILTR otherwise
blank (NaN) where current_test is 11 (SAW 011) or 211
```

Codes 311, 312, 313, 212, 213 stay. QC flag-1-only, SLA, temperature and
salinity policies are unchanged. The subtraction sign is the LWS convention
proven earlier the same day: subtracting improves scores, adding degrades them.

## Part 1, store rewrite (COMPLETE, all gates pass)

Target `s3://oceanbench-bucket/dev/observations2024-v2`, plus the local mirror
`/scratch/jseillade/obs-rebuild/store-v2`. The legacy production
`observations2024` bucket was never touched.

Script `rewrite_v21.py`, modelled line for line on the v2.0.1 `patch_lon.py`
playbook: patch the mirror, rewrite `.zattrs` and the `.zattrs` entry inside
`.zmetadata`, append a manifest `patches` entry carrying the builder sha and the
patch-script sha, upload only the changed objects. Run as a 13-task month array
(`rewrite.sbatch`, job 34974), one state file per month under `state/`.

Unlike the longitude patch this operation is **not idempotent**, so every day is
guarded three ways before any write: the recorded `obs_basis_version` must be
`2024-v2.0.1`, the manifest must not already carry the patch entry, and the
per-month state file must not record the day as done.

### Refusal gates inside the rewrite (none fired on any of the 370 days)

| gate | meaning |
| --- | --- |
| `REFUSED_NON_CURRENT_ROW_CHANGED` | a row that is not `obs_type == 3` moved |
| `REFUSED_BLANK_COUNT_MISMATCH` | newly blanked rows are not exactly the finite 11/211 rows |
| `REFUSED_CORRECTION_TOO_LARGE` | a wind slippage above 5 m/s |
| `REFUSED_UNEXPECTED_ROW_CHANGED` | a row changed that had neither finite slippage nor a dropped code |

### Measured over all 370 days

| quantity | value |
| --- | --- |
| days rewritten | 370, all `ok` |
| rows in store | 230 853 512 (unchanged) |
| currents rows | 10 132 425 |
| currents finite before | 8 636 806 (85.24%) |
| currents finite after | 8 605 782 (84.93%) |
| rows blanked by the 211 drop | 31 024 |
| rows shifted by the slippage | 4 942 429 |
| rows changed in total | 4 973 453 |
| rows keeping FILTR unchanged (slippage missing) | 5.82% of the finite set |
| max absolute slippage applied | 0.6055 m/s |

The 31 024 blanked rows match the independent 370-day census of `current_test`
exactly: code 211 has 35 067 rows of which 31 024 were finite in the default
columns. Code 11 contributed nothing because the builder had already blanked it.

Full code census over 370 days, currents rows:

| code | rows | of which finite before the rewrite |
| --- | --- | --- |
| 11 | 1 400 457 | 0 |
| 211 | 35 067 | 31 024 |
| 212 | 207 599 | 186 955 |
| 213 | 257 048 | 232 110 |
| 311 | 628 275 | 625 546 |
| 312 | 3 110 721 | 3 092 839 |
| 313 | 4 493 258 | 4 468 332 |

### Verification, `verify_v21.py`, days 20240101 20240712 20241002 20241003 20241005

All five days `PASS`, `ALL_PASS` overall. Gates:

- **a1** new default equals the pre-rewrite `views2/LWS` view with the 11/211
  blanking applied, bit for bit. LWS was materialised from the pre-rewrite store
  by a different script, so this is an independent recompute.
- **a2** new default equals `EWCT_FILTR`/`NSCT_FILTR` read straight out of the
  raw netCDF archive, joined on `obs_id` the way `materialize_strata.py` does,
  minus the stored slippage. 22 852, 22 975, 22 199, 22 008 and 21 991 rows
  compared, exact match on every one.
- **b** row counts unchanged against the pre-rewrite record.
- **c** all fifteen other float columns byte-identical to their pre-rewrite
  sha256, including `uo_raw`, `vo_raw`, `uo_ws`, `vo_ws`, temperature, salinity
  and SLA.
- **d** kept fraction 0.797 to 0.857 per day, 0.84933 over the year.
- **e** the remote objects equal the local mirror by sha256 on both velocity
  columns, same row count.

### Store version, swept over every day

370 of 370 remote days and 370 of 370 mirror days report
`obs_basis_version = 2024-v2.1.0`, `policy.policy_version = v2.1.0`,
`policy.drop_undrogued_current_test = [11, 211]`, and exactly one
`currents-wind-slippage-and-211-drop` manifest patch entry.

The stored policy dict also gained `current_wind_slippage_removed: true` and
`current_wind_slippage_source`, and `row_counts_after_policy.currents` was
updated to the new finite count. The previous policy dict and counts are
preserved inside the manifest patch entry.

## Part 2, rescore

Scoring view `views2/V21`, the nine legacy columns copied verbatim out of the
rewritten store by `materialize_v21.py` (job 34991, 12 shards, 370 days, no
arithmetic, longitude asserted in range). The published store carries about
forty columns including wide string columns, which the class-4 scorer does not
need.

Rungs:

| rung | obs root | meaning |
| --- | --- | --- |
| OLD | `/scratch/jseillade/obs-rebuild/store-legacy` | legacy published observations2024 |
| NEW | `/scratch/jseillade/obs-rebuild/views2/V21` | 2024-v2.1.0 |

Both sides run the same `score_rung.py`, the same chunking and the same warm
challenger stage, so every delta is a basis change only. 52 start dates in 13
chunks of 4, regions `global` and `ibi`, variables `thetao,so,sla,uo,vo`.

Jobs: 35003 (smoke, tasks 0, 26, 156, 260), 35010 (the fleet, 360 tasks at
concurrency 24 on `monoproc`) and 35117 (the glonet-only resubmission). 364 tasks
were generated in `tasks.txt`.

### Fleet rescore DEFERRED by Julien, 2026-08-06

Julien stopped the fleet rescore part way: too much else was running on lir and he
will do the proper all-challenger scoring later. Only glonet was wanted, as a small
colleague-ready deliverable.

`scancel 35003 35010` was issued. 35003 ended 3 tasks completed and 1 cancelled
(the wenhai task, index 156, was mid-flight). 35010 ended 63 completed, 22
cancelled while running, and the rest cancelled before ever starting.
`sbatch --array=0-51%13 rescore.sbatch` was then resubmitted as job 35117, which
covers exactly the glonet block of `tasks.txt` (task indices 0 to 51). All 156
steps `COMPLETED 0:0`: every task whose CSV already existed skipped itself in
under a second, and the three genuinely missing ones, `NEW_global` chunks 1, 2
and 4, ran.

**glonet is complete: 52 of 52 CSVs, 13 chunks on each of NEW/OLD x global/ibi.**

Partial CSVs from the cancelled fleet run are kept on disk because they are tiny
and will be reusable when the full rescore is redone: `results/glo12/` 13 files,
`results/glonet2_e230_icefix_r10/` 1 file, `results/wenhai/` empty. They are not
in the published table, which was restricted to glonet.

Timings measured before the cancellation, useful for sizing the deferred run:
glonet NEW global 4:41, glonet OLD global 5:00, glonet2_e230 NEW global 6:27
(MaxRSS 6.9 GB, so 96 GB per task is generous). wenhai, twelfth degree, needed
443 s for temperature alone, so order 20 to 25 minutes per global task.

### Challenger coverage

The brief asked for twelve challengers, and the fleet run was later deferred
anyway. For the record, only seven can be scored with a challenger definition
that actually exists in a 0.4.0-lineage checkout on lir:

| challenger | checkout | venv |
| --- | --- | --- |
| glonet, glo12, langya, wenhai, xihe | `oceanbench-src-align` (0917d23, #305) | `oceanbench-venv-align` |
| glonet2_e230_icefix_r10, glonet2_e230_icefix_r10_forcingfix | `oceanbench-src-glonet2-040` (c681f1d) | `oceanbench-venv-glonet2-040` |

`climatology`, `persistence`, `glonet2_e228_icefix_r7`, `glonet2_e231_icefix_r10`,
`glonet2_e232_scalesep_r7` and `glonet2_e233_mixed135710` are staged under
`oceanbench-stage-0.2.1` but have no definition in any 0.4.0-lineage checkout.
`climatology` and `persistence` exist only in `oceanbench-rebuild-src`, which is
the 0.5.0 rebuild tree with a different class-4 path. No challenger definition
was invented for them: scoring a name the code cannot resolve, or inventing a
source URL, would risk silently scoring the wrong data.

The split across two checkouts is safe for this table because #305 changed the
gridded metrics only; class-4 and lagrangian outputs were verified byte-identical
across that commit.

## Aggregation

`compare_v21.py` takes an optional list of challenger names and restricts itself
to them. The published glonet table was produced with

```
/scratch/jseillade/obs-rebuild/venv/bin/python \
  /scratch/jseillade/obs-rebuild/rescore/full-rescore-v21/compare_v21.py glonet
```

Writes under `tables/`: `RESULTS.md`, `compare_pooled.csv`,
`compare_per_lead.csv`, `rmsd_per_depth_bin.csv`, `coverage.csv`. Check
`coverage.csv` first: every challenger must show 13 chunks for each of the four
rung and region combinations before the table is read as final. glonet does.

Resume the deferred fleet run with `sbatch --array=0-363%24 rescore.sbatch`.
Tasks skip themselves when their CSV already exists, so the glonet block and the
partial glo12 and glonet2 CSVs cost nothing to re-submit.

## The glonet result

RMSD pooled over depth bins and leads. Percent is OLD to NEW, negative is better.
The currents delta is the whole basis change in one number against the legacy
published store: inertial FILTR basis, minus wind slippage, minus current_test
011 and 211.

| region | stream | OLD | NEW | delta | % | n OLD | n NEW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | temperature | 0.87216 | 0.87319 | +0.00103 | +0.12% | 63612236 | 63076748 |
| global | salinity | 0.14741 | 0.14669 | -0.00073 | -0.49% | 52649339 | 52189847 |
| global | sla | 0.05987 | 0.05993 | +0.00006 | +0.10% | 154335529 | 154335811 |
| global | uo | 0.21205 | 0.13268 | -0.07937 | -37.43% | 14047004 | 11974208 |
| global | vo | 0.19594 | 0.12849 | -0.06745 | -34.43% | 14047004 | 11974208 |
| ibi | temperature | 0.79768 | 0.79832 | +0.00064 | +0.08% | 1385612 | 1382590 |
| ibi | salinity | 0.18219 | 0.18239 | +0.00021 | +0.11% | 1181401 | 1178380 |
| ibi | sla | 0.05063 | 0.05061 | -0.00002 | -0.04% | 1791498 | 1817402 |
| ibi | uo | 0.20496 | 0.10305 | -0.10191 | -49.72% | 214906 | 183641 |
| ibi | vo | 0.17726 | 0.10229 | -0.07497 | -42.30% | 214906 | 183641 |

Per lead, currents:

| region | stream | L1 OLD | L1 NEW | L1 % | L5 OLD | L5 NEW | L5 % | L9 OLD | L9 NEW | L9 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | uo | 0.20423 | 0.12221 | -40.16% | 0.21018 | 0.13153 | -37.42% | 0.21700 | 0.13955 | -35.69% |
| global | vo | 0.18710 | 0.11853 | -36.65% | 0.19609 | 0.12867 | -34.38% | 0.20110 | 0.13508 | -32.83% |
| ibi | uo | 0.20177 | 0.09869 | -51.09% | 0.19436 | 0.10124 | -47.91% | 0.21140 | 0.10641 | -49.66% |
| ibi | vo | 0.17414 | 0.09666 | -44.49% | 0.17749 | 0.10183 | -42.63% | 0.18075 | 0.10845 | -40.00% |

Temperature, salinity and SLA move by at most half a percent, which is the
expected row-set effect of dedup, day alignment and position and time QC, not a
value change. The check that shows this: in IBI, at every lead where the old and
new row counts are equal (temperature and salinity leads 1, 4, 5 and 8) the RMSD
is identical to six significant figures. A difference appears only at leads where
a profile was dropped.

Currents counts fall from 14.05 M to 11.97 M globally, 14.8 percent, which is the
011 and 211 drop plus the rows the FILTR basis does not carry.

## Files

| path | content |
| --- | --- |
| `census_codes.py`, `census_codes.json` | 370-day `current_test` and slippage census |
| `rewrite_v21.py`, `rewrite.sbatch` | the in-place store rewrite |
| `state/rewrite-*.json` | per-day rewrite record, digests before and after |
| `verify_v21.py` | the five verification gates |
| `materialize_v21.py`, `view.sbatch` | the lean scoring view |
| `tasks.txt`, `rescore.sbatch` | the 364 scoring tasks |
| `compare_v21.py` | old versus new comparison tables |
| `logs/` | every job log |
