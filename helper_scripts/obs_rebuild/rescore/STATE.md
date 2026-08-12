<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Obs-store rebuild rescore: ladder state (COMPLETE)

Working dir: /scratch/jseillade/obs-rebuild/rescore
Challenger: **glonet** (switched from glo12 mid-task on Julien's instruction; no glo12
scoring run was ever submitted, so no glo12 artifact exists). Staged glonet has 52 start
dates and **10** lead days, quarter degree, 21 depths. Leads 1-10 scored, 1/5/9 reported.
Regions: global and ibi. Fast class-4 only, no lagrangian, no gridded metrics.

## Code and environment

- Scoring checkout /scratch/jseillade/oceanbench-src-align (0917d23, #305 on top of 0.4.0)
- Venv /scratch/jseillade/oceanbench-venv-align
- Data-work venv (s3fs, zarr 2) /scratch/jseillade/obs-rebuild/venv
- Challenger stage /scratch/jseillade/oceanbench-stage-0.2.1, OCEANBENCH_STAGE=challenger,observations
  (observation staging is disabled inside score_rung.py so no rung reads another rung's cache;
  the observations stage key is left on only so the GLO12 MDT is read from the warm stage)

## Data staged locally, nothing read remotely during scoring

| path | content |
| --- | --- |
| store-legacy | legacy public observations2024, 370 days, copied over anonymous HTTPS |
| store-v2 | observations2024-v2 from CloudFerro, 370 days, 77330 objects, 92 MB |
| views (first pass, superseded) | L1 L2 L2b L3, no longitude correction |
| views2 (used for the final ladder) | L1 L1q L2 L2b L3, longitude normalised to [-180,180) |

CloudFerro read creds went to a mode-600 file, were used by the fetch, and were deleted.

## Rung definitions as finally scored

| rung | obs source | content |
| --- | --- | --- |
| L0 | store-legacy | published basis |
| L1 | views2/L1q | legacy equivalent: currents uo_raw/vo_raw on every row incl. undrogued; temperature and salinity temp_raw/psal_raw masked only by temp_qc/psal_qc == 1; sla as written |
| L2 | views2/L2 | L1 currents restricted to drogued == 1 |
| L2b | views2/L2b | raw-basis currents on the full-policy row set, added so L3 minus L2b is the pure inertial filter |
| L3 | views2/L3 | full default policy, legacy columns as written |
| L1raw | views2/L1 | diagnostic only: temperature and salinity straight from temp_raw/psal_raw with no measurement QC |

## Two corrections made mid-run, both worth reporting

1. **SLA longitude convention.** The v2 store writes obs_type 4 (SLA) longitude on
   0..360 while every in-situ stream is on -180..180 and legacy wrote SLA on -180..180.
   Scored as built, every SLA row east of 180 falls off the model grid and returns a NaN
   model value: global SLA count dropped from 15.71 M to 6.77 M at lead 1 and RMSD rose
   about 10 percent; IBI SLA collapsed from 181064 to 17893 points and RMSD rose 70
   percent. views2 normalises longitude and the defect disappears. This is a store bug to
   fix at the source, not a scoring artefact.
2. **Raw salinity is not a legacy equivalent.** psal_raw holds unmasked fill values up to
   234 psu, so an L1 built on it scores 4.66 psu RMSD. Legacy applied measurement QC flag 1
   and the rebuild left that unchanged, so the legacy-equivalent L1 masks by
   temp_qc/psal_qc == 1. The uncorrected variant is retained as rung L1raw.

## Known limitation, SLA

The store does not retain the filtered SLA value for rows that fail the policy, only
sla_unfiltered which is a different DUACS variable. So the L1 sla column equals L3 and no
QC or bounds attribution is possible for SLA from inside the store. On 2024-01-01 the
excluded SLA rows are 4852 of 269852 (1.8 percent), all day_misaligned, zero out of bounds.

## Runs

| job | what | state |
| --- | --- | --- |
| 34346 | first submit on partition small | cancelled, QOSMaxJobsPU = 4 there |
| 34351 | array 0-129%28 monoproc, all rungs on the first-pass views | complete, 130/130 exit 0 |
| 34482 | array 0-77%28, L1 L3 L1raw on views2 | complete, 78/78 exit 0 |

monoproc allows 32 jobs per user but only 1 cpu per job, so tasks are 1 cpu 48 GB.
Each task is one rung x region x chunk of 4 start dates, about 5 min.

## Artifacts

- results/glonet/*.csv, 156 files, per-chunk sumsq and count per variable, depth bin, lead
- results/glonet-uncorrected-lon/, the first pass kept as evidence of the longitude defect
- tables/glonet/ladder.md, ladder.csv, rmsd_pooled.csv, rmsd_per_depth_bin.csv, coverage.csv
- logs/task-34351_*.out, logs/task-34482_*.out, logs/fetch_v2.log, logs/fetch_legacy.log, logs/views.log

## Headline result

L1 reproduces L0 within noise everywhere, so the rebuild is validated:
currents bit identical, temperature +0.05 to +0.13 percent, salinity +0.04 to +0.09
percent, global SLA +0.08 to -0.08 percent, IBI SLA -0.22 to +0.04 percent.

The score change is dominated by the inertial filter on currents: L3 minus L2b is
-28 to -33 percent global and -33 to -40 percent IBI. The drogue drop is small,
-0.1 to -0.7 percent global. Temperature, salinity and SLA move by less than 0.6 percent.

## Resume

Rerun `sbatch --export=ALL,CHALLENGER=glonet --array=0-<n>%28 ladder.sbatch` (tasks skip
themselves when their CSV exists), then `aggregate_ladder.py glonet`. To run another
challenger, set CHALLENGER and the results land in results/<challenger>/.
