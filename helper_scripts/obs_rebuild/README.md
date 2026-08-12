<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# observations2024-v2 builder scripts

These scripts built the `observations2024-v2` class-4 observation store: QC-1-only
on the FILTR basis, undrogued-drifter drop, and the SLA longitude 0..360 patch.
The final store is published at the CloudFerro `oceanbench` bucket under the
prefix `2024-v2.0.1`.

They were rescued from the lir HPC, where the data working directory was
`/scratch/jseillade/obs-rebuild`. Paths inside the scripts still refer to that
directory. Only code and docs were copied here; the zarr stores, raw archive,
materialized views, logs, and result CSVs stay on lir and CloudFerro.

The `rescore/` subtree holds the ladder rescore that validated the rebuild.

These files are kept as they were run, so they are not black-formatted and do not
pass the repository flake8 configuration. They are an archived provenance record,
not maintained package code. Only two changes were made on import: the SPDX header
required by `reuse lint`, and this section.

The original working notes from that directory follow.

## OceanBench observation store rebuild

`build_observations.py` rebuilds the class-4 observation store, one zarr per UTC day,
from the same Copernicus Marine sources as the colleague notebook
`creation_data_2025.ipynb`. It is a flagged archive: nothing is dropped silently,
rows that fail the default policy are kept with `qc_keep=0` and their flags.

## Credentials

Nothing is hardcoded. The script reads:

| variable | use |
| --- | --- |
| `COPERNICUSMARINE_SERVICE_USERNAME` | copernicusmarine login for the DUACS L3 SLA datasets |
| `COPERNICUSMARINE_SERVICE_PASSWORD` | same |
| `CF_KEY` (or `AWS_ACCESS_KEY_ID`) | CloudFerro write key for the target bucket |
| `CF_SECRET` (or `AWS_SECRET_ACCESS_KEY`) | same |
| `CF_ENDPOINT` (optional) | overrides `https://s3.waw3-1.cloudferro.com` |

The notebook had the Copernicus password and the CloudFerro key pair inline in
cell 0. Those were deliberately not copied here. They should be treated as
exposed and rotated.

The source bucket `mdl-native-08` is read anonymously, no credentials needed.

## Sources

| stream | product | dataset | file |
| --- | --- | --- | --- |
| Argo T/S | `INSITU_GLO_PHY_TSASSIM_DISCRETE_NRT_013_047` | `cmems_obs-ins_glo_phy-temp-sal_nrt_assim_irr_202211` | `CO_PR_PF_{YYYYMMDD}_MERC.nc` |
| Drifter SST | same | same | `CO_TS_DB_{YYYYMMDD}_MERC.nc` |
| Drifter currents | `INSITU_GLO_PHY_UVASSIM_DISCRETE_NRT_013_054` | `cmems_obs-ins_glo_phy-cur_nrt_drifter-filt-assim_irr_202311` | `GL_TS_DC_{YYYYMMDD}_FILTR.nc` |
| SLA | DUACS L3 `my` PT1S | `alg, c2n, h2b, s3a, s3b, s6a_lr, swon` | via `copernicusmarine.get` |

In-situ key layout: `native/{product}/{dataset}/{YYYY}/{MM}/{file}` on bucket
`mdl-native-08`, endpoint `https://s3.waw3-1.cloudferro.com`.

### SLA missions and their 2024 coverage

Days of 2024 present in `SEALEVEL_GLO_PHY_L3_MY_008_062`, listed 2026-08-05 with
the file pattern `*_1hz_2024*`:

| mission | dataset | days in 2024 |
| --- | --- | --- |
| `alg` | `cmems_obs-sl_glo_phy-ssh_my_alg-l3-duacs_PT1S` | 366 |
| `c2n` | `cmems_obs-sl_glo_phy-ssh_my_c2n-l3-duacs_PT1S` | 365 |
| `h2b` | `cmems_obs-sl_glo_phy-ssh_my_h2b-l3-duacs_PT1S` | 281 |
| `s3a` | `cmems_obs-sl_glo_phy-ssh_my_s3a-l3-duacs_PT1S` | 366 |
| `s3b` | `cmems_obs-sl_glo_phy-ssh_my_s3b-l3-duacs_PT1S` | 366 |
| `s6a_lr` | `cmems_obs-sl_glo_phy-ssh_my_s6a-lr-l3-duacs_PT1S` | 366 |
| `swon` | `cmems_obs-sl_glo_phy-ssh_my_swon-l3-duacs_PT1S` | 366 |

The legacy notebook asked for `al` (SARAL nominal orbit). That dataset now has
zero 2024 files: the catalogue splits the drifting phase into `alg`, which is
what the legacy build actually received. Requesting `al` is the reason the
2024-06-15 pull returned only five satellites. `h2b` is a genuine data gap, the
mission covers 281 of 366 days and has no file on 2024-06-15, so a normal day
yields six or seven missions.

`j3n` (Jason-3 interleaved) has all 366 days of 2024 and is deliberately left
out: it was not in the legacy set, and adding it raises the 2024-06-15 SLA count
by 50220 points, about 18 percent. It is a basis change, not a bug fix.

Zero 2024 coverage, checked and not candidates: `al`, `c2`, `en`, `enn`, `g2`,
`h2a`, `h2ag`, `j3`, `j3g`, `swonc`.

## Schema

Single dimension `obs`, zarr v2, consolidated metadata.

**Longitude convention.** Every row is on [-180, 180), enforced once on the
combined frame by `normalize_longitude()` before the policy runs. The in-situ
products already publish that convention, but the DUACS L3 along-track files
publish 0..360, so without this the SLA rows east of 180 fall off a -180..180
model grid and score as missing with no error raised.

### Legacy columns (names and dtypes unchanged, existing scorer reads them as-is)

| column | dtype | source | semantics |
| --- | --- | --- | --- |
| `depth` | float64 | `DEPH` / 0.0 for SLA | metres, positive down |
| `latitude` | float64 | `LATITUDE` | degrees north |
| `longitude` | float64 | `LONGITUDE` | degrees east, normalised to [-180, 180) for every stream |
| `time` | U19 string | `TIME` / `JULD` / `time` | `YYYY-MM-DDTHH:MM:SS`, second truncated, kept as string for compatibility |
| `sea_surface_height_above_geoid` | float64 | `sla_filtered` | metres, NaN unless the row passes policy |
| `sea_water_potential_temperature` | float64 | `TEMP` | degC, NaN unless it passes policy and temp QC |
| `sea_water_salinity` | float64 | `PSAL` | psu, NaN unless it passes policy and psal QC |
| `eastward_sea_water_velocity` | float64 | `EWCT_FILTR` | m/s, NaN unless it passes policy |
| `northward_sea_water_velocity` | float64 | `NSCT_FILTR` | m/s, NaN unless it passes policy |

### New columns

| column | dtype | source | semantics |
| --- | --- | --- | --- |
| `time_ns` | datetime64[ns] | decoded source time | full precision UTC, naive |
| `obs_type` | int8 | derived | 1 argo_profile, 2 drifter_sst, 3 drifter_current, 4 sla |
| `obs_id` | U96 | derived | see below |
| `platform_code` | U32 | `PLATFORM_CODE` / `PLATFORM_NUMBER` / `{mission}_c{cycle}_t{track}` | platform identifier |
| `platform_source` | U64 | `WMO_INST_TYPE`, falling back to `SOURCE` when it is blank, `duacs_l3_my` for SLA | instrument or provider |
| `argo_cycle` | int32 | `CYCLE_NUMBER` | -1 when absent. The Coriolis `CO_PR_PF_*_MERC.nc` files carry no `CYCLE_NUMBER`, so this is -1 everywhere for now |
| `data_mode` | U1 | `DATA_MODE` | R, D, A or empty |
| `temp_qc`, `psal_qc`, `deph_qc`, `position_qc`, `time_qc` | int8 | source `*_QC` | OceanSITES table 2, 9 = missing |
| `temp_raw`, `psal_raw` | float64 | `TEMP`, `PSAL` | pre-QC-mask values, never blanked |
| `temp_adjusted`, `psal_adjusted` | float64 | `TEMP_ADJUSTED`, `PSAL_ADJUSTED` | NaN when the product does not carry them |
| `temp_adjusted_qc`, `psal_adjusted_qc` | int8 | `*_ADJUSTED_QC` | 9 when absent |
| `uo_raw`, `vo_raw` | float64 | `EWCT`, `NSCT` | unfiltered currents, inertial band present |
| `uo_ws`, `vo_ws` | float64 | `EWCT_WS_FILTR`, `NSCT_WS_FILTR` (fallback `*_WS`) | wind slippage correction at drogue depth |
| `uo_qc`, `vo_qc` | int8 | `EWCT_FILTR_QC`, `NSCT_FILTR_QC` | QC of the basis actually used |
| `ws_type` | int8 | `WS_TYPE_OF_PROCESSING` | wind slippage method, -1 when absent |
| `current_test` | int32 | `CURRENT_TEST` | 3-digit SAW drogue-loss code, -1 fill |
| `drogued` | int8 | derived from `current_test` | 1 drogued, 0 undrogued (code 011), -1 unknown |
| `sla_unfiltered` | float64 | `sla_unfiltered` | metres |
| `sla_mission` | U8 | loop key | `alg`, `c2n`, `s3a`, ... |
| `sla_flag_keep` | int8 | derived | 1 if inside the SLA bound, 0 outside, -1 for non-SLA rows |
| `qc_keep` | int8 | derived | 1 if the row passes the default policy, else 0 |
| `qc_reason` | U48 | derived | first failing check: `position_qc`, `time_qc`, `day_misaligned`, `deph_qc`, `temp_psal_qc`, `current_qc`, `undrogued`, `sla_out_of_bounds` |

`uo_qc` and `vo_qc` are additions beyond the brief's list, needed because the
current basis has its own QC variables distinct from `temp_qc` and friends.

### obs_id and collisions

`obs_id = "{obs_type}:{platform_code}:{time_ns isoformat with microseconds}:{depth:.2f}:{group}"`
where group is `ts`, `sst`, `cur` or `sla`. The key never includes a measured
value, so a re-run that changes a value keeps the same identifier. If two rows
inside a day produce the same key, the second and later ones get `-1`, `-2`
appended in file order. Dedup then keeps one row per obs_id, preferring a
`qc_keep=1` row over a `qc_keep=0` one (`first_kept`).

## Default policy

All of it lives in the `POLICY` dict at the top of the script and is written
verbatim into the zarr attrs and the manifest.

- Measurement QC accepts **flag 1 only** (good data), matching the legacy store.
  Position and time QC accept **1 and 2**.
- Depth QC additionally accepts **7** (nominal value). Surface drifters carry
  `DEPH_QC = 7` on every level because their depth is a nominal platform depth.
  Accepting 7 is not optional: without it the drifter SST stream does not exist
  at all, every row of it is rejected.
- Currents use **`EWCT_FILTR` / `NSCT_FILTR`** as `uo` / `vo`.
- Rows with `CURRENT_TEST == 11` (SAW code 011, drogue considered missing) are
  flagged out of the legacy columns. Unknown drogue status is kept.
- SLA kept when `|sla| <= 2.0 m`, else flagged.
- Rows whose `time_ns` falls outside the target UTC day are flagged
  `day_misaligned`, not silently included and not silently dropped.
- Dedup on `obs_id`, first kept.

A failing row is retained in the store with `qc_keep=0`, its flags and its raw
values. Only the legacy measurement columns are NaN for it.

Every row keeps its own `temp_qc`, `psal_qc`, `deph_qc`, `position_qc`, `time_qc`
and `uo_qc` / `vo_qc`, so relaxing QC is a scoring-time choice. A scorer that
wants flag 2 as well can select on the stored flags without a rebuild. The flag-1
default only decides what reaches the legacy measurement columns.

## Gates

The script refuses to write a day and exits nonzero when:

- any in-situ stream (currents, profiles, drifter SST) has zero files, or
- fewer than `--min-satellites` (default 5) SLA satellites were found and the
  date is older than about 6 months. Seven missions are requested, `h2b` is
  absent on 85 days of 2024, so six is the normal yield and five is one short of
  that. For a more recent date, an explicit
  `--allow-missing-sla` is required.

An existing target is never touched unless `--overwrite` is given. Writes always
go to `{YYYYMMDD}.zarr.tmp` and are renamed on success.

## Manifest

Alongside each zarr, `{YYYYMMDD}.manifest.json` records every source file (key,
size, etag or sha256 prefix, download time, rows extracted), per-stream row
counts before and after policy, duplicates removed, SLA satellites found,
package versions, the policy dict, and the sha256 of the builder script itself.
Resume logic reads this manifest, so a day is only skipped when its manifest says
`written` and the zarr carries consolidated metadata.

## Usage

```bash
export COPERNICUSMARINE_SERVICE_USERNAME=...
export COPERNICUSMARINE_SERVICE_PASSWORD=...
export CF_KEY=...
export CF_SECRET=...

# local smoke test on one day
python build_observations.py --dates 2024-06-15 \
  --target /scratch/jseillade/obs-v2-test \
  --tmp-root /scratch/jseillade/tmp

# the 2024 build on lir
python build_observations.py \
  --start 2024-01-01 --end 2024-12-31 \
  --target s3://oceanbench-bucket/dev/observations2024-v2 \
  --obs-basis-version 2024-v2.0.1 \
  --tmp-root /scratch/jseillade/tmp \
  --archive-dir /scratch/jseillade/obs-v2-raw \
  --workers 6 \
  --results-csv /scratch/jseillade/obs-v2-results.csv
```

Launch it detached (`setsid nohup ... > log 2>&1 &`) rather than in a login
shell. `--workers N` runs N days in parallel processes; the bottleneck is the
copernicusmarine download of 7 SLA datasets per day, and CloudFerro saturates
near 8 concurrent streams, so 6 is a reasonable start.

## Expected sizes

Per day, roughly: SLA about 1.2 to 1.8 M rows across 7 satellites, drifter
currents about 25 to 30 k rows, drifter SST and Argo profiles a few hundred
thousand levels combined. The new store carries about 30 extra columns, most of
them int8, so expect roughly 2 to 3 times the legacy per-day size after zarr
compression, order 50 to 120 MB per day and order 20 to 40 GB for a full year.
With `--archive-dir` set, the raw netCDF copies add order 1 GB per day, which is
the dominant cost. Measure one day before committing the disk.

## Differences vs the legacy store

1. **FILTR basis.** Legacy used raw `EWCT` / `NSCT`, which contain the inertial
   band. The legacy velocity columns now carry `EWCT_FILTR` / `NSCT_FILTR`
   (3-day Lanczos). The raw values remain in `uo_raw` / `vo_raw`.
2. **Drogue drop.** Legacy kept undrogued drifters, whose velocities are surface
   circulation contaminated by direct wind drag. Rows with `CURRENT_TEST == 11`
   are now flagged out of the legacy columns. Legacy had no drogue information at
   all, so this cannot be reconstructed from the old store.
3. **QC flag 1.** Unchanged from legacy: measurement QC accepts flag 1 only.
   Earlier drafts of this rebuild accepted 1 and 2; that is now a scoring-time
   option rather than the default, since the flags are stored per row.
4. **SLA bound.** Legacy kept any finite SLA. Values beyond 2 m are now flagged
   out, with `sla_flag_keep` recording the decision.
5. **Dedup.** Legacy did not deduplicate, so a platform reported twice in a day
   could be counted twice. Dedup is now on `obs_id`.
6. **Aligned days.** Legacy took whatever timestamps the daily file contained.
   Rows outside the target UTC day are now flagged `day_misaligned`.
7. **Flag don't drop.** Legacy dropped failing rows during extraction. Every row
   is now present, so the store is auditable and the policy can be replayed
   without a rebuild.

### Measured on the 2024-06-15 smoke test

| stream | v2 vs legacy | cause |
| --- | --- | --- |
| drifter currents | -14.1% | undrogued rows flagged out of the legacy columns |
| temperature | -0.7% | dedup, day alignment, position and time QC |
| salinity | about +0.2% | measured under the earlier QC 1 and 2 default, expected to shrink or turn slightly negative under flag 1 only |
| SLA | 269072 vs 272174, -1.14% | six missions found after the `al` to `alg` fix, against a raw file total of 273578 for the same six; the residual is the 2 m bound, day alignment and dedup |

The currents, temperature and salinity numbers were measured before the QC flip
to flag 1 only and will move slightly; the SLA number is measured with the final
mission list.

Because of 1, 2 and 4, scores computed against this store are not comparable to
scores computed against the legacy store. A rescore of every challenger is
required.

## Changelog

### 2024-v2.0.1 (2026-08-05)

**SLA longitude convention.** The SLA extractor passed DUACS L3 longitude
through unchanged on 0..360 while every in-situ stream was on -180..180 and the
legacy store wrote SLA on -180..180. Scored as built, every SLA row east of 180
fell off the model grid and returned a NaN model value, silently: the ladder
rescore measured global SLA counts dropping from 15.71 M to 6.77 M at lead 1
with RMSD about 10 percent worse, and IBI SLA collapsing from 181064 to 17893
points with RMSD 70 percent worse. Fixed by `normalize_longitude()`, applied
once to the combined frame so every stream shares one convention.

The 370 published days were patched in place rather than rebuilt: the defect
touches exactly one float64 column on the `obs_type == 4` rows, and `obs_id`
never includes longitude, so no row identity, count or other column changes.
Each patched day records the operation in its manifest under `patches` and
carries `obs_basis_version = 2024-v2.0.1`.

### 2024-v2.0.0 (2026-08-05)

Initial published build of the rebuilt store, 370 days.
