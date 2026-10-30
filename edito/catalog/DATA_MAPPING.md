<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->
# OceanBench data mapping (AI-160)

Bucket: `https://s3.waw3-1.cloudferro.com/oceanbench-bucket` (listed anonymously on 2026-10-30).
Stores are weekly Zarr named by first day unless stated otherwise.

## Used by the library

### Challengers → collection `challengers`

| Item | Location | Code |
|---|---|---|
| glonet | `public/ml-forecast-outputs/glonet/` (157, 2023-01-04..2025-12-31) | `challenger_datasets.py` |
| xihe | `public/ml-forecast-outputs/xihe/` (157, 2023-01-04..2025-12-31) | `challenger_datasets.py` |
| wenhai | `public/ml-forecast-outputs/wenhai/v2/` (157) | `challenger_datasets.py` |
| langya | `public/ml-forecast-outputs/langya/v2/` (52, 2024) | `challenger_datasets.py` |
| hclimrep | `public/ml-forecast-outputs/hclimrep/` (52, 2024) | `challenger_datasets.py` |
| glo12_persistence | `public/baseline-forecasts/persistence/` (52, 2024) | `challenger_datasets.py` |
| glo36v1 | `public/glo36v1/` (53, 2023-01-04..2024-01-03) | `challenger_datasets.py` |
| glo12 | `dev/additionnal-data/GLO12/glo12_rg_1d-m_fcst_R<date+1>.zarr` (grouped by variable) | `challenger_datasets.py` |

Each `*_1_degree` variant is interpolated on the fly: no stored data.

### References → collection `references`

| Item | Location | Code |
|---|---|---|
| glorys 1/4° | `public/glorys14_refull_2024/` (52) | `references/glorys.py` |
| glorys 1° | `public/glorys_1degree_2024_V2/` (52) | `references/glorys.py` |
| glorys 1/12° | Copernicus Marine `cmems_mod_glo_phy_my_0.083deg_P1D-m` | `references/glorys.py` |
| glo12 analysis 1/4° | `public/glo14/` (52) | `references/glo12.py` |
| glo12 analysis 1° | `public/glo12_1degree_2024_V2/` (52) | `references/glo12.py` |
| glo12 analysis 1/12° | Copernicus Marine `cmems_mod_glo_phy-{thetao,so,cur}_anfc_0.083deg_P1D-m`, `cmems_mod_glo_phy_anfc_0.083deg_P1D-m` | `references/glo12.py` |
| observations 2024 (Class 4) | `public/observations2024-v2/<YYYYMMDD>.zarr` + `.manifest.json` (370, 2024-01-01..2025-01-04) | `references/observations.py` |
| mean dynamic topography | `public/GLO12_MDT/GLO-MFC_001_024_mdt{,_025deg,_1_deg}.zarr` | `references/observations.py` |
| ocean mask v2 | `public/ocean_mask/oceanbench-ocean-mask-v2.zarr` (built from Copernicus Marine `cmems_mod_glo_phy_anfc_0.083deg_static`, `cmems_mod_glo_phy_my_0.083deg_static`) | `ocean_mask.py` |

### Input → collection `inputs`

| Item | Location | Code |
|---|---|---|
| glo12 nowcasts | `dev/additionnal-data/GLO12/glo12_rg_1d-m_nwct_R<date>.zarr` (2023-01-04..2025-12-31) | `input_datasets.py` |
| ifs forcings | `dev/additionnal-data/IFS/ifs_forcing_rg_forecasts_R<date>.zarr` (2023-01-03..2025-12-30) | `input_datasets.py` |

### Training → collection `training`

| Item | Location | Code |
|---|---|---|
| glorys 1° 1993-2019 | `public/glorys_1degree_1993_2019/<YYYYMM>.zarr` (324) | `training_datasets.py` |
| glorys 1/12° 1993-2019 | Copernicus Marine `cmems_mod_glo_phy_my_0.083deg_P1D-m` | `training_datasets.py` |

### Used by helper scripts only

- `public/GLO12_NOWCAST/` (52 stores, 2024): input of `helper_scripts/baseline_generation/generate_persistence_forecasts.py`.

## In the bucket, not used by the library

All of them are catalogued as items with `oceanbench:library_usage: unused`.
Decision pending: **integrate** (I) or **drop** (D). Proposals in the last column.

| Prefix | Content | Size / count | Proposal |
|---|---|---|---|
| `public/IFS/` | `IFS_<date>.nc`, 2024; referenced by `input_datasets/metadata.json` (Croissant) | 52 files, 147 GB | I as input (documented, public) |
| `public/GLO12/` (+ `.nc` in `public/GLO14/`) | 2024 zarr/NetCDF duplicates of GLO12 analysis | 52 zarr; 53 nc, 171 GB | D (superseded by `glo14`) |
| `public/GLO12_NOWCAST_NATIVE/` | native-grid nowcast NetCDF | 271 files, 268 GB | I as input (native resolution), else D |
| `public/GLO12_MDT/*.nc` | NetCDF copies of the MDT zarr | 2 files, 55 MB | D |
| `public/glorys12_mdt_2024/`, `public/glorys12_mean_sea_surface_height_2024/`, `class4/Glorys_mssh2024/`, `class4/statics/` | GLORYS MDT variants (`GLO-MFC_001_030`) | small | I (as reference statics) or D |
| `public/glo12_1degree_2024/`, `public/glorys_1degree_2024/` | 2 stores each, v1 | tiny | D (superseded by V2) |
| `public/glo12_1degree_2024_coarsened/`, `public/glorys_1degree_2024_coarsened/` | coarsened 1° variants | 51-52 | I or D |
| `public/glonet_full_2024/`, `public/glorys14_full_2024/` | superseded by `ml-forecast-outputs/glonet`, `glorys14_refull_2024` | 52 each | D |
| `public/WENHAI/`, `public/XIHE/` | 2024 duplicates of `ml-forecast-outputs` | 52 each | D |
| `public/ml-forecast-outputs/langya/` (no `v2`) | LangYa version 1, 2024-2025 | 105 | D (keep `v2`) |
| `public/ml-forecast-outputs/hclimrep/v2/` | HClimRep version 2 (the library uses version 1) | 52 | I (switch the library) or D |
| `public/baseline-forecasts/climatology/` | climatology baseline, 2024 | 52 | I as challenger |
| `public/_quarantine-langya-bareload-bug-20260623/` | quarantined buggy langya | 141 | D |
| `public/ibirys36_1-12_glorys-grid/` | IBI reanalysis on GLORYS grid, 202401 | 1 | I as reference or D |
| `public/ifs-nowcast/`, `ifs-nowcast-corrected/`, `ifs-nowcasts24/`, `ifs-nowcasts/` | IFS nowcast experiments | 52 each | D unless documented |
| `public/observations2023/` | 2023 observations | 365 | I as reference (matches glo36v1 2023) |
| `public/observations2024/` | v1, superseded by `observations2024-v2` | 370 | D |
| `public/observations2025/` | 2025-01-01..2026-05-31 | 516 | I as reference |
| `public/obs_finales_v2/` | 15 days, 2024-01 | 15 | D |
| `public/live_observations/` + `availability.json` | rolling daily 2026-10 | 10 | I as reference (live) |
| `public/evaluation-reports{,-multi-year}/`, `public/logo/` | website artifacts | small | out of scope for the catalog |
| `class4/SLA/`, `class4/scores/`, `class4/swot/l3/**`, root `class4/*` | legacy class 4 files, SWOT L3 (NetCDF + zarr), scripts | 6042 files, 124 GB | I SWOT as reference, D the rest |
| `scores/{glo12,glorys}/` | score NetCDF per challenger | 10 files | D (superseded by reports) |
| `dev/additionnal-data/OLA/` | `.dat` catalogs | 2 files, 450 MB | D |
| `dev/additionnal-data/IFS/ocean_bench_R*.zarr` | non-conforming names | few | D |
| `dev/observations2024-v2/` | duplicate of public | 370 | D |
| `dev/observations2025-v2/`, `dev/observations2025-v2-smoke/` | 2025 v2 observations (staging) | 369 / 3 | I once promoted to `public/` |
| `dev/ocean_mask/` (v1) | superseded | 1 | D |
| `dev/artifacts/`, `dev/ensemble/` | `sigma-lookup-v3.{0,1}.0.zarr` | 1 / 2 | D from catalog (internal) |
| `dev/ml-forecast-outputs/**` | experimental challengers: glonet2(-ens*), glowcascade*, smoke tests | many | Keep out of catalog until promoted to `public/` |
| `dev/evaluation-reports/`, `dev/benchmark/` | preview reports | small | out of scope |

Not catalogued, as they are not datasets: `public/evaluation-reports*/`, `public/logo/`, `dev/evaluation-reports/`, `dev/benchmark/`, scripts, `provenance.json`, `summary.json`, `availability.json`.

## Catalog layout

- Project catalog `oceanbench`
  - Collections `oceanbench-challengers`, `oceanbench-references`, `oceanbench-inputs`, `oceanbench-training`, `oceanbench-evaluations`
  - One item per dataset (not per store), declared in [`datasets.py`](datasets.py).
    Assets point to the first and latest objects and the bucket listing; examples use the `example-links` extension (OceanBench library, xarray, pandas or Copernicus Marine toolbox).

Run `make update-edito-catalog CATALOG_OUTPUT_DIRECTORY=<directory>` to write the catalog locally,
or `make update-edito-catalog` with `EDITO_OFFLINE_TOKEN` to publish it.
`OCEANBENCH_CATALOG_ENVIRONMENT=<name>` publishes a private copy suffixed by `<name>`.
The [Update EDITO catalog](../../.github/workflows/update-edito-catalog.yml) workflow launches it hourly as an EDITO generic process.
