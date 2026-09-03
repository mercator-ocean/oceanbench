<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Changelog

All notable changes to OceanBench are documented in this file.

**How to read this file.** The **version number tracks how scores are computed.** A new
version means the scoring methodology changed and scores are **not directly comparable** to
the previous version — every challenger is re-evaluated. Each version begins with a one-line
**Scores** summary stating whether and why scores moved.

Changes that do **not** change the methodology — a challenger added, or a challenger's
forecasts refreshed — do **not** bump the version. They are recorded as dated bullets under a
**Challengers** heading in the current version's section, and the affected reports are
re-published (never silently overwritten).

## 0.5.1 - 2026-09-02

**Scores:** unchanged vs 0.5.0. This release only moves public data access paths from the
previous EDITO MinIO project buckets to the CloudFerro `oceanbench-bucket` public area.

### Fixed

- Updated package, documentation, website and metadata data URLs to read OceanBench public
  datasets and report assets from CloudFerro instead of the previous EDITO MinIO project
  buckets.

### Reports

- Official reports: unchanged from `public/evaluation-reports/0.5.0/`.

## 0.5.0 - 2026-08-13

**Scores:** grid-averaged scores change for the challengers whose grids do not land exactly on
the reference grid (GLO36v1, LangYa, WenHai, XiHe) vs 0.4.0 — the reference grid is now snapped
onto the challenger grid before the difference is taken, so points that were previously
misaligned by a fraction of a grid cell are compared like for like. Challengers already on the
reference grid (GLONET, GLONET 1 degree) are unchanged, and the observation-based Class IV
scores (temperature, salinity, sea level anomaly and currents vs observations) are unchanged for
every challenger. See the
[evaluation methods documentation](https://oceanbench.readthedocs.io/en/latest/evaluation-methods.html).

### Changed

- Reference latitude and longitude coordinates are snapped to the challenger grid with a 1e-4 degree nearest-neighbour tolerance before grid-averaged RMSD is computed, instead of relying on exact coordinate equality ([#308](https://github.com/mercator-ocean/oceanbench/issues/308), [#305](https://github.com/mercator-ocean/oceanbench/issues/305)).
- Spatial alignment now raises an explicit error when it is ambiguous, or when fewer than 99.9% of the challenger grid points can be matched, instead of silently producing a misaligned comparison.
- Class IV vertical interpolation supports up to 128 depth levels instead of 64; the depth level count is no longer encoded in a fixed-width integer ([#307](https://github.com/mercator-ocean/oceanbench/issues/307)).

### Reports

- Official reports: `public/evaluation-reports/0.5.0/`

## 0.4.0 - 2026-07-07

**Scores:** every score computed against the gridded GLORYS/GLO12 reference changes for every
challenger vs 0.3.0 — grid-averaged RMSD (variables, mixed-layer depth, geostrophic currents)
now uses cos(latitude) area weighting, and Lagrangian trajectory seeds are drawn with the same
cos-latitude weighting, so scores reflect skill per unit ocean area rather than per grid cell.
The observation-based Class IV scores (temperature, salinity, sea level anomaly and currents vs
observations) are unchanged. See the
[evaluation methods documentation](https://oceanbench.readthedocs.io/en/latest/evaluation-methods.html).

### Changed

- Grid-averaged RMSD — variables, mixed-layer depth and geostrophic currents against both GLORYS and GLO12 — now uses cos-latitude area weighting instead of an unweighted lat/lon mean.
- Lagrangian trajectory seed points are drawn with cos-latitude area probabilities.

### Reports

- Official reports: `public/evaluation-reports/0.4.0/`

## 0.3.0 - 2026-07-07

**Scores:** global sea-level-anomaly scores change for every challenger vs 0.2.1 — global SSH
is now converted to SLA using the GLO12 mean dynamic topography instead of the GLORYS MDT,
matching the GLO12 datum the global challengers are initialised on and correcting the
overestimated SLA reported in [#293](https://github.com/mercator-ocean/oceanbench/issues/293).
IBI SLA and all other scores are unchanged.

### Changed

- Global SSH→SLA conversion now uses the GLO12 mean dynamic topography (GLO-MFC_001_024) paired with the GLO12 datum shift, replacing the GLORYS MDT ([#295](https://github.com/mercator-ocean/oceanbench/issues/295)).

### Reports

- Official reports: `public/evaluation-reports/0.3.0/`

## 0.2.1 - 2026-06-16

**Scores:** GLO12, GLONET, XiHe and WenHai change vs 0.2.0. The GLONET, XiHe and WenHai
forecasts were recomputed with updated GLO12 nowcast initial conditions and IFS atmospheric
forcings (WenHai substantially, correcting the surface forcing in #269; GLONET and XiHe
slightly). GLO12 now uses the full GLO12 operational forecast (50 depth levels). Methodology
is unchanged from 0.2.0.

### Challengers

- 2026-06-30 — WenHai forecasts recomputed to mask initial-condition land as NaN, fixing the spurious minimum-value points from the land-sea mask mismatch ([#294](https://github.com/mercator-ocean/oceanbench/issues/294)).
- 2026-06-23 — LangYa added: a machine-learning model from IOCAS producing 7-day global ocean forecasts initialized from GLO12 nowcasts and IFS atmospheric forcings.
- 2026-06-16 — GLONET, XiHe and WenHai forecasts recomputed with updated GLO12 nowcasts and IFS atmospheric forcings. For WenHai this resolves the surface-forcing issue reported in [#269](https://github.com/mercator-ocean/oceanbench/issues/269): the model is now forced with net shortwave radiation, replacing the previously corrupted shortwave input that had inflated its error.
- 2026-06-16 — GLO12 now reads the full GLO12 operational forecast (50 depth levels), replacing the previous reduced-depth product.

### Changed

- Updated the GLO12, GLONET, XiHe and WenHai challenger dataset sources to the new forecasts.

### Reports

- Official reports: `public/evaluation-reports/0.2.1/`

## 0.2.0 - 2026-06-15

**Scores:** change vs 0.1.4 — the 600 m mixed-layer-depth cap and the one-day Class IV
observation realignment change computed scores. 1-degree challenger scores added.

### Added

- 1-degree evaluation track with 1-degree challenger and reference datasets (`glo12_1_degree`, `glonet_1_degree`, `wenhai_1_degree`, `xihe_1_degree`).
- Weekly GLO12 nowcast and IFS forcing input datasets covering 2023-2025, exposed via `oceanbench.datasets.input`; see the [input datasets documentation](https://oceanbench.readthedocs.io/en/latest/input-datasets-for-oceanbench-challenger-evaluation.html).
- Historical version selector on the scores website, with report discovery driven by a published version index.

### Changed

- Optimized Class IV model interpolation by materializing each forecast first-day block once, with an opt-in fast path enabled by `OCEANBENCH_CLASS4_FAST_INTERPOLATION`.

### Fixed

- Aligned Class IV observations with forecast lead days, correcting a one-day offset in observation-to-lead-day matching.
- Capped native-grid mixed layer depth at 600 m and added a fallback to the deepest valid level when the density threshold is never crossed (previously the surface depth was returned).

### Reports

- Official reports: `public/evaluation-reports/0.2.0/`

## 0.1.4 - 2026-05-20

**Scores:** Class IV scores change vs 0.1.3 — observations in overlapping forecast windows
are now preserved for every matching forecast. Other scores unchanged.

### Fixed

- Fixed Class IV observation staging so observations in overlapping forecast windows are preserved for each matching forecast.
- Prevented Class IV observation evaluations from reusing overlap-unsafe staged cache.
- Fixed website report discovery tests so expected report URLs use the shared report version configuration.

### Reports

- Official reports: `public/evaluation-reports/0.1.4/`

## 0.1.3 - 2026-05-13

**Scores:** unchanged — local lagrangian staging correctness fix.

### Fixed

- Fixed lagrangian local staging so cached surface-current inputs are keyed by the evaluated horizontal domain.
- Prevented global and regional lagrangian evaluations from reusing each other's staged cache.

### Reports

- Official reports: `public/evaluation-reports/0.1.3/`

## 0.1.2 - 2026-05-12

**Scores:** unchanged — local reference staging correctness fix.

### Fixed

- Fixed local staging of 1/12-degree GLORYS and GLO12 references so cached reference datasets are keyed by challenger depth grid.
- Prevented derived lagrangian reference caches from mixing reference cache variants.

### Reports

- Official reports: `public/evaluation-reports/0.1.2/`

## 0.1.1 - 2026-05-06

**Scores:** IBI regional scores added. Global scores unchanged.

### Added

- Regional evaluation support.
- IBI benchmark region support.
- Region-aware report generation and website display.
- Official global and IBI benchmark report storage under `public/evaluation-reports/0.1.1/`.

### Changed

- Evaluation report filenames now include the evaluated region: `{challenger}.{region}.report.ipynb`.

### Reports

- Official reports: `public/evaluation-reports/0.1.1/`

## 0.1.0 - 2026-05-06

**Scores:** initial global benchmark.

### Added

- Interactive OceanBench website based on generated evaluation notebooks.
- Class IV observation validation support.
- Local staging and remote retry support for large evaluations.
- Official global benchmark report storage under `public/evaluation-reports/0.1.0/`.

### Changed

- Evaluation reports are generated as notebooks and parsed by the website to populate benchmark score tables.

### Reports

- Official reports: `public/evaluation-reports/0.1.0/`
