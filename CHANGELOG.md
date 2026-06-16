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
