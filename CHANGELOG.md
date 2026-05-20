<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Changelog

All notable changes to OceanBench are documented in this file.

## 0.1.4 - 2026-05-20

### Fixed

- Fixed Class IV observation staging so observations in overlapping forecast windows are preserved for each matching forecast.
- Prevented Class IV observation evaluations from reusing overlap-unsafe staged cache.
- Fixed website report discovery tests so expected report URLs use the shared report version configuration.

### Reports

- Official reports: `public/evaluation-reports/0.1.4/`

## 0.1.3 - 2026-05-13

### Fixed

- Fixed lagrangian local staging so cached surface-current inputs are keyed by the evaluated horizontal domain.
- Prevented global and regional lagrangian evaluations from reusing each other's staged cache.

### Reports

- Official reports: `public/evaluation-reports/0.1.3/`

## 0.1.2 - 2026-05-12

### Fixed

- Fixed local staging of 1/12-degree GLORYS and GLO12 references so cached reference datasets are keyed by challenger depth grid.
- Prevented derived lagrangian reference caches from mixing reference cache variants.

### Reports

- Official reports: `public/evaluation-reports/0.1.2/`

## 0.1.1 - 2026-05-06

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

### Added

- Interactive OceanBench website based on generated evaluation notebooks.
- Class IV observation validation support.
- Local staging and remote retry support for large evaluations.
- Official global benchmark report storage under `public/evaluation-reports/0.1.0/`.

### Changed

- Evaluation reports are generated as notebooks and parsed by the website to populate benchmark score tables.

### Reports

- Official reports: `public/evaluation-reports/0.1.0/`
