<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Changelog

All notable changes to OceanBench are documented in this file.

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
