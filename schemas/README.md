<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# OceanBench v2 JSON Schemas

JSON Schema (draft 2020-12) definitions for the v2 data artifacts. Each schema
formalizes a structure specified in [`../docs/contracts.md`](../docs/contracts.md),
the authoritative v2 design contract.

| schema | contracts.md section | artifact |
|---|---|---|
| [`catalog.schema.json`](catalog.schema.json) | §5 (with §8 paths) | `catalog.json` at the artifact root — release → year → region → challenger index of `scores.parquet`, insight manifests and viewer zarrs. |
| [`insights-manifest.schema.json`](insights-manifest.schema.json) | §4 | `insights/manifest.json` — semantic insight key → `{kind, schema_version, url, bytes}`. |
| [`spectra.schema.json`](spectra.schema.json) | §4 (`spectra` kind) | PSD insight payload: per variable × region × lead {1,5,10}, shared `wavelength[]` with `challenger_power[]`, `reference_power[]`, `error_power[]`. |
| [`eddies.schema.json`](eddies.schema.json) | §4 (`eddies` kind) | Mesoscale-eddy census payload: per reference, per lead day, `matches` (with displacement km), `spurious`, `missed`; each eddy carries id, lat/lon (4 dp), polarity and contour polygon arrays. Shape adapted from branch 249. |
| [`viewer-manifest.schema.json`](viewer-manifest.schema.json) | §6 | Per-dataset `viewer-manifest.json` — levels, tile size, bounds, variables (units/scale/offset/default colormap+range), start_dates, lead_days. |
| [`challengers.schema.json`](challengers.schema.json) | §2 | `challengers.json` in-repo registry — canonical slug → metadata; baselines are `is_baseline: true`. |

## Conventions honored (contracts.md §2)

- `lead_day` is 1-based (1..10) everywhere.
- `null` encodes NaN in float arrays.
- Depth labels are machine keys (`surface`, `15m`, `50m`, …).
- Dates are ISO-8601.

## Notes

- Payload schemas (`spectra`, `eddies`) carry a `kind` discriminator and a
  `schema_version` so a manifest entry's `kind`/`schema_version` can be checked
  against the payload it points to.
- Schemas are strict (`additionalProperties: false`) on fixed-shape objects.
  Objects that are genuinely open maps keyed by a dynamic id — the catalog's
  release/year/region/challenger levels, the manifest's semantic keys, and the
  challenger registry — constrain their keys with `propertyNames` and their
  values with a `$ref`.
