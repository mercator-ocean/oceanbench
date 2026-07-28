<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Evaluation (`oceanbench evaluate`)

Score a forecast and get the scores file the benchmark website reads. The target is either
**your own model** or the slug of a **challenger already in the benchmark**. Your own model
also gets a self-contained overlay scorecard laying it over the published challengers, so you
can see where you stand before publishing.

References and observations are read **live from the public EDITO objects**, so there is no
download step to run first.

## Score your own forecast

```sh
oceanbench evaluate ./my-forecasts.zarr
```

That writes `scores.parquet`, `scores-summary.json` and `scorecard/index.html` under
`./oceanbench-evaluation`. Open the scorecard by double-clicking it; no server needed.

Narrow the run to what you care about:

```sh
oceanbench evaluate ./my-forecasts.zarr \
  --output ./my-evaluation \
  --region gulfstream \
  --metrics rmsd class4
```

## Score a challenger already in the benchmark

Pass its slug instead of a path; the library opens its published forecasts itself:

```sh
oceanbench evaluate glonet_1_degree
```

Published challengers are aggregated over the **same forecast starts** as your model, so
re-scoring a published challenger this way reproduces its published row exactly.

## Add the map viewer

Scores are the only default output. `--viewer-artifacts` additionally builds everything the map needs
(Class-4 match-up parquet, eddy census, field pyramid, year-mode JSON) plus a local viewer site:

```sh
oceanbench evaluate ./my-forecasts.zarr --viewer-artifacts --output ./my-evaluation
python -m http.server --directory ./my-evaluation/viewer 8799
# open http://127.0.0.1:8799/?data_base=local
```

Building it reads the current official viewer catalog, so `--viewer-artifacts` needs network access.
Your own pyramid, scores and scorecard are then fully local; the official comparison layers
are fetched from the public EDITO MinIO objects named by `datasets.json` as you browse them.

## Running without network: `--offline-references`

An **offline reference bundle** is a downloadable, versioned directory built by `ingest` from
the staged reference data, holding everything `evaluate` would otherwise read live. Point at
one to run with no network at all:

```sh
oceanbench evaluate ./my-forecasts.zarr --offline-references ./pack-quick-2024
```

It is an optimisation for offline or repeated runs, never a prerequisite. A bundle is
**self-describing**: `evaluate` reads `pack-manifest.json` alone to locate every source. It
carries:

- `references/<name>.zarr` — the gridded references (GLORYS, GLO12). A `quick` bundle carries
  **surface fields only**; a `full` bundle carries all depths.
- `observations/observations.zarr` — the Class-4 observation match-up store.
- `class4-mean-dynamic-topography-…​.zarr` — the GLO12 mean dynamic topography used for the
  SSH → SLA Class-4 conversion (stored under its stage-canonical name so the ported Class-4
  code resolves it offline).
- `pack-manifest.json` — stamps the upstream products + retrieval dates (contracts.md §1) and
  carries the evaluation year and region, plus the Copernicus Marine credit and
  disclaimer (contracts.md §11).
- `README.md` — the Copernicus Marine attribution, verbatim.

The bundle's manifest **fixes the year and region**, so `--region` or `--year` that contradict
it are rejected rather than silently ignored. Your model is scored on the intersection of its
own forecast starts with the bundle's starts.

Baselines (climatology / persistence) are not available in a bundle yet: the manifest supports
optional baseline entries and flags their absence (`baselines_available: false`), so
skill-vs-baseline is not computed offline until a bundle ships baselines.

## Required forecast layout

Your forecast must follow the **weekly-store conventions of the challenger datasets**. Two
layouts are accepted:

1. **A single combined zarr** with dims
   `(first_day_datetime, lead_day_index, depth, latitude, longitude)` and the CF-named forecast
   variables (`sea_surface_height_above_geoid`, `sea_water_potential_temperature`,
   `sea_water_salinity`, `eastward_sea_water_velocity`, `northward_sea_water_velocity`).
   `first_day_datetime` is the forecast start date; `lead_day_index` is 0-based.

2. **A directory of weekly zarr stores** named `YYYYMMDD.zarr`, one per forecast start, each
   with a `time` lead-day dimension — the same shape a challenger publishes. The stores are
   concatenated on the forecast-start axis.

## What it produces

Under `--output`:

- `scores.parquet` — the standard long-format per-start records (contracts.md §3.1): gridded
  RMSD + geostrophic currents, Class-4 RMSD, and the realism battery (spectra / activity /
  eddies).
- `scores-summary.json` — the aggregated means and 95% bootstrap CIs (contracts.md §3.4),
  produced by the same aggregation library as the hosted page.
- `scorecard/index.html` — a self-contained overlay scorecard. Written for your own forecast
  only: a challenger that is already published has nothing to overlay.
- With `--viewer-artifacts`, `insights/<slug>/<region>/` — the match-up parquet, eddy census and
  year-mode JSON, and `viewer/` — the static viewer application, `data/<slug>.zarr`, its viewer
  manifest, and a mixed `data/datasets.json`. The local descriptor uses relative URLs; official
  descriptors use absolute public MinIO URLs.

## The overlay scorecard

The scorecard reuses the no-ranking scorecard semantics of the website scores page
(`website-rebuild/scores/`): mean ± 95% CI over forecast starts, baselines pinned, neutral
order, no composite score. Your model is laid over the official published challengers
and highlighted as **"your model"**.

The overlay data is **inlined** into `index.html` (not fetched), and the renderer is a classic
script (not an ES module). This is deliberate: a page opened over `file://` cannot `fetch()`
sibling files nor load an ES module — both are blocked by the browser same-origin policy for
`file://`, which is exactly what the website's `app.js` relies on. Inlining lets you **open
`index.html` with a plain double-click, no server required**. The aggregation itself is the same
`oceanbench.publish.aggregate` code the hosted page uses, so the numbers match.

## Flags

| flag | meaning |
|---|---|
| `--output DIR` | output directory (default: `./oceanbench-evaluation`) |
| `--region REGION` | region to score over (default: `global`) |
| `--year YEAR` | evaluation year (default: `2024`) |
| `--metrics M [M ...]` | select metric families: `rmsd`, `mld`, `geostrophic`, `class4`, `lagrangian`, `realism` (default: all) |
| `--viewer-artifacts` | also build the map viewer and its serving artifacts (off by default) |
| `--offline-references DIR` | read references and observations from a downloaded bundle instead of live EDITO |

Published scores, challenger metadata, and viewer datasets use the official MinIO release.
Set `OCEANBENCH_PUBLISHED_BASE` to override that base URL.
