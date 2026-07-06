<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Local evaluation (`oceanbench evaluate-local`)

Score your own ocean-forecast model locally against an OceanBench **evaluation pack**
(contracts.md §7) and get the same artifacts as the hosted run plus a self-contained
overlay scorecard that lays your model over the published challengers.

## Evaluate your own forecast

From scores to a browser comparison with the official products:

```sh
oceanbench evaluate-local ./my-forecasts.zarr --pack ./pack-quick-2024 --artifacts all --output ./my-evaluation
python -m http.server --directory ./my-evaluation/viewer 8799
# open http://127.0.0.1:8799/?data_base=local
```

The first command needs network access to read the current official viewer catalog; obtaining
the evaluation pack also needs network access unless it is already downloaded. The generated
challenger pyramid, scores, scorecard, and viewer application are fully local. Browsing the
challenger remains local, while displaying official comparison layers needs network access to
the public EDITO MinIO objects named by `datasets.json`.

```sh
oceanbench evaluate-local ./my-forecasts.zarr \
  --year 2024 --region global \
  --pack ./pack-quick-2024 \
  --published ./scores.parquet \
  --published-challengers ./challengers.json \
  --output ./my-evaluation \
  --starts-limit 8
```

## What a pack is

An evaluation pack is a downloadable, versioned directory built by `ingest` from the staged
reference data. It is **self-describing**: `evaluate-local` reads `pack-manifest.json` alone to
locate every reference. A pack carries:

- `references/<name>.zarr` — the gridded references (GLORYS, GLO12). A `quick` pack carries
  **surface fields only**; a `full` pack carries all depths.
- `observations/observations.zarr` — the Class-4 observation match-up store.
- `class4-mean-dynamic-topography-…​.zarr` — the GLO12 mean dynamic topography used for the
  SSH → SLA Class-4 conversion (stored under its stage-canonical name so the ported Class-4
  code resolves it offline).
- `pack-manifest.json` — stamps the upstream products + retrieval dates (contracts.md §1) and
  carries the Copernicus Marine credit and disclaimer (contracts.md §11).
- `README.md` — the Copernicus Marine attribution, verbatim.

Baselines (climatology / persistence) are not available yet: the manifest supports optional
baseline entries and flags their absence (`baselines_available: false`), so skill-vs-baseline
is not computed locally until a pack ships baselines.

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

Your model is scored on the intersection of its own forecast starts with the pack's starts.

## What it produces

Under `--output`:

- `scores.parquet` — the standard long-format per-start records (contracts.md §3.1): gridded
  RMSD (surface for a quick pack) + geostrophic currents, Class-4 RMSD, and the realism battery
  (spectra / activity / eddies).
- `scores-summary.json` — the aggregated means and 95% bootstrap CIs (contracts.md §3.4),
  produced by the same aggregation library as the hosted page.
- `scorecard/index.html` — a self-contained overlay scorecard.
- With `--artifacts viewer` or `all`, `viewer/` — the static viewer application,
  `data/your_model.zarr`, its viewer manifest, and a mixed `data/datasets.json`. The local
  descriptor uses relative URLs; official descriptors use absolute public MinIO URLs.

## The overlay scorecard

The scorecard reuses the no-ranking scorecard semantics of the website scores page
(`website-rebuild/scores/`): mean ± 95% CI over forecast starts, baselines pinned, neutral
order, no composite score. Your model is laid over the published challengers from `--published`
and highlighted as **"your model"**.

The overlay data is **inlined** into `index.html` (not fetched), and the renderer is a classic
script (not an ES module). This is deliberate: a page opened over `file://` cannot `fetch()`
sibling files nor load an ES module — both are blocked by the browser same-origin policy for
`file://`, which is exactly what the website's `app.js` relies on. Inlining lets you **open
`index.html` with a plain double-click, no server required**. The aggregation itself is the same
`oceanbench.publish.aggregate` code the hosted page uses, so the numbers match.

Published challengers are aggregated over the **same forecast starts** as your model, so if you
re-score a published challenger with `evaluate-local` its row coincides with your model's row
exactly. (With `--starts-limit N`, a quick-look mode, comparing your N-start means against the
full 52-start published means would show sampling noise, not the true agreement.)

## Flags

| flag | meaning |
|---|---|
| `--pack` | pack directory (required) |
| `--published` | published `scores.parquet` to overlay onto |
| `--published-challengers` | optional `challengers.json` for display names |
| `--artifacts scores\|viewer\|all` | build scores (default), only the viewer, or both |
| `--starts-limit N` | quick-look mode: score only the first N forecast starts (default: all) |
| `--with-lagrangian` | also compute the Lagrangian deviation (excluded by default; slow) |
| `--no-class4` | skip the Class-4 observation track (fast; gridded only) |
| `--no-realism` | skip the realism battery |
