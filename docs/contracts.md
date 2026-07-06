<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# OceanBench — Architecture & Data Contracts

Status: DRAFT (agreed design, 2026-07-03). This document is the contract that all
the pipeline rebuild builds against. Changes here require discussion first.

## 1. Overview

OceanBench's next release is a benchmark for global ocean forecast models. It replaces the
notebook-centric pipeline (papermill execution, HTML-table score parsing,
hand-edited `index.json`, ephemeral staging) with a three-stage batch pipeline
whose only outputs are data artifacts, plus a static website that reads them.

```
ingest  (once per dataset-year)   → obs match-ups, viewer datasets,
                                     evaluation packs
score   (incremental)             → long-format score records + insight artifacts
publish (compaction)              → scores.parquet, catalog.json, static site data
```

Design principles:

- **Two scientific axes.** Primary skill evidence comes from the *observation
  track* (skill vs persistence and climatology at obs points, bootstrap CIs
  over the 52 weekly starts). The *realism battery* (PSD / effective
  resolution, error spectrum, activity ratio, eddy census) answers "is the
  model physically alive or a blurry RMSD-optimizer". Gridded RMSD vs
  GLORYS/GLO12 is a diagnostic layer.
- **No ranking.** OceanBench presents diagnostics; it does not crown a best
  model. The score page is a sortable scorecard with **no composite score and
  no default rank order** (neutral ordering, baselines pinned). Skill vs
  baselines with CIs is per-metric evidence, not a leaderboard. Each model
  additionally gets a plain-language summary card (non-expert reading level)
  above the expert table.
- **Native resolution everywhere.** Each challenger is scored on its native
  grid against the matching-resolution reference (status quo). No coarsened
  or common-grid scores. Cross-resolution honesty is provided by the obs track
  (grid-agnostic by construction) and the effective-resolution column.
- **Precompute the standardized battery; compute snapshots client-side.**
  Batch precomputes only (a) aggregates over the 52-start ensemble and
  (b) non-browser algorithms (eddy detection, Parcels advection, Class-4
  match-ups). Single-snapshot visuals (maps, differences, animation, box PSD)
  derive from the viewer zarr in the browser. No payload-file-per-figure.
- **Baselines are challengers** (`is_baseline: true`): climatology,
  persistence. Skill scores are *derived at aggregation/display time* from
  per-start records — never hardcoded in the scoring run.
- **Incremental by content address.** A score run is keyed by
  (challenger id+version, metric version, reference version, year, region).
  Unchanged keys are never recomputed.
- **Reference data is fetched live at scoring time, never mirrored.** The
  observation and gridded references are read directly from Copernicus Marine /
  the source buckets through the resilient chunk-fetch engine (below), backed by
  a **persistent local cache directory** — the existing stage mechanism, kept
  between runs instead of living in `$TMPDIR`. No reference mirror is built or
  published: reference access is internal plumbing, swappable without touching
  any product artifact. Copernicus stays the source of truth.
- **Reproducibility is preserved by stamping, not by pinning a mirror.** Every
  scoring run records the upstream product identifiers/versions and retrieval
  dates in its output metadata — `scores.parquet` rows already carry input
  versions, and run manifests carry retrieval dates. An upstream reprocessing
  triggers an **explicit, announced benchmark-wide re-score**, never silent
  drift; old and new scores coexist in `scores.parquet` (rows carry input
  versions).
- **Product storage (MinIO) holds only product artifacts** — scores, insights,
  viewer pyramids, evaluation packs; reference data is never among them.
  Evaluation packs remain published snapshots, stamped with the upstream
  versions they derive from and refreshed on an upstream bump. The
  **single-writer / atomic-publish** discipline (build directory → rename +
  manifest, read-only consumers) applies to these product artifacts — pyramids,
  packs, the catalog — not to reference reads.
- **Fetch engine = the PR #285 resilient chunk-fetch machinery**
  (`resilient-chunk-fetch` branch). Fetching is chunk-level with retries
  *during compute* (not only at dataset open), retriability is HTTP-status
  aware (retry 408/429/5xx, never a permanent 4xx), truncated bodies are
  rejected before caching, and each fetched chunk is written atomically to the
  persistent cache (pid+tid temp file + `os.replace`, no shared index,
  lock-free). The same engine protects both live scoring reads and cache
  warm-up. The pure-online mode (no cache) is retained as the fallback for
  local `evaluate` against remote data.

Out of scope for v1 (schema-ready only): ensemble metrics (CRPS, spread-skill).
NRT validation (branch 272) stays a separate dev branch; it will later become
another producer writing these same contracts, tagged obs-only.

## 2. Conventions

- Variable keys: CF standard names (`sea_surface_height_above_geoid`,
  `sea_water_potential_temperature`, `sea_water_salinity`,
  `eastward_sea_water_velocity`, `northward_sea_water_velocity`,
  `ocean_mixed_layer_thickness`, `geostrophic_*`). Display names/units live in
  ONE shared metadata table consumed by both the library and the website.
- `lead_day` is **1-based** in every artifact, including zarr coords. (The
  legacy 0-based `lead_day_index` does not appear in any next-release artifact.) A
  metric MAY legitimately begin later than day 1 (lagrangian deviation starts
  at lead day 2) and a challenger MAY have a shorter horizon (langya: 7 days).
  Consumers MUST NOT assume lead day 1 exists or a fixed 10-day horizon.
- Depth labels are machine keys: `surface`, `50m`, `100m`, `200m`, `300m`,
  `500m`; Class-4 bins `0-5m`, `5-100m`, `100-300m`, `300-600m`, `15m`.
- Dates ISO-8601. All JSON written `sort_keys=True`. Floats: `null` for NaN.
- Calibration constants carried forward verbatim (validated, do not re-derive):
  GLO12/global SSH→SLA shift `-0.1148` (GLO12 MDT); IBI shift `-0.0674`
  (IBYRIS MDT); climatology baseline shift `-0.1329`; MLD threshold
  0.03 kg/m³ capped at 600 m; velocity obs target depth 15.0 m.

### Regions (v1)

| id | bounds (lat, lon) | used for |
|---|---|---|
| `global` | — | all metrics |
| `ibi` | 26.17–56.08 N, −19.08–5.08 E | all metrics |
| `gulfstream` | 30–45 N, −80 – −50 E | realism battery only |
| `kuroshio` | 25–45 N, 130–165 E | realism battery only |

### Challenger registry

`challengers.json` (in-repo, versioned): canonical slug → metadata.

```json
{
  "glonet":      {"display_name": "GLONET",  "organization": "Mercator",
                  "nominal_resolution_deg": 0.25, "is_baseline": false,
                  "lead_days": 10, "source": "s3://.../ml-forecast-outputs/glonet/"},
  "climatology": {"display_name": "Climatology", "is_baseline": true, "...": "..."}
}
```

One slug per challenger everywhere (paths, parquet, catalog). The NRT
`octo-<name>-p1d` ids map onto these slugs when NRT integrates.

## 3. Score contract

### 3.1 `scores.parquet` (primary product, whole benchmark)

One row per (challenger, year, region, metric, …, lead_day, start_date).
Per-start values are the norm — aggregation (means, bootstrap CIs, skill vs
baselines) happens downstream and is recomputable against any baseline.

| column | type | notes |
|---|---|---|
| `challenger` | str | registry slug |
| `challenger_version` | str | forecast dataset version/tag |
| `year` | int32 | evaluation year (2024 at launch) |
| `region` | str | region id |
| `metric` | str | see 3.2 |
| `reference` | str? | `glorys` \| `glo12` \| `observations` \| null |
| `variable` | str? | CF standard name, null for non-variable metrics |
| `depth` | str? | depth label or bin, null if not applicable |
| `lead_day` | int8 | 1-based |
| `start_date` | date | forecast initialization date |
| `band` | str? | spectral band (`large`, `regional`, `mesoscale`), else null |
| `polarity` | str? | `cyclone` \| `anticyclone`, else null |
| `value` | float64 | |
| `unit` | str | |
| `n` | int32? | sample size (e.g. obs count in a Class-4 cell) |
| `oceanbench_version` | str | scoring code version |

### 3.2 Metric keys (v1)

Gridded (vs `glorys` and `glo12`, area-weighted cos-lat, native grid):
`rmsd` (per variable×depth), `rmsd` with `variable=ocean_mixed_layer_thickness`,
`rmsd` with geostrophic variables, `lagrangian_deviation_km`.

Observation track (vs `observations`): `class4_rmsd` (T/S bins, SLA surface,
currents 15 m).

Realism battery (native grid, per region incl. WBC boxes):
`psd_band_energy_fraction` (band column set), `effective_resolution_km`
(wavelength where challenger PSD falls to half of reference),
`error_spectrum_band_energy` (PSD of challenger−reference, band column set),
`activity_ratio` (challenger anomaly std / reference anomaly std),
`eddy_count`, `eddy_hit_rate`, `eddy_miss_rate`, `eddy_mean_displacement_km`
(polarity column set).

The eddy census detects SSH-anomaly extrema using physical, grid-independent
parameters. Gaussian background and detection scales are specified in kilometres
and converted to latitude/longitude cell sigmas from the grid spacing (longitude
spacing is scaled by the cosine of the domain-mean latitude). Candidate peaks are
separated by great-circle distance. Closed contours are accepted by spherical
cell area in km² and convexity; matching also uses great-circle distance. The
default parameter set is:

| parameter | default |
|---|---:|
| `background_sigma_km` | 1334.3391 km |
| `detection_sigma_km` | 166.7924 km |
| `min_peak_separation_km` | 889.5594 km |
| `amplitude_threshold_meters` | 0.04 m |
| `contour_level_step_meters` | 0.01 m |
| `min_eddy_area_km2` | 197,828.9874 km² |
| `max_eddy_area_km2` | 74,185,870.2689 km² |
| `min_contour_convexity` | 0.75 |
| `max_match_distance_km` | 200 km |

These physical defaults calibrate to the former 12, 1.5 and 8 cell scales and
16–6000 cell contour range on an equator-centred 1° grid. Contour filtering is on
by default (`apply_eddy_contour_filtering=true`): a detected peak is retained only
if it anchors a closed SSH-anomaly contour whose spherical cell area falls within
the `min_eddy_area_km2`–`max_eddy_area_km2` bounds and whose convexity (region
solidity) is at least `min_contour_convexity` (0.75). This closed-contour
definition is applied consistently to the eddy metrics, matching and the census.
`apply_eddy_contour_filtering=false` recovers the raw-peak census that reproduces
the already-published `glonet_1_degree` artifact.

Each eddy census reference entry includes a `parameters` object containing the
complete parameter set, the resolved contour-filter switch and the OceanBench code
version, so filtered and raw-peak artifacts remain distinguishable.

Reserved for later (schema needs no change): `crps`, `spread`, `spread_skill_ratio`.

### 3.3 Per-run increment

Each scoring run writes
`runs/<challenger>/<year>/<region>/scores-<content_hash>.parquet` (same schema).
`publish` compacts all runs into the single public `scores.parquet`.
A small per-challenger `scores.json` (aggregated means only, nested legacy
`ModelScore` shape) is emitted by an adapter for transition-period
compatibility with the existing website; it is deprecated from day one.

### 3.4 Derived at display/aggregation time (never stored per-run)

- mean over starts, bootstrap CI (resample the 52 starts, 1000 draws, 95%),
- skill score `1 − RMSD_model / RMSD_baseline` vs `persistence` and
  `climatology`, with CI via paired bootstrap.

## 4. Insight artifacts

Per (challenger, year, region), under `insights/`, referenced by a
`manifest.json` mapping semantic key → `{kind, schema_version, url, bytes}`.
Blobs are content-hash named (immutable, CDN-cacheable).

| kind | file | content |
|---|---|---|
| `aggregate-map` | small zarr (or webp+json meta) | time-mean bias AND rmse per variable, leads {1,5,10}, surface |
| `spectra` | JSON | per variable×region×lead {1,5,10}: wavelength[], challenger_power[], reference_power[], error_power[] |
| `eddies` | JSON | per lead: matches (with displacement km), spurious, missed; contour polygons point-limited |
| `trajectories` | JSON | Parcels particle trajectories (challenger vs reference), decimated |
| `class4-matchups` | parquet | one row per obs point: obs value, model value, lat, lon, depth, time, variable, lead_day |

Schemas for `spectra` and `eddies` are adapted from branch 249's payload
formats (proven shapes) with `kind` + `schema_version` fields added and the
widget/XHR coupling removed.

TODO pipeline: `class4-matchups` manifest entries may add
`row_group_index: {by_variable: {<variable>: [row_group_index, ...]}}`. The
viewer opportunistically uses this to scatter samples by variable; without it,
large global parquet overlays sample evenly spaced row groups across the file.

## 5. Catalog

`catalog.json` at the artifact root — **generated by `publish`, never
hand-edited**:

```json
{
  "schema_version": "2.0",
  "generated_at": "…",
  "scores_url": ".../scores.parquet",
  "releases": {
    "2.0.0": {
      "years": {
        "2024": {
          "regions": {
            "global": {
              "glonet": {
                "insights_manifest_url": "…",
                "viewer_zarr_url": "…"
} } } } } } }
}
```

## 6. Viewer datasets (display copies — scoring never reads these)

One multiscale zarr **pyramid** per (challenger|reference|baseline, year),
produced by `ingest`. Viewer data is always **numeric** (never PNG/WebP) so the
client can read values on hover, difference any two datasets, recolor, and
compute box-PSD — images survive only as static thumbnails in insight
artifacts.

### Storage

- **Base level = native grid** (1/12°, 1/4°, or 1° per dataset). Users zoom to
  full model resolution everywhere; no regional-box subsetting.
- Pyramid levels halve resolution from native up to ~1° (e.g. 1/12° → 4–5
  levels), each level a zarr group `level/<k>` with dims
  `(start_date: 52, lead_day: 10, latitude, longitude)`.
- Variables: `zos, thetao, so, uo, vo` at surface + `uo, vo` at 15 m.
- Encoding: uint16 with per-variable `scale_factor`/`add_offset` attrs
  (precision-matched: ≪ model error at all variables), zstd compression,
  explicit `_FillValue` for land.
- Chunking: **1024×1024 spatial tiles**, one (start_date, lead_day) per chunk.
  The tile is sized so each chunk object is a browser-friendly HTTP fetch: a
  native 1/12° tile is ~2 MB raw (≈1–2 MB DEFLATE, ≤4 MB uncompressed worst
  case), hitting the ≤4 MB read-path target below, and coarse levels collapse to
  a single tile. This cuts a 1/12° dataset-year from ~680k tiny objects (the old
  256-cell tile, ~0.12 MB/fetch) to ~62k objects (~11×), with proportional S3
  publish-time savings. Eventual zarr v3 **sharding** (one shard per
  (start_date, lead_day, level), tiles fetched by HTTP range read) is swappable
  behind the builder once a v3 writer ships; until then the running zarr-python
  2.x writes plain v2 tile chunks. The viewer reader
  (`website/viewer/modules/zarr.js`) is driven by each array's `.zarray` chunk
  grid, so the tile size is a builder parameter with no reader change.
- Consolidated metadata; identical layout for references (GLORYS, GLO12) and
  baselines so any pair can be differenced client-side.
- Volume: ~40–50 GB per 1/12° dataset-year compressed (×1.33 pyramid
  overhead included); ~0.5 TB per benchmark year across all datasets.
  Write-once, content-addressed paths → immutable/CDN-cacheable.

### Read path (what "fluid" means, testable)

- Client (static SPA: maplibre or deck.gl + zarrita) fetches only viewport-
  visible tiles at the zoom-matched pyramid level: **≤ ~4 MB per displayed
  layer at any zoom**, target < 500 ms to first paint on a warm cache.
- WebGL rendering: colormap in shader; **difference mode** = two tile
  sources subtracted in-shader (growing-error view at native resolution);
  animated currents = GPU particle advection over `uo/vo` tiles.
- Time scrubbing prefetches adjacent lead days for the active layer.
- Overlays from insight artifacts: eddy contours per lead day, Lagrangian
  trajectories (with divergence vs reference), Class-4 obs locations/errors.
- Per-dataset `viewer-manifest.json`: levels, tile size, bounds, variables
  with units/scale/offset/default colormap+range, start_dates, lead_days.

### Viewer UX contract

- **Default theme: cinematic dark** (dark ocean canvas, glowing GPU particle
  flows, luminous colormaps). A light "publication" theme is available for
  exporting paper-ready figures.
- **2D equirectangular maps, no globe.** Pixel = grid cell at native zoom.
  (Optional polar projection is a post-v1 add-on.)
- **Comparison is the primitive.** 1/2/4 synchronized panels (linked viewport
  + lead day); panel = {dataset, variable, start_date, lead_day}; first-class
  **difference panels** (A − B in shader, diverging colormap centered at 0);
  blink/swipe toggle within a panel.
- **Small multiples for error growth** (lead 1/3/5/7/10 strip, shared
  colorbar); **animation only for motion** (GPU current particles, Lagrangian
  divergence trails). Never animate error maps.
- **Context rail**: quantitative curves for the current view (RMSE vs lead,
  regional PSD, score rows) linked to the map state.
- **Overlays with purpose-modes**, toggleable, never all at once: eddies
  (matched/spurious/missed contours), Class-4 obs points colored by error,
  particle trajectories. Default state = one panel, one field.
- **Colors**: perceptually uniform only (cmocean: thermal/haline/balance…);
  compared panels always share a fixed colorbar.
- **Every view state is a URL** (panels, viewport, lead, overlays). The score
  page deep-links into the viewer (bad score cell → preconfigured difference
  view). Artifact contracts above already provide everything these features
  read; no additional server-side capability is implied.

### Infra prerequisites (Phase 0 checks, blocking for viewer work)

- **RESOLVED (2026-07-03).** Benchmark ran against both candidates, EDITO
  MinIO and CloudFerro S3. They tie on throughput (50-way concurrent 256 KB
  range reads clean on both, no throttling), but EDITO MinIO already serves
  `Access-Control-Allow-Origin: *` with a working OPTIONS preflight, exposes
  `Accept-Ranges`/`Content-Range`/`ETag`, and speaks HTTP/2; CloudFerro
  returns no CORS headers and 403 on preflight. **Decision: EDITO MinIO
  serves all browser-facing artifacts.** CloudFerro remains a server-side
  ingest source only. CDN deferred (origin is not the bottleneck); the data
  contract is unchanged if one is added later.

## 7. Evaluation packs (local `oceanbench evaluate`)

Downloadable, versioned bundles produced by `ingest`:

- `pack-quick-<year>`: obs match-up inputs + 1/4° surface reference fields +
  climatology/persistence baselines. Target: evaluate a model locally in
  minutes–1 h; answers "is it good" + "is it blurry".
- `pack-full-<year>`: adds multi-depth gridded references for the official
  gridded track.

`oceanbench evaluate ./forecasts/ --pack ./pack-quick-2024`
produces the same
artifacts as the hosted run plus a local HTML scorecard overlaying the user's
model on the published `scores.parquet`. The required `year` and `region` fields
in `pack-manifest.json` define the evaluation context; the command never accepts
independent overrides for them.

With `--artifacts all`, it also produces a local viewer
pyramid and static viewer directory. Its `datasets.json` combines the local
challenger (relative store/manifest URLs) with official products (absolute public
MinIO URLs), so official pyramids do not need to be downloaded.

## 8. S3 layout

```
s3://project-oceanbench/dev/benchmark/<release>/
  catalog.json
  scores.parquet
  scores-summary.json                  (precomputed aggregate for the score page)
  challengers.json                     (copy of registry at publish time)
  <year>/<region>/<challenger>/
    runs/scores-<hash>.parquet
    insights/manifest.json
    insights/<content-hash blobs>
  viewer/<year>/<slug>.zarr            (challengers, references, baselines)
  packs/pack-{quick,full}-<year>/
```

Artifacts publish under the **dev prefix** `dev/benchmark/<release>/` on the
`project-oceanbench` bucket, uploaded by `oceanbench publish-s3` (see
`oceanbench/publish/s3.py`). The endpoint is EDITO MinIO,
`https://minio.dive.edito.eu`. Anonymous (public) read on the dev prefix is
enabled manually by a maintainer through the EDITO console — the publish step
never touches bucket policy. CORS is already configured bucket-wide, so browser
range GETs against the published tree work without any per-publish setup. The
current site and `public/evaluation-reports/` remain untouched until parity
(see Phase gates). The earlier `benchmark-dev/` dev prefix is retired.

## 9. Port vs rebuild inventory

**Port (validated science — do not rewrite):** `core/rmsd.py`,
`core/classIV*.py` (incl. SLA shifts), `core/mixed_layer_depth.py`,
`core/geostrophic_currents.py`, `core/lagrangian_trajectory.py`,
`core/regions.py`, reference/challenger URL registry, and from branch 249:
`core/psd.py`, `core/eddies.py` (plus their tests). Cherry-pick later from
272: IBI `-0.0674` constant, `class4_drifters.py`. Ingest network engine from
`origin/resilient-chunk-fetch` (PR #285): `core/remote_http.py` (resilient
zarr chunk mapper + HTTP-status-aware retriability) and the content-keyed
`core/computed_dataset_cache.py` (atomic single-writer store for computed,
non-plain-zarr datasets) — both with their tests (`tests/test_remote_http.py`,
`tests/test_computed_dataset_cache.py`). PR #285 is not rebased onto main; its
engine is cherry-picked into the rebuild and the PR is closed with credit at cutover.

**Rebuild:** CLI/runner (no papermill), ingest stage (forecast-at-obs
extraction, viewer pyramids, packs — reference reads stay live through the
resilient engine + persistent cache, never a mirror), publish/compaction,
website score page
(reads `scores.parquet`; DuckDB-WASM or plain JS), viewer (static SPA,
zarr range reads).

**Delete:** notebook templates, `python2jupyter`, papermill wiring,
`widget_assets.py` XHR coupling, `notebook_score_parser.py`,
hand-maintained `index.json` flow.

## 10. Phase plan & gates

- **Phase 0 — foundations.** Long-lived `pipeline-rebuild` branch in this repo; this doc
  merged; JSON Schemas for catalog/manifest/insight payloads; storage
  benchmark MinIO vs CloudFerro (§6); Copernicus Marine redistribution-terms
  check for evaluation packs; parity harness capturing current published 2024
  scores as golden data.
  Status (2026-07-03): storage check DONE (EDITO MinIO, §6); licensing DONE
  (§11); parity goldens DONE (9,810 rows — 10 challengers × ≤2 regions ×
  9 metrics, published version 0.2.1, `tests/parity/`). Caveat: published-
  report provenance vs main-tip code (#298 area-weighted RMSD reapply) must
  be verified as Phase 1's opening task before the parity gate is trusted.
- **Phase 1 — score runner.** Port metrics, emit long-format per-start
  records. **Gate: reproduces published 2024 scores for glonet (1/4°) and one
  1/12° challenger within numerical tolerance.**
- **Phase 2 — ingest.** Live reference fetch through the resilient chunk-fetch
  engine backed by the persistent cache (upstream-version stamping in run
  metadata), obs match-up extraction, viewer zarrs, packs. **Gate: full
  re-score of one 1/12° challenger in << 24 h; incremental re-run is a no-op.**
- **Phase 3 — publish + score page + local evaluate.** Compaction,
  catalog, static score page with skill-vs-baseline + CIs, overlay scorecard.
- **Phase 4 — realism battery.** Port 249 PSD/eddies; error spectrum,
  activity ratio, effective resolution; WBC regions; insight artifacts.
  (3 and 4 parallelize.)
- **Phase 5 — viewer v1.** Battery browser + snapshot maps/differences/
  current animation from viewer zarr.
- **Phase 6 — viewer v2.** Hand-drawn box PSD, free-form exploration.
- **Cutover:** parallel run under `dev/benchmark/<release>/` until the score page
  matches published numbers; then switch. Then: 2023/2025 ingest (with branch 241
  coordination), ensembles, NRT integration.

## 11. Attribution & licensing (Copernicus Marine)

Redistribution of CMEMS-derived packs and viewer copies is permitted under
the Copernicus Marine Service License, which grants redistribution and
value-added-product rights (worldwide, royalty-free, perpetual).

OceanBench policy: use the **derived-work credit everywhere** (a strict
superset of the plain-redistribution credit — never wrong):

> Generated using E.U. Copernicus Marine Service Information;
> https://doi.org/10.48670/moi-00021 ; https://doi.org/10.48670/moi-00016

(GLORYS12 = `GLOBAL_MULTIYEAR_PHY_001_030`, GLO12 =
`GLOBAL_ANALYSISFORECAST_PHY_001_024`.)

The credit plus the standard CMEMS liability/no-warranty disclaimer MUST
appear in:

- zarr attrs of packs and viewer pyramids,
- pack READMEs,
- the viewer footer,
- a website data-provenance page.

Derivatives are labeled OceanBench-generated and never presented as the
authoritative Copernicus product. No notification to Mercator Ocean is
required.
