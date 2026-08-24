<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# OceanBench fields explorer (viewer)

Static single-page app that reads the OceanBench viewer pyramids (contracts.md §6)
directly in the browser: no server-side rendering, no build step, no framework.
It is the comparison-first "fields explorer" half of the viewer: snapshot maps,
differences, animated currents and insight overlays of forecast fields. It carries
no ranking or score content (the score page is separate; bad-score cells deep-link
into this viewer).

## What it does

- **1 or 2 synchronized forecasts.** Each panel is `{dataset, variable}`; panels share
  one viewport (pan/zoom), lead day and start date. Same-variable panels share a fixed
  colorbar (§6). All view state lives in the URL hash (`layout`, `l`, `s`, `z`, `cx`,
  `cy`, `ov`, `region`, `dm`, per-panel `p0`/`p1`, and the rest listed under
  [URL state](#url-state)).
- **Two map scopes.**
  - *One date* is the default: the selected variable for one start date and lead day.
  - *Whole year* swaps the map for a precomputed error-geography raster over every start
    date, with `|error|` and signed `bias` metrics, plus an RMSE-by-start-date chart in
    the rail whose points drill back down into the one-date scope.
- **Three two-forecast displays.** *Side by side*, *Swipe* (one map, Forecast 1 left of a
  draggable divider and Forecast 2 right of it), and *Difference* (first-class A minus B
  on the diverging `balance` map centred at 0, coordinate-registered so different grids,
  GLONET 168 rows against references 170, align).
- **Currents as a variable.** Selecting *Currents* (surface or 15 m) draws a
  speed-magnitude background under windy-style advected particles over `uo`/`vo`, with
  fading trails and dark-theme glow. Play/pause and a speed multiplier live in the
  controls drawer; particle count adapts to the viewport.
- **Lead-day playback.** A play button and 0.5x/1x/2x speeds step the lead slider and
  loop; touching any other control pauses.
- **Insight overlays, one at a time**: eddy census (matched / spurious / missed contours
  against GLORYS or GLO12), Class-4 observation errors (points coloured by
  `|obs - model|`, read from the decimated match-up parquet with the vendored hyparquet
  and density-managed by zoom), illustrative trajectory fans seeded by clicking the map,
  and a water-column profile on click where a column store is published.
- **Context rail** for the active view: RMSE vs lead day with bootstrap CI band (from
  `scores-summary.json`, one series per reference), RMSE vs depth, RMSE by start date in
  the year scope, the clicked water-column profile, trajectory separation, and a live
  power spectrum computed in the browser from an explicit rectangle you drag on the map.
  Plain SVG, no chart library.
- **Light default theme** with a dark theme toggle, a hover readout in real units, and an
  about/glossary dialog.

## URL state

`writeHash` serializes the whole view and `readHash` parses it back, so any view can be
shared as a link:

| Key | Meaning |
| --- | --- |
| `layout` | 1 or 2 forecasts (older 4-panel links degrade to 2) |
| `s`, `l` | start-date index and lead day |
| `z`, `cx`, `cy` | zoom and normalized viewport centre |
| `p0`, `p1` | per-panel `dataset,variable,field` |
| `dm` | two-forecast display: `side`, `swipe`, `diff` |
| `scope`, `metric` | `year` scope and its `bias` metric (omitted at their defaults) |
| `ov`, `eref`, `col` | overlay mode, eddy reference, clicked water-column point |
| `region` | `global` or `ibi` |
| `psdOn`, `psd` | live-spectrum toggle and rectangle |
| `theme`, `play`, `spd` | theme, particle playback, particle speed |
| `rail`, `ctrl`, `rw`, `cw` | drawer collapse state and widths |

## Data path

The zarr reader (`modules/zarr.js`) is hand-rolled for our fixed layout: it reads
the consolidated `.zmetadata`, fetches the spatial tiles for one
`(start_date, lead_day)` slice, and decodes `uint16` to real units. Tiles are
**DEFLATE**-compressed and inflated natively with `DecompressionStream('deflate')`, with
a software inflater for browsers that lack it. Decoded chunks sit in an LRU cache bounded
by decoded bytes, and in-flight chunk requests are shared and refcounted so a fast scrub
never fetches the same tile twice. The fetch layer is driven by each array's `.zarray`
chunk grid and the manifest `levels` array, so multi-level pyramids work without code
change.

Insight artifacts (`modules/insights.js`) are read lazily and memoised: eddy/spectra
JSON by `fetch`, the Class-4 match-up parquet with the vendored **hyparquet** (MIT,
snappy codec) in a worker, with a prefetch of neighbouring lead days that is promoted
rather than refetched when the user arrives. Colormaps (`vendor/cmocean/`) are cmocean
LUTs (MIT), byte-packed and base64-encoded so nothing is fetched from a CDN.

## Running locally

The `data/` directory is generated (git-ignored). Populate it with viewer pyramids,
a `datasets.json`, and, for overlays and the rail, the insight artifacts:

```
data/
  <slug>.zarr  <slug>.viewer-manifest.json     # viewer pyramids (per dataset)
  datasets.json                                # [{slug,label,store,manifest}, …]
  insights.json                                # index: per (slug, region) URLs + region bounds
  scores-summary.json                          # aggregated mean ± CI (publish artifact)
  insights/<slug>/<region>/eddies.json         # eddy census (contracts.md §4)
  insights/<slug>/<region>/spectra.json        # realism PSD
  insights/<slug>/ibi/class4-matchups.parquet  # decimated Class-4 (snappy, for hyparquet)
```

`insights.json` maps `datasets[slug][region] → {eddies, spectra, class4_matchups,
rmsd_by_depth, year_error_geography, year_rmsd_by_start}` and carries `regions[id]`
lat/lon bounds and the `scores_summary` URL. The Class-4 parquet is a decimated copy of
the published match-up artifact (deterministic stride + snappy so hyparquet can read it
in-browser).

The website-integrated viewer defaults to the EDITO MinIO rebuild-preview data
base. To force local generated data while developing this static app, open it with
`?data_base=local` or set `window.OCEANBENCH_VIEWER_CONFIG.dataBaseUrl` before the
module script. Then serve and open:

```sh
python -m http.server -d website/viewer 8799
# open http://127.0.0.1:8799/?data_base=local
```

#### Overriding the data root without editing `config.js`

The data root is resolved at startup in this priority order:

1. `window.OCEANBENCH_VIEWER_CONFIG.dataBaseUrl`, set by an inline script before the
   module (unchanged escape hatch for the Quarto integration).
2. the `?data=` query parameter (`?data_base=` and `?dataBaseUrl=` remain accepted,
   and the value `local` still means `./data/`).
3. an optional `viewer-config.json` fetched from beside `index.html`; a 404 is the
   normal case and is ignored silently.
4. the built-in EDITO MinIO rebuild-preview prefix.

`viewer-config.example.json` shows the file's shape (`dataBaseUrl`, and optionally
`columnsBaseUrl` for the separately published `<slug>.columns.zarr` stores). Copy it
to `viewer-config.json` (git-ignored) to pin a root for a local checkout; an offline
`oceanbench view <dir>` mode would write the same file next to the copied viewer.

```sh
# both data roots exercised against the live, anonymously readable bucket
python -m http.server -d website/viewer 8765
# default (bucket): http://127.0.0.1:8765/index.html
# override:         http://127.0.0.1:8765/index.html?data=https://minio.dive.edito.eu/project-oceanbench/dev/benchmark/rebuild-preview/viewer/data/
```

In the Quarto website, `store`/`manifest`/insight URLs resolve against the EDITO
MinIO rebuild-preview viewer data prefix by default (CORS-enabled, §6).

### Developing the viewer inside the Quarto site (one command)

Quarto ships the viewer via `resources:` in `website/_quarto.yml`, but it only copies
resources into `_site/` on a full **render**: `quarto preview` never picks up viewer
edits on its own. Instead of manually copying changed files into `_site/viewer/`
(the old, cache-confusing workflow), run once per checkout:

```sh
website/scripts/dev-viewer-sync.sh
```

This replaces `website/_site/viewer/` with a symlink to `../viewer`, so every edit to
a viewer file is served live: no re-copy, no stale-cache guesswork. Then serve the
site as usual (`quarto preview`, or `python -m http.server -d website/_site`) and edit
viewer files freely.

`_site/` is a build output (git-ignored). `quarto render` wipes and regenerates it
with real file copies for production, so the symlink is a dev-only convenience: just
re-run the one-liner (it is idempotent) after any full render.

## File layout

- `app.js` is the application: panels, rendering, overlays, controls, rail and URL state.
- `state/` holds what the panels agree on: `view-modes.js` names the closed vocabularies
  (scope, display, overlay, year metric, region, eddy reference, theme), `shared-view.js`
  owns the shared view state and the setters that validate those values.
- `modules/` holds the pieces app.js reads rather than contains: the zarr reader, the
  insight loaders, the charts, the overlay draws, the Class-4 point index, the colour and
  raster helpers, grid arithmetic, readout formatting and the variable vocabulary.
- `styles/` is the stylesheet in load order: `shell.css` (theme, app shell, bars, drawers,
  controls), `map.css` (map, panels, strip, colour scale, legends), `rail.css` (the context
  rail and its charts) and `responsive.css` (viewport and container queries, last so it
  wins). `tokens.css` carries the design primitives and is loaded first.
- `qa/` is the harness. `vendor/` is third-party code, vendored.

## Shared with the scores page

`website/scores-summary.js` imports `config.js` and `modules/scores-data.js` from here.
Those two files are a cross-page contract; everything else under `modules/` is
viewer-private. Their module headers say so. `tokens.css` is shared the other way round:
the Quarto site loads it site-wide through `_quarto.yml`, so the design primitives are one
file rather than a viewer copy and a site copy.

## Verification

One harness, shipped, in `qa/`:

```sh
node qa/run.mjs
```

It boots a static server, drives the real page in headless Chromium under Playwright
across eight configurations, and then runs a behavioural pass that compares the drawn
pixels of twelve viewer states against frozen fingerprints. It checks page and console
errors, keyboard lead scrubbing, layout box sizes against `qa/expectations.json`, a
classified network budget for a warm back-scrub, em dashes in rendered text, the
fast-scrub render-token race, colour-range grow-only stability, particle liveness and
the software DEFLATE decode path. `qa/README.md` documents each probe, how to reseed the
two baselines, and what is deliberately not covered.
