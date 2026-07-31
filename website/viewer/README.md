<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# OceanBench fields explorer (viewer)

Static single-page app that reads the OceanBench viewer pyramids (contracts.md §6)
directly in the browser — no server-side rendering, no build step, no framework.
It is the comparison-first "fields explorer" half of the viewer: snapshot maps,
differences, animated currents and insight overlays of forecast fields. It carries
no ranking or score content (the score page is separate; bad-score cells deep-link
into this viewer).

## What it does

- **1 / 2 / 4 synchronized panels.** Each panel is `{dataset, variable, mode}`;
  panels share one viewport (pan/zoom), lead day and start date. Same-variable
  panels share a fixed colorbar (§6). All view state lives in the URL hash
  (`layout`, view, lead, start, overlay, region, per-panel `p0…p3`).
- **Three panel modes.**
  - *Field* — the variable on its perceptually uniform cmocean colormap.
  - *Difference* — first-class A − B on the diverging `balance` map centred at 0,
    coordinate-registered so different grids (GLONET 168 rows vs references 170)
    align.
  - *Currents* — windy-style GPU-of-canvas advected particles over `uo/vo` with
    fading trails and dark-theme glow, on a speed-magnitude background; play/pause
    and current-scaled speed. Particle count adapts to the viewport; structured so
    1/12° tiles drop in unchanged.
- **Blink / swipe A/B compare within a field panel** — drag the divider, or hold
  `B` to blink the whole panel between the two datasets.
- **Insight overlays with purpose-modes** (never all at once): eddy census
  (matched / spurious / missed contours vs GLORYS or GLO12), Class-4 obs points
  coloured by `|obs − model|` (read from the decimated match-up parquet with the
  vendored hyparquet, density-managed by zoom).
- **Context rail** — for the active view: the skill-vs-lead curve with bootstrap CI
  band (from `scores-summary.json`, one series per reference) and the realism PSD
  spectrum (challenger vs reference vs error power, from `spectra.json`). Plain SVG,
  no chart library.
- **Small-multiples error strip** — lead 1/3/5/7/10 mini-maps of A − B at a shared
  diverging scale, shown whenever the active panel has an A/B pair.
- **Cinematic dark** default theme with a light "publication" theme toggle; hover
  readout in real units.

## Data path

The zarr reader (`modules/zarr.js`) is hand-rolled for our fixed layout — it reads
the consolidated `.zmetadata`, fetches the spatial tiles for one
`(start_date, lead_day)` slice, and decodes `uint16` → real units. Tiles are
**DEFLATE**-compressed and inflated natively with `DecompressionStream('deflate')`.
The fetch layer is driven by each array's `.zarray` chunk grid and the manifest
`levels` array, so multi-level pyramids work without code change.

Insight artifacts (`modules/insights.js`) are read lazily and memoised: eddy/spectra
JSON by `fetch`, the Class-4 match-up parquet with the vendored **hyparquet** (MIT,
snappy codec). Colormaps (`vendor/cmocean/`) are cmocean LUTs (MIT), byte-packed and
base64-encoded so nothing is fetched from a CDN.

## Running locally

The `data/` directory is generated (git-ignored). Populate it with viewer pyramids,
a `datasets.json`, and — for overlays and the rail — the insight artifacts:

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

`insights.json` maps `datasets[slug][region] → {eddies, spectra, class4_matchups}`
and carries `regions[id]` lat/lon bounds and the `scores_summary` URL. The Class-4
parquet is a decimated copy of the published match-up artifact (deterministic
stride + snappy so hyparquet can read it in-browser).

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
resources into `_site/` on a full **render** — `quarto preview` never picks up viewer
edits on its own. Instead of manually copying changed files into `_site/viewer/`
(the old, cache-confusing workflow), run once per checkout:

```sh
website/scripts/dev-viewer-sync.sh
```

This replaces `website/_site/viewer/` with a symlink to `../viewer`, so every edit to
a viewer file is served live — no re-copy, no stale-cache guesswork. Then serve the
site as usual (`quarto preview`, or `python -m http.server -d website/_site`) and edit
viewer files freely.

`_site/` is a build output (git-ignored). `quarto render` wipes and regenerates it
with real file copies for production, so the symlink is a dev-only convenience — just
re-run the one-liner (it is idempotent) after any full render.

## Verification

Two harnesses (in the pipeline scratchpad, not shipped):

- `verify_viewer.mjs` — jsdom data-path checks: chunk decode `uint16` round-trip
  within the quantization step, coordinate-aligned difference, symmetric diverging
  range, URL-hash round-trip.
- `smoke_viewer.mjs` — real headless-Chromium smoke (Playwright): loads the page,
  asserts panels paint, scrubs leads, toggles particles (canvas pixels change frame
  to frame, ≈120 fps), switches 1→2→4 panels, renders a difference panel + error
  strip, enables eddy and Class-4 overlays (overlay-canvas pixels drawn), checks the
  rail SVGs, asserts the URL hash carries the full panel state, and requires **zero
  console errors**. It also dumps PNGs of each major feature. See the Phase 5b report.
