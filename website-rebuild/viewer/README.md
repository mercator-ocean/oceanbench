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
  vendored hyparquet, density-managed by zoom), and a stubbed Lagrangian-trajectory
  loader (no artifact is produced yet — it reports that).
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
stride + snappy so hyparquet can read it in-browser). Then serve and open:

```sh
python -m http.server -d website-rebuild/viewer 8799
# open http://127.0.0.1:8799/
```

In production `store`/`manifest`/insight URLs point at the EDITO MinIO
`viewer/<year>/<slug>.zarr` and `insights/` artifacts (CORS-enabled, §6).

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
