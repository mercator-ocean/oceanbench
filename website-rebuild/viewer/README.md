<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# OceanBench fields explorer (viewer v1)

Static single-page app that reads the OceanBench viewer pyramids (contracts.md §6)
directly in the browser — no server-side rendering, no build step, no framework.
It is the "fields explorer" half of the viewer: snapshot maps and differences of
forecast fields. It carries no ranking or score content.

## What it does (v1)

- **Single equirectangular map panel** (2D, no globe): pixel = grid cell at native
  zoom, rendered from a selected `{dataset, variable, start_date, lead_day}`.
- **Difference mode**: pick a second dataset; renders A − B on the diverging
  `balance` colormap centred at 0. Datasets are registered on latitude/longitude,
  so pairs with different grids (GLONET 168 rows vs GLORYS/GLO12 170) align
  correctly.
- **Lead-day scrubber** (1…10) with prefetch of the adjacent leads; start-date
  selector.
- **Hover readout**: value in real units (via `scale_factor`/`add_offset`) with
  latitude/longitude.
- **Perceptually uniform colormaps** (cmocean thermal/haline/balance/speed +
  dense/delta), vendored as small LUTs. Compared views share a fixed colorbar with
  min/max labels.
- **Zoom / pan** (scroll + drag), reset control.
- **Cinematic dark** default theme with a light "publication" theme toggle.
- **Every view state is in the URL hash** (dataset, variable, start, lead, diff,
  colormap, zoom, pan, theme) — copy the URL to share the exact view.

Deferred to Phase 5b (see the pipeline report): GPU/canvas particle advection over
`uo/vo`; multi-panel synchronized comparison and blink/swipe; insight overlays
(eddy contours, Class-4 points, trajectories); small-multiples error-growth strip;
context rail curves.

## Data path

The zarr reader (`modules/zarr.js`) is hand-rolled for our own fixed layout — a few
hundred lines, no generic zarr client. It reads the consolidated `.zmetadata`,
fetches the spatial tiles for one `(start_date, lead_day)` slice, and decodes
`uint16` → real units. Tiles are **DEFLATE**-compressed and inflated natively with
`DecompressionStream('deflate')`, which is exactly why the pyramid builder writes
zlib rather than Blosc (see `docs/viewer-pyramids.md`). The fetch layer is driven by
each array's `.zarray` chunk grid and the manifest `levels` array, so multi-level
pyramids work without code change.

Colormaps (`vendor/cmocean/`) are cmocean LUTs (MIT), byte-packed and base64-encoded
so nothing is fetched from a CDN.

## Running locally

The `data/` directory is generated (git-ignored), like `scores/data/`. Populate it
with viewer pyramids and a `datasets.json`, then serve the directory:

```sh
# 1. Build pyramids with oceanbench.pyramids.build_pyramid (writes <slug>.zarr +
#    <slug>.viewer-manifest.json), then place or symlink them under data/:
mkdir -p website-rebuild/viewer/data
ln -s /path/to/glonet_1_degree.zarr                website-rebuild/viewer/data/
ln -s /path/to/glonet_1_degree.viewer-manifest.json website-rebuild/viewer/data/
#    …repeat for each dataset.

# 2. List them (copy the committed example and edit):
cp website-rebuild/viewer/datasets.example.json website-rebuild/viewer/data/datasets.json

# 3. Serve and open:
python -m http.server -d website-rebuild/viewer 8777
# open http://127.0.0.1:8777/
```

`datasets.json` is `{ "datasets": [ { slug, label, store, manifest }, … ] }` where
`store`/`manifest` are URLs (relative to the page or absolute). In production these
point at the EDITO MinIO `viewer/<year>/<slug>.zarr` artifacts (CORS-enabled, §6).

## Verification

`data path + colorization` and `DOM wiring` are checked headlessly (jsdom cannot
drive a real canvas): manifest load, chunk decode with a `uint16` scale/offset
round-trip within the quantization step, coordinate-aligned difference, symmetric
diverging range, URL-hash round-trip, and the element-id contract between
`index.html` and `app.js`. See the Phase 5 report for the harness and timings.
