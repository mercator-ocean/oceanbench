<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Viewer pyramid builder

`oceanbench/pyramids/` builds one multiscale zarr pyramid per (challenger |
reference | baseline, year) as specified in `contracts.md` §6, plus the
per-dataset `viewer-manifest.json` (validated against
`schemas/viewer-manifest.schema.json`).

## Layout

- Groups `level/<k>`: level 0 is the native grid, each further level halves the
  spatial resolution (2×2 ocean-cell mean) up to about one degree. A dataset
  already at ~1° collapses to a single `level/0` (the degenerate case).
- Dims `(start_date, lead_day, latitude, longitude)`; `lead_day` is 1-based.
- Variables: `zos, thetao, so, uo, vo` at the surface and `uo, vo` at 15 m.
- Each variable is `uint16` with per-variable `scale_factor`/`add_offset`
  (chosen from the native data range with margin; quantization step ≪ model
  error), explicit `_FillValue` (`65535`) for land, **DEFLATE (zlib)**
  compression (see "Browser-decodable codec" below).
- 256×256 spatial tiles, one `(start_date, lead_day)` per chunk. Consolidated
  metadata. Root group carries the Copernicus Marine attribution + disclaimer
  (`contracts.md` §11).

## API

```python
from oceanbench.pyramids import build_pyramid, viewer_layers

layers, specs = viewer_layers(challenger_dataset)   # extract surface + 15 m layers
result = build_pyramid(layers, specs, output_path=".../glonet_1_degree.zarr",
                       dataset_slug="glonet_1_degree", year=2024)
```

`build_pyramid` validates the manifest against the schema and refuses to emit an
invalid one. The layer extraction (`viewer_layers`) is separate from the builder
so pyramid tiling logic is unit-tested on synthetic multi-level grids even when
the real 1° datasets are single-level.

## Browser-decodable codec (decided Phase 5)

The viewer reads chunks directly in the browser, so the tile codec must be one the
platform decodes without shipping a heavy wasm decoder. Blosc/zstd (the zarr
default) is not: it would force a wasm blosc build into every page. The builder
therefore compresses every array — data tiles **and** coordinate arrays — with
**DEFLATE (`numcodecs.Zlib`)**, which the browser inflates natively via
`DecompressionStream('deflate')` (verified byte-exact round-trip). xarray/zarr
writes zlib natively, so this is a one-line compressor swap, not a format change.

Size impact is negligible. On the full-range quantized `uint16` fields the
byte-shuffled Blosc/zstd advantage nearly vanishes; measured on a real SSH tile,
zlib was even fractionally smaller (59.4 KB vs 60.8 KB), and whole-store sizes
moved by about 2 %:

| dataset | Blosc/zstd | DEFLATE (zlib) |
|---|---|---|
| `glonet_1_degree` | 292 MB | 299 MB |
| `glorys_one_degree` | 295 MB | 303 MB |
| `glo12_one_degree` | 293 MB | 303 MB |

Compression stays swappable behind the builder's `_compressor()`; a future zarr v3
sharding writer can revisit the codec without touching the layer/manifest contract.

## Zarr v2 vs v3 (flagged)

The contract calls for zarr **v3 with sharding** (one shard per
`(start_date, lead_day, level)`) so tiles are HTTP range reads without millions
of small objects. The environment ships **zarr-python 2.18** (no v3 sharding
writer), so the builder writes **zarr v2 with plain tile chunks**. Object-count
implication: level 0 of a 1° pyramid holds `variables × start_dates × lead_days ×
lon_tiles × lat_tiles` = `7 × 52 × 10 × 2 × 1 = 7,280` chunk objects per dataset
(more for finer, multi-level datasets). The format is swappable behind the
builder API: moving to v3 sharding changes only the `to_zarr` writer and the
compressor/codec wiring, not the layer/manifest contract.

## Real 1° pyramids (parity data, 2024)

Built to scratchpad from the warm stage cache (single level each, 7 variables):

| dataset | grid | size (DEFLATE) | build time |
|---|---|---|---|
| `glonet_1_degree` | 168×360 | ~299 MB | ~15 s |
| `glorys_one_degree` | 170×360 | ~303 MB | ~17 s |
| `glo12_one_degree` | 170×360 | ~303 MB | ~15 s |

Note the two grids are offset: GLONET spans 168 latitudes (−77.5…89.5°), GLORYS and
GLO12 span 170 (−79.5…90.5°). They share 1° spacing but not an origin, so the
viewer registers difference pairs on coordinates, never on raw array index.
