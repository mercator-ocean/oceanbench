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
- 1024×1024 spatial tiles, one `(start_date, lead_day)` per chunk. Consolidated
  metadata. Root group carries the Copernicus Marine attribution + disclaimer
  (`contracts.md` §11).

## Tile size / fetch size (decided during rebuild)

The tile is sized so each chunk object is a single browser-friendly HTTP fetch.
The viewer reads one `(start_date, lead_day)` slice at a time and fetches every
spatial tile of the visible level, so per-fetch size = per-tile compressed size.
At the former 256-cell tile a native 1/12° tile was ~0.12 MB (uint16 barely
compresses on full-range ocean fields) and a 1/12° dataset-year emitted **~680k**
chunk objects — masses of tiny S3 objects, ~4 h to publish. The **1024-cell**
tile makes a full native tile ~2 MB raw (≈1–2 MB DEFLATE, ≤4 MB uncompressed
worst case: `2 × 1024² = 2 MB`, so even incompressible data stays under the
4 MB read-path ceiling), collapses coarse levels to a single tile, and cuts the
object count to **~62k (~11×)** with proportional publish-time savings. Measured
on a synthetic near-incompressible 1/12° field (1200×2400, 7 vars × 2 starts × 3
leads): objects 3252 → 606, whole-store size unchanged (308 → 309 MB), max data
tile 0.127 MB → 2.03 MB. Tile size is a `build_pyramid(tile_size=…)` parameter;
the reader is driven by each array's `.zarray` chunk grid, so no reader change is
needed. Zarr v3 sharding (many sub-chunks per object via byte-range reads) would
let tiles shrink again while keeping few objects, but needs a v3 writer (not in
zarr-python 2.x) and reader byte-range support — deferred.

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
writer), so the builder writes **zarr v2 with plain tile chunks**, sized at
1024 cells (see "Tile size / fetch size" above) so each object is a 1–2 MB
fetch and coarse levels stay a single tile. Object-count implication: level 0 of
a 1° pyramid (180×360, one tile) holds `variables × start_dates × lead_days ×
lon_tiles × lat_tiles` = `7 × 52 × 10 × 1 × 1 = 3,640` chunk objects per dataset;
a 1/12° pyramid across its 5 levels holds ~62k (vs ~680k at the old 256 tile).
The format is swappable behind the builder API: moving to v3 sharding changes
only the `to_zarr` writer and the compressor/codec wiring, not the layer/manifest
contract.

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
