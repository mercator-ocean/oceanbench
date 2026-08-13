<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Class-4 serving layout

How the Class-4 match-ups are shaped for the viewer, why the current published
vintage costs 15 to 22 MB before the first pixel, and what the next regen writes
instead. Companion to `contracts.md` §4 and §6.

Everything here is already implemented in
`oceanbench/publish/viewer_artifacts.py` and
`oceanbench/publish/class4_overlays.py`; the next regen produces it with no
further code change. The viewer changes described in
[Viewer wave](#viewer-wave-what-changes-later) are **not** done and are
specified here for a later wave.

## 1. What is published today, measured

Measured on 2026-08-13 by range-reading the parquet footers of every
`(dataset, region)` in `dev/benchmark/rebuild-preview` on CloudFerro. No file
was downloaded.

| dataset / region | file | footer | row groups | rows | columns | codec |
|---|---|---|---|---|---|---|
| glo12 / global | 4.38 GB | 5.98 MB | 6 760 | 306 374 266 | 8 | ZSTD |
| glo12 / ibi | 0.07 GB | 5.25 MB | 6 240 | 5 509 010 | 8 | ZSTD |
| langya / global | 3.03 GB | 4.19 MB | 4 732 | 214 407 247 | 8 | ZSTD |
| wenhai / global | 6.45 GB | 6.69 MB | 6 760 | 306 374 266 | 9 | SNAPPY |
| xihe / global | 6.46 GB | 6.69 MB | 6 760 | 306 374 266 | 9 | SNAPPY |

Two vintages are live. The old one (`wenhai/global`, `xihe/global`) is SNAPPY
with nine columns including the stored `abs_error`; every other file is the new
ZSTD eight-column one. Both share the same row-group layout.

### Where the 6 MB footer comes from

The old writer emitted **one row group per
`(start_date, lead_day, variable, depth_bin)` block**. On a global year that is
52 starts x 10 leads x 13 blocks = **6 760 row groups**, and the footer carries
one `ColumnChunk` structure per row group per column: 6 760 x 8 = **54 080
column chunks at 110.6 bytes each**.

Attributed by writing the same synthetic table at three row-group counts:

| statistics written | bytes per column chunk |
|---|---|
| all 8 columns | 112 |
| 4 grouping columns only | 87 |
| none | 80 |

So the footer is **linear in the column-chunk count** and roughly 71 % fixed
per-chunk metadata (path, offsets, encodings, sizes) with 29 % statistics. The
row-group count is the lever; dropping statistics alone saves a fifth.

Note `glo12/ibi`: 5.25 MB of footer on a 70 MB file. The footer is **7 % of the
file** and is paid in full on every open.

### Where the rest of the first load goes

Per `(start_date, lead_day)` on glo12/global: 586 807 rows (p50), 9.15 MB
uncompressed, about 8.4 MB compressed across its 13 blocks. Per selection
`(start, lead, variable, depth_bin)`:

| variable | depth bin | rows (p50) | compressed (p50) |
|---|---|---|---|
| sea_surface_height_above_geoid | surface | 313 858 | 5 818 KB |
| eastward_sea_water_velocity | 15m | 27 690 | 452 KB |
| northward_sea_water_velocity | 15m | 27 690 | 447 KB |
| sea_water_potential_temperature | 300-600m | 52 608 | 439 KB |
| sea_water_potential_temperature | 100-300m | 41 249 | 374 KB |
| sea_water_salinity | 300-600m | 44 774 | 317 KB |
| sea_water_salinity | 100-300m | 35 929 | 274 KB |
| sea_water_potential_temperature | 5-100m | 22 804 | 211 KB |
| sea_water_salinity | 5-100m | 19 881 | 156 KB |
| sea_water_potential_temperature | surface | 2 952 | 55 KB |
| sea_water_potential_temperature | 0-5m | 1 290 | 20 KB |
| sea_water_salinity | 0-5m | 1 085 | 14 KB |

6 MB of footer plus a 5.8 MB SSH selection is the 15 to 22 MB first load, and
the footer is paid again for every dataset the user compares against.

Column weight over the whole file (glo12/global, compressed): `model_value`
1 686 MB, `longitude` 1 082 MB, `latitude` 1 025 MB, `observation_value`
580 MB, the four key columns 2.9 MB together. Dictionary-encoded key columns are
free; the float payload is the file.

## 2. Restructured match-up parquet

Same eight columns, same dtypes, same sort order
`(start_date, lead_day, variable, depth_bin)`, same ZSTD level 3, same
provenance metadata. Two changes:

**Row groups pack whole blocks of one pair.** Consecutive
`(variable, depth_bin)` blocks of the same `(start_date, lead_day)` accumulate
into one row group until it reaches `TARGET_ROW_GROUP_ROWS` (400 000). A pair
boundary always closes the group. A single block above
`MAXIMUM_ROW_GROUP_ROWS` (1 000 000) is split across consecutive groups.

On the measured global block sizes this yields **2 row groups per pair** (the
first closes after the SSH block pushes past the target, the second holds the
T/S tail), so **1 040 row groups instead of 6 760**. On IBI the whole pair is
10 600 rows, so **1 row group per pair, 520 total**.

**Statistics only on the four grouping columns.** No reader has ever filtered
on a latitude or a model value.

Predicted footers at 87 bytes per column chunk:

| file | row groups | predicted footer | today |
|---|---|---|---|
| global | 1 040 | ~0.72 MB | 5.98 MB |
| ibi | 520 | ~0.36 MB | 5.25 MB |

`test_packing_shrinks_the_footer_by_the_block_count_and_stays_under_the_budget`
measures the per-row-group footer cost on a fixture and asserts that
1 040 row groups stay under 1 MB, so the budget is checked rather than assumed.

**The invariant the viewer detects on is preserved.** A row group still holds
exactly one `(start_date, lead_day)` pair, which is precisely what
`allRowGroupsSinglePair` in `class4-worker.js` tests. `verify_matchup_parquet`
enforces it. What changes is that a group now spans several
`(variable, depth_bin)` blocks: the worker's `rowGroupHasVariable` already keeps
groups whose `variable` statistics straddle the request, and `app.js` already
drops rows whose `variable` does not match, so **the existing viewer reads the
new file unmodified**. The two pipeline readers of the parquet
(`_write_year_artifacts`, `rmsd_by_depth`) already select by
`(variable, depth_bin)` inside each group.

The cost is honest: an unmodified viewer that asks for one variable of one pair
now transfers that pair's packed group (about 6.7 MB for the SSH group, 1.8 MB
for the T/S group) rather than the block alone. That is why the overlay extracts
below exist, and why the viewer should stop using the parquet for painting.

## 3. Overlay extracts (new)

The parquet is the analysis artifact. Painting the scatter overlay needs four
arrays for one selection and nothing else, so the pipeline now writes one small
file per `(variable, depth_bin, start_date, lead_day)`.

### Format: `.obx`, packed binary

Chosen over the alternatives on measured size at the same point count:

- **tiny parquet** — float32 columns compress to about 3.5 bytes per value in
  the published file, so 50 000 points is ~700 KB, plus a footer and a second
  parquet reader path for a file that needs no predicate pushdown at all.
- **zstd JSON** — the browser cannot decompress zstd natively
  (`DecompressionStream` is gzip/deflate only) and the viewer's vendored fzstd
  would have to be pulled in for every overlay; uncompressed JSON of 50 000
  points is megabytes.
- **packed binary** — 4 arrays x uint16 = **8 bytes per point**, decoded with
  three lines of `DataView`/typed-array code, no dependency, no compression to
  undo.

Layout, little-endian throughout:

```
"OBX1"                     4 bytes, magic
uint32 header_length       length of the JSON header including its padding
JSON header                UTF-8, zero-padded to an 8-byte boundary
uint16[displayed_count]    latitude
uint16[displayed_count]    longitude
uint16[displayed_count]    observation_value
uint16[displayed_count]    model_value
```

Values are quantized with the pyramid's existing convention
(`oceanbench/pyramids/quantization.py`): `value = code * scale_factor +
add_offset`, the range taken per column per extract with a 1 % margin over 65 534
levels. On a global extract that is about 0.003 degrees of latitude (~300 m) and
0.0055 degrees of longitude (~610 m) at the equator, 5e-4 °C on a temperature
selection and 1e-4 m on SSH: below the plotting resolution of any map the viewer
draws, and far below model error. The header carries the exact
`scale_factor`/`add_offset` per column so the client decodes without guessing.

Header fields: `format`, `version`, `dataset`, `region`, `variable`,
`depth_bin`, `start_date`, `lead_day`, `observation_count`, `matched_count`,
`displayed_count`, `display_point_cap`, `decimated`, `columns`, `quantization`,
`oceanbench_version`.

### Decimation policy, stated plainly

Three counts are recorded on every extract and never conflated:

- `observation_count` — every match-up row of the selection.
- `matched_count` — those with a finite observation **and** a finite model
  value. Non-finite pairs cannot be plotted and are dropped here.
- `displayed_count` — what the file actually stores.

If `matched_count` exceeds `DISPLAY_POINT_CAP` (**50 000**), the extract holds a
uniform sample without replacement of that many points, seeded from a SHA-256 of
the selection key so a regen reproduces the same file byte for byte. A uniform
sample, not a stride: match-ups arrive in observation-file order (satellite
tracks, float profiles) and any fixed stride risks sampling a track geometry
rather than the ocean.

**No score is ever computed from an extract.** RMSD, bias and the year artifacts
come from the parquet, over every observation. The extracts are a display copy.
The viewer must show `displayed_count` against `matched_count` whenever
`decimated` is true (see the viewer wave below); a decimated overlay that says
nothing is the failure mode this policy exists to prevent.

Only SSH exceeds the cap on the measured data (313 858 points, sampled to
50 000). Every other selection is written whole.

### Size budget

| selection | points | file |
|---|---|---|
| SSH surface (capped) | 50 000 | 400 KB |
| median selection | 27 791 | 222 KB |
| T 0-5m | 1 290 | 11 KB |

Against 15 to 22 MB today, the first paint becomes **one request of 11 to
400 KB**, with no footer to pay first and nothing to re-pay per dataset.

### Layout and naming

```
viewer/data/insights/<slug>/<region>/class4-overlays/
    <variable>/<depth_bin>/<start_date>-lead<NN>.obx
    manifest.json
```

`<NN>` is the lead day zero-padded to two digits. `<variable>` and `<depth_bin>`
are the literal values from the match-ups (`sea_water_salinity`, `100-300m`,
`surface`, `15m`) — already path-safe, and rejected by the writer if they ever
stop being. A global dataset/region is 6 240 extracts totalling about 1.3 GB;
all 22 published `(dataset, region)` pairs come to roughly 137 000 objects.

### Indexing

`insights.json` gains one key per `(dataset, region)`, next to
`class4_matchups`, about 350 bytes each:

```json
"class4_overlays": {
  "format": "obx/1",
  "template": "./data/insights/glo12/global/class4-overlays/{variable}/{depth_bin}/{start_date}-lead{lead_day}.obx",
  "display_point_cap": 50000,
  "manifest": "./data/insights/glo12/global/class4-overlays/manifest.json"
}
```

The template is everything the viewer needs to paint: no manifest fetch on the
first paint, no 404 probing. `manifest.json` is the optional availability index
(`variable -> depth_bin -> start_date -> lead_day -> [observation_count,
matched_count, displayed_count]`, about 200 KB for a global year) for enumerating
what exists or showing counts without opening an extract.

`build_viewer_artifacts` returns this object as
`ViewerArtifactsResult.class4_overlay_index_entry`, and writes the same object
at the top of `manifest.json`, so whatever assembles `insights.json` copies it
rather than reconstructing the paths.

## 4. Compatibility and migration

Three vintages will coexist while the regen rolls through:

1. **old** — SNAPPY, 9 columns with `abs_error`, one row group per block.
2. **current** — ZSTD, 8 columns, one row group per block.
3. **next** — ZSTD, 8 columns, row groups packed by pair, plus extracts.

Nothing is deleted and no artifact is rewritten in place; a regen replaces a
`(dataset, region)` directory wholesale.

Detection logic, in order:

1. **Extracts present?** `insights.json` entry has a `class4_overlays` object
   with a `template` → build the URL, fetch one `.obx`, paint. This is vintage 3
   only.
2. **Otherwise** → the existing hyparquet path on `class4_matchups`, unchanged.
   It already handles vintages 1 and 2 and, as argued above, vintage 3 too:
   `allRowGroupsSinglePair` tests only `start_date`/`lead_day`,
   `rowGroupHasVariable` keeps straddling groups, `withAbsoluteError` derives
   `abs_error` when the column is absent, and the ZSTD codec is supplied for
   both codecs.

So the compatibility requirement is met by **not removing** the parquet path
rather than by adding a second one. A viewer shipped before the regen keeps
working against the new files; a viewer shipped after the regen keeps working
against the old ones.

## 5. Viewer wave: what changes later

Not implemented here. File-level specification for the wave that follows.

**`website/viewer/modules/class4-worker.js`**

- Accept an `overlay` op carrying `{template, variable, depth_bin, start_date,
  lead_day}`. Build the URL by substituting the four placeholders, with
  `lead_day` as `String(lead).padStart(2, "0")`.
- `fetch` the whole file (one request, no ranges), then decode: check the magic
  `OBX1`, read `header_length` at offset 4 as `uint32`, `JSON.parse` the header
  slice with trailing NULs trimmed, then take four `Uint16Array` views of
  `displayed_count` each at the aligned payload offset. Decode a column as
  `code * scale_factor + add_offset` from `header.quantization[name]`.
- Return `{rows, total: header.matched_count, displayed: header.displayed_count,
  decimated: header.decimated, overlay: true}`. Derive `abs_error` per row the
  same way `withAbsoluteError` does today, so the row shape reaching the map is
  identical to the parquet path.
- Cache by URL. Keep the existing parquet path untouched as the fallback.

**`website/viewer/app.js`**

- When the `insights.json` entry has `class4_overlays`, ask the worker for the
  overlay op; on any failure fall back to the parquet op for that selection.
- When `decimated` is true, label the overlay with `displayed` of `total`
  observations. This is required, not cosmetic.
- The progress bar has no footer phase on the overlay path: one phase, one
  request, `content-length` known up front.

**No change** is needed for the restructured parquet itself. That is the point
of keeping the pair invariant.

## 6. Where the code is

| what | where |
|---|---|
| packing, statistics, verification | `oceanbench/publish/viewer_artifacts.py` (`_packed_row_group_spans`, `TARGET_ROW_GROUP_ROWS`, `verify_matchup_parquet`) |
| extract encode/decode, manifest, index entry | `oceanbench/publish/class4_overlays.py` |
| wiring into the regen | `build_viewer_artifacts(..., enable_class4_overlays=True)` |
| tests | `tests/publish/test_matchup_layout.py`, `tests/publish/test_class4_overlays.py` |

Extracts are written from the same per-start partitions that stream into the
parquet, so the regen makes one pass over the match-ups and never re-reads the
4 GB file to produce them.
