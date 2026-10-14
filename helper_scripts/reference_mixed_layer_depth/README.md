<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Reference mixed layer depth builder

`build_reference_mixed_layer_depth.py` builds the reference mixed layer depth stores that
`oceanbench/core/references/mixed_layer_depth_reference.py` reads, one zarr per Wednesday first day
holding ten daily fields:

```
s3://oceanbench-bucket/public/references/<reference>_mixed_layer_depth_<resolution>_2024/<YYYYMMDD>.zarr
```

with `<reference>` in `glorys`, `glo12` and `<resolution>` in `twelfth_degree`, `quarter_degree`,
`one_degree`. It is not part of the installed package.

## Method

Each store is computed by `build_reference_mixed_layer_depth_week`, which calls the same
`compute_mixed_layer_depth` as the challenger side of the metric, on the reference's full vertical levels
down to the first level at or below 600 m. The reference mixed layer depth therefore does not depend on
the challenger depth levels.

- twelfth degree: native GLORYS or GLO12 temperature and salinity
- quarter degree: the same native fields sampled at the points of the existing quarter degree reference
  grid, which all fall on the native grid
- one degree: the existing full level one degree reference

## Running it

Sources are zarr path templates with a `{first_day}` placeholder. A source written by an OceanBench
`--stage-dir` run, with its `lead_day_index` dimension, is accepted as is. Without
`--twelfth-degree-source` the native fields are opened from Copernicus Marine, which needs a
`copernicusmarine login`. The quarter and one degree defaults are the public reference stores.

```sh
python build_reference_mixed_layer_depth.py --reference glorys --output out \
    --twelfth-degree-source '<stage>/reference-glorys-twelfth_degree-depths-50-<hash>-10d/{first_day}.zarr' \
    --first-days 2024-01-03 2024-01-10 --workers 32
```

Each store is written to a temporary path and renamed on success, and existing stores are skipped, so a
run can be resumed. The stores carry the script hash in `builder_script_sha256`.
