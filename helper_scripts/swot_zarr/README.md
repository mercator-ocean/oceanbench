<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# SWOT daily zarr converter

`convert_swot_passes_to_daily_zarr.py` builds the SWOT observation store that
`oceanbench/core/references/swot_observations.py` reads, one zarr per UTC day:

```
s3://oceanbench-bucket/class4/swot/l3/zarr-v2/<YYYY>/<YYYYMMDD>.zarr
```

from the SWOT L3 KaRIn `LR_SSH_Basic` passes stored in the same bucket:

```
s3://oceanbench-bucket/class4/swot/l3/<reproc|forward>/<YYYY>/SWOT_L3_LR_SSH_Basic_*.nc
```

It is not part of the installed package.

## Policy

- the sea level anomaly is `ssha_filtered`; a pass without it fails, and so does its day
- a day is built from the reprocessed passes when any cover it, from the forward passes otherwise
- the two-dimensional swath is flattened into a list of observations; an observation is kept only if its
  sea level anomaly, latitude, longitude and time are all present
- as in the nadir observation store, sea level anomalies are bounded at 2 metres in absolute value
- a pass crossing midnight is read for both days, and each observation is kept only in the UTC day
  of its own time
- longitudes are converted from [0, 360) to [-180, 180)
- one observation out of `--subsample-factor` (10 by default) is kept, drawn at random with the date as
  seed, so that converting a day again keeps the same observations

The Class IV ocean mask is not applied here: it is applied at scoring time, to SWOT as to nadir
observations, and removes in particular the coastal and inland observations.

Each day store holds `sea_surface_height_above_geoid`, `latitude`, `longitude` and `time` along the
`observations` dimension. The sea level anomaly, latitude and longitude keep the integer packing of the
source files, and the time keeps their time encoding. The root attributes record `obs_basis_version`,
`maximum_absolute_sea_level_anomaly_in_meters`, `source_variable`, `processing_mode`,
`source_pass_files` and `subsample_factor`.

## Basis version

The reader refuses any day store that does not declare the expected `obs_basis_version` and a
`subsample_factor` of 10. The current version is `v2.2`:

- `v2.0`: first version, with times written in nanoseconds since 1970, a unit the pinned xarray version
  cannot decode
- `v2.1`: times keep the encoding of the source files
- `v2.2`: sea level anomalies bounded at 2 metres in absolute value, as in the nadir observation store

## Running it

Environment variables: `CF_KEY` and `CF_SECRET` (or the standard `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`) for the bucket. Besides the OceanBench dependencies, the script needs `s3fs`
and `h5netcdf`.

```sh
python convert_swot_passes_to_daily_zarr.py
python convert_swot_passes_to_daily_zarr.py --start 2024-01-01 --end 2025-01-10
```

Without `--start` and `--end`, every day covered by a pass in the bucket is considered.

A day is converted again only when its pass files, the subsample factor or the basis version have
changed since its last conversion, or with `--overwrite`. Running the script again after new passes
reach the bucket therefore converts only the new days and the days those passes complete. A day whose
conversion fails, for instance because of an unreadable pass file, is not written, and is converted at
the next run.
