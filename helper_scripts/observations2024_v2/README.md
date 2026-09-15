<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# observations2024-v2 builder

`build_observations.py` produced the Class IV observation store that
`oceanbench/core/references/observations.py` reads, one zarr per UTC day:

```
s3://oceanbench-bucket/dev/observations2024-v2/<YYYYMMDD>.zarr
```

It is kept here for provenance and for rebuilding a day, not as part of the
installed package. Nothing imports it.

## Policy

The `POLICY` dictionary at the top of the script is the only place the decisions
live. In short:

- only quality control flag 1 reaches the scored columns, with position flag 1,
  time flags 1 and 2 and depth flags 1, 2 and 7 accepted for the row itself
- drifter currents come from the Copernicus filtered basis (`EWCT_FILTR` and
  `NSCT_FILTR`, inertial band removed) rather than the raw components
- the wind slippage estimate shipped with the drifter files is subtracted from
  the velocities wherever it is finite
- undrogued drifters, `CURRENT_TEST` codes 11 and 211, are flagged out
- sea level anomalies are bounded at 2 metres in absolute value
- rows falling outside the target UTC day are flagged rather than dropped
- every source row is kept: a row that fails the policy carries `qc_keep=0`, its
  raw values, its own flags and a `qc_reason`, and its scored measurement
  columns are blank

The nine scored variable names and dtypes are those read by
`oceanbench/core/references/observations.py`.

## Basis versions

The published store carries the basis version it was built on in the root
attribute `obs_basis_version`, and the reader refuses any day store that does
not declare the expected one.

- `2024-v2.0.0` first build
- `2024-v2.0.1` sea level anomaly longitude normalised onto [-180, 180)
- `2024-v2.1.0` currents policy: filtered components minus the wind slippage
  where it is finite, `CURRENT_TEST` codes 11 and 211 blanked

The published store is at `2024-v2.1.0`. The two later steps were applied to it
in place; they are now folded into this script, so a fresh run with the default
policy reproduces the published store directly.

## Running it

Credentials come from the environment only, nothing is hardcoded:
`COPERNICUSMARINE_SERVICE_USERNAME` and `COPERNICUSMARINE_SERVICE_PASSWORD` for
the sea level anomaly downloads, and `CF_KEY` and `CF_SECRET` (or the standard
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`) for the target bucket. The
source bucket is read anonymously.

```sh
python build_observations.py --start 2024-01-01 --end 2025-01-04 --workers 8
```

Each day writes to a temporary prefix and is renamed on success, alongside a
`<YYYYMMDD>.manifest.json` recording the policy, the source files, the row
counts before and after the policy, and the package versions used.

## Note on the recorded script hash

Each day store records `builder_script_sha256`, the hash of the script as it ran
on the build machine. The published days were built and patched in steps, so
that hash does not match the copy here.
