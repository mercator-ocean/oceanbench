<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# observations2024-v2 builder

`build_observations.py` builds the Class IV observation store that
`oceanbench/core/references/observations.py` reads, one zarr per UTC day:

```
s3://oceanbench-bucket/dev/observations2024-v2/<YYYYMMDD>.zarr
```

It is not part of the installed package.

## Policy

The `POLICY` dictionary at the top of the script holds the choices:

- only quality control flag 1 reaches the scored columns, with position flag 1,
  time flags 1 and 2 and depth flags 1, 2 and 7 accepted for the row itself
- drifter currents come from the Copernicus filtered basis (`EWCT_FILTR` and
  `NSCT_FILTR`, inertial band removed) rather than the raw components
- the wind slippage estimate shipped with the drifter files is subtracted from
  the velocities wherever it is finite
- undrogued drifters, `CURRENT_TEST` codes 11 and 211, are flagged out
- sea level anomalies are bounded at 2 metres in absolute value
- rows falling outside the target UTC day are flagged
- every source row is kept: a row that fails the policy carries `qc_keep=0`, its
  raw values, its own flags and a `qc_reason`, and its scored measurement
  columns are blank

The nine scored variable names and dtypes are those read by
`oceanbench/core/references/observations.py`.

## Basis version

Each day store carries the basis version in the root attribute
`obs_basis_version`, and the reader refuses any day store that does not declare
the expected one. The default policy is `2024-v2.1.0`.

## Running it

Environment variables: `COPERNICUSMARINE_SERVICE_USERNAME` and
`COPERNICUSMARINE_SERVICE_PASSWORD` for
the sea level anomaly downloads, and `CF_KEY` and `CF_SECRET` (or the standard
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`) for the target bucket. The
source bucket is read anonymously.

```sh
python build_observations.py --start 2024-01-01 --end 2025-01-04 --workers 8
```

Each day writes to a temporary prefix and is renamed on success, alongside a
`<YYYYMMDD>.manifest.json` recording the policy, the source files, the row
counts before and after the policy, and the package versions used.

Each day store also records `builder_script_sha256`, the hash of the script as
it ran on the build machine.
