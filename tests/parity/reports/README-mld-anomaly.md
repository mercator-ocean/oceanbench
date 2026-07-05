<!--
SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# MLD parity anomaly

The wave-1 like-for-like parity run compared gridded RMSD in unweighted mode
against the published v0.2.1 golden. Non-SLA rows were gated at `5e-7`.
All clear MLD failures were gridded mixed-layer-depth rows:

| challenger | metric | rows | max abs diff | mean abs diff |
|---|---|---:|---:|---:|
| glonet | `rmsd_mld_glo12` | 10 | `5.14505e-6` | `2.35739e-6` |
| glonet | `rmsd_mld_glorys` | 10 | `5.43094e-6` | `1.68689e-6` |
| xihe | `rmsd_mld_glo12` | 10 | `4.41656e-6` | `1.96709e-6` |
| xihe | `rmsd_mld_glorys` | 10 | `1.01715e-5` | `3.60332e-6` |

Two non-MLD xihe gridded groups were only marginally over the same gate:
`rmsd_geostrophic_glo12` at `5.26009e-7` and
`rmsd_variables_glo12` at `5.08339e-7`.

## Best-supported explanation

The current implementation computes MLD in `oceanbench/core/mixed_layer_depth.py`
by:

- capping the native depth coordinate at 600 m before density computation;
- deriving absolute salinity with `gsw.SA_from_SP`;
- deriving potential density with `gsw.pot_rho_t_exact`;
- selecting the first native depth where
  `potential_density - surface_density >= 0.03`.

The runner path in `oceanbench/runner/run.py` uses the same
`compute_mixed_layer_depth` transform for challenger and reference before
calling gridded RMSD. I did not find a separate runner-specific MLD formula.

The anomaly is therefore most plausibly numerical sensitivity in the MLD
diagnostic rather than an RMSD aggregation bug. MLD is a thresholded,
depth-index-valued diagnostic: tiny density changes from regenerated staged
fields, xarray/dask chunk execution, float32 input/interpolation ordering, or a
`gsw`/dependency version difference can move a small number of grid cells across
the `0.03 kg/m^3` threshold, then RMSD averages those discrete depth changes.
That explains why ordinary gridded variables are at parse/gate precision while
MLD is consistently one order of magnitude larger (`4e-6` to `1e-5`) across both
wave-1 challengers and both references.

The detailed `runs-unweighted/` parquet artifacts referenced by the parity
verdict were not present in this worktree during this bounded inspection, so I
could not localize the MLD drift by start date or lead day. Do not relax the
`5e-7` non-SLA parity gate silently; either preserve this as a documented known
MLD sensitivity or rerun with retained per-row artifacts to prove the exact
stage/library source.
