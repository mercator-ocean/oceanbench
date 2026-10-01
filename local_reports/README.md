<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Local reports

Report notebooks that ship with this branch instead of being downloaded from the
public bucket. ``website/helpers/local_challengers.py`` declares which challenger and
region each file covers; `website/download_reports.py` copies them into
`website/reports/<version>/` next to the officially published ones.

Nothing here is uploaded anywhere.

## glow

`0.6.0/glow.global.report.ipynb` is a local evaluation of GLOW run
`glowcascade_v5_ring`, a 930M parameter cascade. It is not an official submission.

Lead 1 comes from the annealed checkpoint, step 87616, md5
`5215b85c13f7bb356ff38235286abc70`. Leads 2 to 9 come from the ring-clean ladder
checkpoint, step 2400, md5 `8ce27dab259d588e4ba9030c6d51dc17`, switched in after
lead 1. Both use EMA weights and the equatorial notch is off.

It covers all 52 Wednesday starts of 2024, 20240103 through 20241225, nine scored
lead days, forced with the as-issued IFS package and initialised from the as-issued
GLO12 nowcast, with a zos datum shift of 0.0.

The forecast stores are exported on the canonical 672x1440 grid with 21 serve
levels. Since 2026-09-21 every store is masked level by level with the combined
GLO12 and GLORYS wet mask, taking the first reference level at or deeper than the
GLOW level, so no below seabed values are served.

The harness that produced this notebook is oceanbench `main` 7e5ec87, the pending
0.6.0, run under a local venv.

`tables/glowcascade_v5_ring_060_remask.*.csv` holds the nine metric tables read back
out of that notebook. The website reads the notebook, not the CSV files; they are
kept for provenance.

The earlier pre-remask report is kept under `local_reports/withdrawn/0.6.0/` for
reference. The `tables/glowcascade_v4_nofilter.*.csv` files and their license
sidecars belong to that withdrawn report.

## glow117m

`0.6.0/glow117m.global.report.ipynb` is a local evaluation of GLOW run
`glowcascade117m_v5_ring`, the 117M parameter variant of the `glowcascade_v5_ring`
cascade above. It is not an official submission.

Same recipe and same steps as the 930M chain: pretrain plus anneal to step 87616,
hot stage 2, ladder, then the ring penalty stage of 2400 steps. Lead 1 comes from
the 117M annealed checkpoint, step 87616, md5 `bd84b8a687e2d387e3b80bce675ef84b`.
Leads 2 to 9 come from the 117M ring checkpoint, step 2400, md5
`cff4866fdd517aa9d81b3e5b06f00ce9`, switched in after lead 1. Both use EMA weights
and the equatorial notch is off.

Starts, initial conditions, IFS forcing, zos datum shift, 672x1440 grid with 21
serve levels, and the combined GLO12 and GLORYS wet mask are the same as for
`glowcascade_v5_ring`; the mask is applied at write time and matches the 930M
stores cell for cell. The stores hold the nine scored lead days only. Harness:
oceanbench `main` 7e5ec87, the pending 0.6.0, same venv and flags as the 930M run.

`tables/glowcascade117m_v5_ring_060.*.csv` holds the nine metric tables read back
out of that notebook, kept for provenance.

## glow21m

`0.6.0/glow21m.global.report.ipynb` is a local evaluation of GLOW run
`glowcascade21m_v5_ring`, the 21M parameter variant of the `glowcascade_v5_ring`
cascade above. It is not an official submission.

Same recipe and same steps as the 930M and 117M chains: pretrain plus anneal to
step 87616, hot stage 2, ladder, then the ring penalty stage of 2400 steps. Lead 1
comes from the 21M annealed checkpoint, step 87616, md5
`42363108532a37c0f2f97e7dd6797eae`. Leads 2 to 9 come from the 21M ring checkpoint,
step 2400, md5 `8709422ad6b8dbcdc5c54fb4605bd3e2`, switched in after lead 1. Both
use EMA weights and the equatorial notch is off.

Starts, initial conditions, IFS forcing, zos datum shift, 672x1440 grid with 21
serve levels, and the combined GLO12 and GLORYS wet mask are the same as for
`glowcascade_v5_ring`; the mask is applied at write time and matches the 930M
stores cell for cell. The stores hold the nine scored lead days only. Harness:
oceanbench `main` 7e5ec87, the pending 0.6.0, same venv and flags as the 930M run.

`tables/glowcascade21m_v5_ring_060.*.csv` holds the nine metric tables read back
out of that notebook, kept for provenance.

## hclimrep

`0.5.0/hclimrep.global.report.ipynb` is a local evaluation of HClimRep, the ECMWF
WeatherGenerator ocean model, run `ft0818`
(`glory_finetuning_20260818_105654`): 52 weekly Wednesday starts, 2024-01-03 through
2024-12-25, each initialised from the GLORYS nowcast of the preceding Tuesday,
quarter degree, 10 lead days, region global, OceanBench 0.5.0 harness. It is not an
official submission. The harness version is 0.5.0, which is also the default published
version, so it sits in the 0.5.0 leaderboard column alongside systems scored with the
same version.

Forecast store: `wg_ft0818_B1.zarr` from
https://huggingface.co/datasets/kacpnowak/hclimrep-ocean-oceanbench, staged locally on
gpu1 and rechunked without any change to the values.

`tables/hclimrep_ft0818.*.csv` holds the nine metric tables read back out of that
notebook. The website reads the notebook, not the CSV files; they are kept for
provenance.
