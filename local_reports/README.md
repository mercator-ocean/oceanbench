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

`0.5.0/glow.global.report.ipynb` is a local evaluation of run `glowcascade_v4`:
all 52 Wednesday challenger starts of 2024, 20240103 through 20241225, each
initialised from the as-issued GLO12 nowcast of the Tuesday before it, quarter
degree, not an official submission.

`glowcascade_v4` is a replay of the served cascade recipe on the 32 epoch pretrain,
with two checkpoints and nothing else changed:

- lead 1 from the v4 anneal checkpoint, step 87616
  (`glow930m_scratch_v4/runs/scratch_v4/keep_epochs/ckpt_step00087616.pt`)
- leads 2 to 9 from the v4 ladder h5 checkpoint, step 1869
  (`glow930m_v4_ladder/runs/stage2_ladder/keep_epochs/ckpt_step00001869.pt`)

It replaces the earlier `glowcascade_final` entry that stood here before; only one
GLOW report is kept at a time, the website shows a single file per challenger.

It scores NINE lead days, not ten. The IFS forecast package that forces the run
carries lead_day_index 0..9, so the tenth forecast day would be driven by persisted
lead 9 forcing rather than by a real forecast. The entry stops at the nine days a
real IFS forecast covers. The forecast store still holds ten days per start; the
challenger module drops the last time step on the way in, nothing was recomputed.
Both harnesses accept a nine day submission natively: seven of the nine metrics
return lead days 1 to 9 and the two Lagrangian metrics return lead days 2 to 8, with
no empty cells. Beside a ten day baseline the website simply leaves the lead 10 cell
blank.

The harness that produced this notebook is OceanBench 0.5.1. The file sits in the
0.5.0 directory because that key is what places a system in the 0.5.0 leaderboard
column, and 0.5.0 is the default published version, so this entry appears next to
officially published systems that were themselves scored with 0.5.0. Keep that
difference in mind when reading small gaps between GLOW and its neighbours.

`tables/glowcascade_v4.*.csv` holds the nine metric tables read back out of that
notebook. The website reads the notebook, not the CSV files; they are kept for
provenance.

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
