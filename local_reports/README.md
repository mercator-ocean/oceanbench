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

`0.6.0/glow.global.report.ipynb` is a local evaluation of run
`glowcascade_v4_nofilter`: all 52 Wednesday challenger starts of 2024, 20240103
through 20241225, each initialised from the as-issued GLO12 nowcast of the Tuesday
before it, quarter degree, not an official submission.

`glowcascade_v4_nofilter` is the `glowcascade_v4` recipe with the equatorial notch
turned off, that is the rollout run without `--eqfilter`. That is the configuration
GLOW actually serves, so the preview entry now matches the served model. The two
checkpoints and everything else are the same:

- lead 1 from the v4 anneal checkpoint, step 87616
  (`glow930m_scratch_v4/runs/scratch_v4/keep_epochs/ckpt_step00087616.pt`)
- leads 2 to 9 from the v4 ladder h5 checkpoint, step 1869, switched in after lead 1
  (`eqfilter_v2/stage/ckpt_ladder_step00001869.pt`)

The notch is not a small thing in the box it acts on: the in-box band power median
runs 58 to 181 times higher without it for temperature, 130 to 232 for the zonal
current, and 85 to 403 for sea surface height, growing with lead. It is the headline
scores that barely move, because the box is a small part of the global domain.

It replaces the `glowcascade_v4` entry that stood here before; only one GLOW report
is kept at a time, the website shows a single file per challenger. On the mean of
the nine metric tables the two runs sit 0.012 percent apart, the no-filter run very
slightly the worse of the two.

It scores NINE lead days, not ten. The IFS forecast package that forces the run
carries lead_day_index 0..9, so the tenth forecast day would be driven by persisted
lead 9 forcing rather than by a real forecast. The entry stops at the nine days a
real IFS forecast covers. The forecast store still holds ten days per start; the
challenger module drops the last time step on the way in, nothing was recomputed.
Both harnesses accept a nine day submission natively: seven of the nine metrics
return lead days 1 to 9 and the two Lagrangian metrics return lead days 2 to 8, with
no empty cells. Beside a ten day baseline the website simply leaves the lead 10 cell
blank.

The harness that produced this notebook is oceanbench `origin/main` 7e5ec87, the
pending 0.6.0; its installed dist-info still stamps `__version__` as 0.5.1, so the
first cell of the notebook reads 0.5.1. The observations metric therefore comes from
the observations-v2 basis, which is why its table carries an observation count column
and lower current errors than the earlier 0.5.1 scored entry did. The file sits in the
0.6.0 directory because that key is what places a system in the 0.6.0 leaderboard
column, which is the default of this preview index, so this entry appears next to
systems rescored with 0.6.0, the same harness basis it was scored with.

`tables/glowcascade_v4_nofilter.*.csv` holds the nine metric tables read back out of
that notebook. The website reads the notebook, not the CSV files; they are kept for
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
