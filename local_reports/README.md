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

`0.5.0/glow.global.report.ipynb` is a local evaluation: 51 Tuesday starts of 2024,
as-issued GLO12 nowcast initial conditions, OceanBench 0.5.1 harness, quarter degree,
not an official submission. It is listed under the 0.5.0 leaderboard column because
0.5.0 is the default published version, so the officially published systems it sits
next to were scored with 0.5.0.

`tables/` holds the nine metric CSV files the harness wrote for the same run. The
website reads the notebook, not the CSV files; they are kept for provenance.

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
