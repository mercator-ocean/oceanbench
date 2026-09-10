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
