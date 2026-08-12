<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Wind slippage rung, challenger glonet

L3 = default v2 policy (currents on the EWCT_FILTR basis, no WS subtraction).
LWS = same rows, currents minus the wind slippage columns where finite.
LWSp = sign-check variant, currents plus WS. Diagnostic only.
RMSD in m/s, pooled over depth bins. Counts are matchup rows at that lead.

## region global

| variable | lead | n | L3 rmsd | LWS rmsd | LWS delta | LWS % | LWSp rmsd | LWSp % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uo | 1 | 1210224 | 0.13772 | 0.12226 | -0.01546 | -11.23% | 0.17105 | +24.20% |
| uo | 5 | 1200771 | 0.14674 | 0.13165 | -0.01509 | -10.28% | 0.17801 | +21.31% |
| uo | 9 | 1212727 | 0.15447 | 0.13970 | -0.01477 | -9.56% | 0.18547 | +20.07% |
| uv | 1 | 2420448 | 0.13265 | 0.12047 | -0.01218 | -9.18% | 0.16078 | +21.21% |
| uv | 5 | 2401542 | 0.14245 | 0.13023 | -0.01222 | -8.58% | 0.16913 | +18.73% |
| uv | 9 | 2425454 | 0.14923 | 0.13742 | -0.01182 | -7.92% | 0.17536 | +17.51% |
| vo | 1 | 1210224 | 0.12738 | 0.11865 | -0.00873 | -6.85% | 0.14981 | +17.61% |
| vo | 5 | 1200771 | 0.13802 | 0.12879 | -0.00923 | -6.69% | 0.15976 | +15.75% |
| vo | 9 | 1212727 | 0.14380 | 0.13509 | -0.00871 | -6.06% | 0.16464 | +14.49% |

## region ibi

| variable | lead | n | L3 rmsd | LWS rmsd | LWS delta | LWS % | LWSp rmsd | LWSp % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| uo | 1 | 18771 | 0.11275 | 0.09869 | -0.01406 | -12.47% | 0.13994 | +24.11% |
| uo | 5 | 18533 | 0.11793 | 0.10124 | -0.01669 | -14.15% | 0.14806 | +25.54% |
| uo | 9 | 18292 | 0.12426 | 0.10641 | -0.01785 | -14.37% | 0.15559 | +25.22% |
| uv | 1 | 37542 | 0.11153 | 0.09768 | -0.01385 | -12.42% | 0.13957 | +25.14% |
| uv | 5 | 37066 | 0.11640 | 0.10154 | -0.01487 | -12.77% | 0.14503 | +24.59% |
| uv | 9 | 36584 | 0.12262 | 0.10743 | -0.01519 | -12.39% | 0.15125 | +23.34% |
| vo | 1 | 18771 | 0.11030 | 0.09666 | -0.01364 | -12.37% | 0.13920 | +26.20% |
| vo | 5 | 18533 | 0.11485 | 0.10183 | -0.01303 | -11.34% | 0.14194 | +23.58% |
| vo | 9 | 18292 | 0.12097 | 0.10845 | -0.01252 | -10.35% | 0.14677 | +21.33% |


## WS column facts, full year, 370 days

Columns `uo_ws` / `vo_ws`, sourced from `EWCT_WS_FILTR` / `NSCT_WS_FILTR`
(long_name "eastward current wind slippage correction at the drog depth
filtered over 3 days"), units m/s, plus `ws_type` from
`WS_TYPE_OF_PROCESSING` (0 nominal, 1 from_mean, 2 from_climatology,
3 adaptative, -1 absent).

| fact | value |
| --- | --- |
| currents rows kept by the default policy | 8636806 |
| of those with finite uo_ws and vo_ws | 8128124, 94.11% |
| IBI kept rows | 137456, of which finite WS 117280, 85.32% |
| ws magnitude sqrt(uo_ws^2+vo_ws^2) median | 0.00515 m/s |
| p90 | 0.12956 m/s |
| p99 | 0.26705 m/s |
| max | 0.70433 m/s |
| rms | 0.07359 m/s |
| mean kept current speed for scale | 0.20422 m/s |
| ws_type over kept rows | 0 nominal 5720395, 1 from_mean 2473357, 2 from_climatology 443054 |

Sign convention. The source long_name calls it a correction, which alone is
ambiguous, so the sign was checked empirically with rung LWSp. Subtracting
improves every current score by 6 to 14 percent, adding degrades every score by
14 to 26 percent. FILTR minus WS is the correct operation, matching the NRT
producer.

Mean signed obs delta (delta = -ws), global 8128124 rows:
du +0.004319 m/s, dv +0.000802 m/s, against an rms ws of 0.073592.
The ratio |mean| / rms is 0.060, so the correction is not a near-constant
offset; it is spatially structured. By 20 degree latitude band:

| band | n | mean du | mean dv | rms ws |
| --- | --- | --- | --- | --- |
| -80 | 73905 | -0.024888 | +0.005245 | 0.086121 |
| -60 | 910384 | -0.019412 | +0.002951 | 0.063131 |
| -40 | 2462863 | +0.004349 | -0.002311 | 0.077621 |
| -20 | 517971 | +0.024934 | -0.011085 | 0.077763 |
| 0 | 665181 | +0.031020 | +0.011842 | 0.089664 |
| 20 | 2453522 | +0.007182 | +0.003462 | 0.075663 |
| 40 | 856011 | -0.008063 | -0.001741 | 0.047851 |
| 60 | 188287 | -0.001949 | -0.000014 | 0.052508 |

IBI kept rows with finite WS, 117280: mean du -0.004748, mean dv +0.008149.

## CURRENT_TEST distribution, side request, full year

Over every currents row in the v2 store (obs_type 3), 10132425 rows.
No row carries the -1 fill, so CURRENT_TEST is present everywhere.

| code | rows in store | pct of store | rows kept by default policy | pct of kept |
| --- | --- | --- | --- | --- |
| 313 | 4493258 | 44.35% | 4468332 | 51.74% |
| 312 | 3110721 | 30.70% | 3092839 | 35.81% |
| 011 | 1400457 | 13.82% | 0 | 0.00% |
| 311 | 628275 | 6.20% | 625546 | 7.24% |
| 213 | 257048 | 2.54% | 232110 | 2.69% |
| 212 | 207599 | 2.05% | 186955 | 2.16% |
| 211 | 35067 | 0.35% | 31024 | 0.36% |

Default policy keeps 8636806 rows, everything except code 011.
Keeping only 313 leaves 4468332 rows, 51.74% of the default set,
a drop of 4168474 rows, and 44.35% of every currents row in the store.

## Run

Array job 34827, partition monoproc, 52 tasks, array 0-51%8, all exit 0.
Global tasks 204s each, IBI tasks 47s each, about 18 min wall clock.
View materialisation 19s for 370 days x 2 rungs.
