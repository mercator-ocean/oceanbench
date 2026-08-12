<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# CURRENT_TEST strata, challenger glonet, currents only

Basis: FILTR minus wind slippage (the LWS rung), default policy row set, plus the
confirmed undrogued stratum 011 restored from the source archive so it is scored on
the same basis. One scoring pass, 52 start dates, region global, leads 1 to 9.
RMSD and bias in m/s. uv pools the two components, so its count is twice the matchups.
Bias is model minus observation.

## Global pooled, leads 1 to 9

| code | matchups | uo rmsd | vo rmsd | uv rmsd | uo bias | vo bias | uv vs 313 | 95% CI | scale 313=0 011=1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 313 | 5585363 | 0.13361 | 0.13009 | 0.13186 | +0.00163 | +0.00520 | reference |  | 0.00 |
| 312 | 3862000 | 0.12603 | 0.12186 | 0.12396 | +0.00163 | +0.00558 | -0.00790 | [-0.01079, -0.00505] | -0.71 |
| 311 | 796514 | 0.13993 | 0.13420 | 0.13709 | +0.00657 | +0.00226 | +0.00217 | [-0.00693, +0.01303] | 0.47 |
| 213 | 288092 | 0.13977 | 0.13644 | 0.13812 | +0.00429 | +0.00323 | +0.00626 | [+0.00139, +0.01102] | 0.56 |
| 212 | 233991 | 0.13348 | 0.12348 | 0.12858 | +0.00586 | +0.00598 | -0.00328 | [-0.00758, +0.00157] | -0.30 |
| 211 | 39761 | 0.14954 | 0.15420 | 0.15189 | +0.01198 | -0.00806 | +0.01863 | [+0.00470, +0.03031] | 1.80 |
| 011 | 1623100 | 0.14761 | 0.13817 | 0.14297 | +0.01304 | +0.00350 | +0.01121 | [+0.00697, +0.01590] | 1.00 |

## Lead 1 against lead 9, global, uv

| code | n lead 1 | uv rmsd lead 1 | vs 313 | n lead 9 | uv rmsd lead 9 | vs 313 | growth |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 313 | 624974 | 0.12279 | +0.00000 | 631055 | 0.13944 | +0.00000 | +0.00000 |
| 312 | 428467 | 0.11422 | -0.00857 | 431677 | 0.13192 | -0.00752 | +0.00105 |
| 311 | 88706 | 0.12871 | +0.00592 | 88412 | 0.14528 | +0.00584 | -0.00008 |
| 213 | 35090 | 0.12659 | +0.00380 | 31685 | 0.14792 | +0.00848 | +0.00467 |
| 212 | 28138 | 0.12292 | +0.00014 | 25538 | 0.13242 | -0.00702 | -0.00715 |
| 211 | 4851 | 0.13991 | +0.01712 | 4365 | 0.15880 | +0.01936 | +0.00224 |
| 011 | 190583 | 0.13276 | +0.00998 | 179548 | 0.15311 | +0.01367 | +0.00369 |

## Latitude bands, uv RMSD, leads 1 to 9

Each cell is the uv RMSD and, in brackets, the matchup count.

| band | 313 | 312 | 311 | 213 | 212 | 211 | 011 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| -80 | 0.1143 (41889) | 0.1182 (48406) | 0.1272 (8929) | 0.1247 (2275) | 0.1250 (2218) | 0.0912 (58) | 0.1638 (6890) |
| -60 | 0.1347 (558935) | 0.1154 (516100) | 0.1307 (91383) | 0.1726 (18102) | 0.1313 (15270) | 0.1294 (3659) | 0.1484 (97073) |
| -40 | 0.1270 (1541626) | 0.1184 (1317322) | 0.1293 (245544) | 0.1505 (78736) | 0.1290 (56322) | 0.1789 (11289) | 0.1324 (725207) |
| -20 | 0.1315 (387496) | 0.1388 (239645) | 0.1344 (44711) | 0.1303 (22708) | 0.1489 (13325) | 0.1315 (2362) | 0.1433 (92340) |
| 0 | 0.1524 (518269) | 0.1609 (285109) | 0.1602 (66185) | 0.1644 (19072) | 0.1682 (15597) | 0.1556 (3018) | 0.1604 (108115) |
| 20 | 0.1337 (1762505) | 0.1253 (1019093) | 0.1474 (240093) | 0.1273 (114033) | 0.1208 (108917) | 0.1421 (16977) | 0.1522 (469155) |
| 40 | 0.1207 (647297) | 0.1084 (367379) | 0.1153 (83242) | 0.1173 (12840) | 0.1303 (13981) | 0.1216 (779) | 0.1568 (66736) |
| 60 | 0.1218 (127346) | 0.1222 (66597) | 0.1436 (16211) | 0.1027 (20326) | 0.0913 (8361) | 0.1265 (1619) | 0.1314 (51378) |

### Band difference against 313, uv RMSD, with a 95 percent bootstrap interval over start dates

| band | code | n | delta uv | 95% CI | distinguishable | scale 313=0 011=1 |
| --- | --- | --- | --- | --- | --- | --- |
| -80 | 312 | 48406 | +0.00360 | [-0.01083, +0.01752] | no | 0.08 |
| -80 | 311 | 8929 | +0.02389 | [+0.00289, +0.05706] | yes | 0.26 |
| -80 | 213 | 2275 | +0.01443 | [-0.00569, +0.03491] | no | 0.21 |
| -80 | 212 | 2218 | +0.02220 | [-0.01338, +0.06265] | no | 0.21 |
| -80 | 011 | 6890 | +0.05343 | [+0.02617, +0.07662] | yes | 1.00 |
| -60 | 312 | 516100 | -0.01925 | [-0.02510, -0.01380] | yes | -1.40 |
| -60 | 311 | 91383 | -0.00623 | [-0.02301, +0.01574] | no | -0.29 |
| -60 | 213 | 18102 | +0.03789 | [+0.01197, +0.06318] | yes | 2.76 |
| -60 | 212 | 15270 | -0.00350 | [-0.02032, +0.01567] | no | -0.24 |
| -60 | 211 | 3659 | -0.01088 | [-0.02661, +0.00904] | no | -0.39 |
| -60 | 011 | 97073 | +0.01379 | [+0.00488, +0.02286] | yes | 1.00 |
| -40 | 312 | 1317322 | -0.00862 | [-0.01272, -0.00437] | yes | -1.60 |
| -40 | 311 | 245544 | +0.00181 | [-0.00686, +0.01140] | no | 0.42 |
| -40 | 213 | 78736 | +0.02342 | [+0.01467, +0.03258] | yes | 4.36 |
| -40 | 212 | 56322 | +0.00192 | [-0.00669, +0.01031] | no | 0.36 |
| -40 | 211 | 11289 | +0.05857 | [+0.02441, +0.08531] | yes | 9.66 |
| -40 | 011 | 725207 | +0.00537 | [+0.00092, +0.00917] | yes | 1.00 |
| -20 | 312 | 239645 | +0.00732 | [+0.00062, +0.01323] | yes | 0.62 |
| -20 | 311 | 44711 | +0.00435 | [-0.00893, +0.01649] | no | 0.25 |
| -20 | 213 | 22708 | -0.00154 | [-0.01202, +0.00865] | no | -0.10 |
| -20 | 212 | 13325 | +0.01405 | [-0.00294, +0.03242] | no | 1.47 |
| -20 | 211 | 2362 | +0.00648 | [-0.01101, +0.01428] | no | 0.00 |
| -20 | 011 | 92340 | +0.01172 | [+0.00545, +0.01805] | yes | 1.00 |
| 0 | 312 | 285109 | +0.00851 | [+0.00179, +0.01501] | yes | 1.06 |
| 0 | 311 | 66185 | +0.00888 | [-0.00548, +0.02384] | no | 0.97 |
| 0 | 213 | 19072 | +0.01202 | [-0.00754, +0.03544] | no | 1.50 |
| 0 | 212 | 15597 | +0.01592 | [+0.00095, +0.03062] | yes | 1.98 |
| 0 | 211 | 3018 | +0.00914 | [-0.00862, +0.02300] | no | 0.40 |
| 0 | 011 | 108115 | +0.00830 | [-0.00488, +0.02140] | no | 1.00 |
| 20 | 312 | 1019093 | -0.00841 | [-0.01200, -0.00467] | yes | -0.45 |
| 20 | 311 | 240093 | +0.00563 | [-0.00343, +0.01818] | no | 0.74 |
| 20 | 213 | 114033 | -0.00641 | [-0.01220, -0.00033] | yes | -0.35 |
| 20 | 212 | 108917 | -0.01285 | [-0.01890, -0.00631] | yes | -0.69 |
| 20 | 211 | 16977 | -0.00059 | [-0.01346, +0.01531] | no | 0.46 |
| 20 | 011 | 469155 | +0.01881 | [+0.01042, +0.02884] | yes | 1.00 |
| 40 | 312 | 367379 | -0.01226 | [-0.01762, -0.00689] | yes | -0.34 |
| 40 | 311 | 83242 | -0.01355 | [-0.03214, +0.00671] | no | -0.15 |
| 40 | 213 | 12840 | -0.00242 | [-0.01457, +0.00819] | no | -0.09 |
| 40 | 212 | 13981 | +0.00938 | [-0.00310, +0.02206] | no | 0.27 |
| 40 | 011 | 66736 | +0.03618 | [+0.02782, +0.04421] | yes | 1.00 |
| 60 | 312 | 66597 | -0.00009 | [-0.00783, +0.00818] | no | 0.04 |
| 60 | 311 | 16211 | +0.01243 | [+0.00433, +0.02503] | yes | 2.27 |
| 60 | 213 | 20326 | -0.01909 | [-0.02948, -0.00853] | yes | -1.99 |
| 60 | 212 | 8361 | -0.03107 | [-0.04046, -0.02168] | yes | -3.18 |
| 60 | 211 | 1619 | -0.02358 | [-0.04795, -0.00511] | yes | 0.49 |
| 60 | 011 | 51378 | +0.00986 | [+0.00049, +0.01906] | yes | 1.00 |

## Wind slippage magnitude per code, whole year, scored row set

Magnitude is sqrt(uo_ws^2 + vo_ws^2) in m/s over the store rows that enter the
scoring set. ws_type is WS_TYPE_OF_PROCESSING: 0 nominal, 1 from_mean,
2 from_climatology, 3 adaptative, -1 absent.

| code | store rows scored | finite WS % | median | p90 | p99 | rms | ws_type mix |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 313 | 4468332 | 95.18 | 0.00033 | 0.08199 | 0.22156 | 0.05384 | 0:64% 1:32% 2:4% |
| 312 | 3092839 | 97.23 | 0.02523 | 0.16461 | 0.29628 | 0.09334 | 0:74% 1:24% 2:2% |
| 311 | 625546 | 71.90 | 0.00000 | 0.12440 | 0.26803 | 0.07074 | 0:43% 1:29% 2:28% |
| 213 | 232110 | 94.17 | 0.00674 | 0.13318 | 0.25039 | 0.07246 | 0:68% 1:27% 2:5% |
| 212 | 186955 | 94.33 | 0.04256 | 0.18327 | 0.30217 | 0.10499 | 0:76% 1:19% 2:5% |
| 211 | 31024 | 75.41 | 0.00000 | 0.15162 | 0.28293 | 0.08017 | 0:49% 1:26% 2:25% |
| 011 | 1311079 | 94.78 | 0.00684 | 0.08578 | 0.16792 | 0.04946 | 0:74% 1:20% 2:5% |

## What a confidence cut would remove

Percentages are of the default policy kept set (011 already excluded).

| cut | codes dropped | store rows removed | % of kept store rows | matchups removed | % of matchups |
| --- | --- | --- | --- | --- | --- |
| strong only | 312 311 213 212 211 | 4168474 | 48.26% | 5220358 | 48.31% |
| drop weak wind correlation | 312 212 | 3279794 | 37.97% | 4095991 | 37.91% |
| drop wind test not performed | 311 211 | 656570 | 7.60% | 836275 | 7.74% |
| drop weak submersion | 213 212 211 | 450089 | 5.21% | 561844 | 5.20% |
| drop 211 only | 211 | 31024 | 0.36% | 39761 | 0.37% |
| drop 211 and 213 | 211 213 | 263134 | 3.05% | 327853 | 3.03% |

## Method and validation

The residuals come from one scoring pass over a new read-time view, views2/STRAT.
That view is the LWS recipe (FILTR minus wind slippage, longitude normalised to
[-180, 180)) with two additions: CURRENT_TEST is carried as a per-row variable, and
the confirmed undrogued rows are restored. The builder blanks the FILTR value of any
row that fails policy, so the 011 values were read back from the source archive
GL_TS_DC_YYYYMMDD_FILTR.nc and joined on obs_id. Only the 011 rows whose sole policy
failure was the drogue rule were restored, that is qc_reason exactly "undrogued", so
the stratum carries the same position, time and current QC as the kept set.

Join and basis checks, all 370 days:

| check | value |
| --- | --- |
| currents rows in the store | 10132425 |
| rows matched to the archive | 10132425, none unmatched |
| max abs difference archive FILTR minus stored value on kept rows | 0.000e+00 |
| default policy kept rows carried into the view | 8636806 |
| 011 rows restored, sole failure undrogued | 1311079 of 1400457 |
| kept-row matchup counts against the LWS rung, chunk 0, per lead | identical except 2 rows of 92745 at lead 5 |
| max relative sumsq difference against the LWS rung, chunk 0 | 3.2e-06 |

The two extra rows and the 3e-06 sumsq difference come from adding CURRENT_TEST to
the depth-interpolation group keys, which separates two coincident drifters that the
default grouping merged. Everything else is the LWS rung reproduced exactly.

Scoring: array job 34927 (chunk 0) and 34933 (chunks 1 to 12), partition monoproc,
13 chunks of 4 start dates, 214 to 227s per task, all exit 0.

## Verdict

The contamination signal is real but it does not follow the confidence digits. The
011 stratum, scored on the identical basis, is 0.01121 m/s worse in uv RMSD than 313
(+8.5 percent relative, bootstrap CI [+0.00697, +0.01590] over the 52 start dates),
and it is the worse stratum in seven of eight latitude bands, so drogue loss does
degrade the truth by a measurable amount. Among the codes the default policy keeps,
only 211 is unambiguously shifted toward the 011 end: +0.01863 m/s against 313, CI
excluding zero, which puts it at 1.8 times the 313-to-011 contamination span, on
0.36 percent of rows. 213 sits about halfway, +0.00626 with a CI excluding zero,
scale 0.56. 311 is not distinguishable from 313 globally (+0.00217, CI spanning
zero), and the two largest weak codes go the other way: 312 is 0.00790 m/s better
than 313 (CI excluding zero) and 212 is 0.00328 better (CI spanning zero). The
latitude bands show why the pooled numbers must be read with care: 213 is +0.038 at
-60 and -0.019 at +60, 212 is -0.031 at +60, so a large part of every pooled
difference is where and when each code samples, not how good its drogue is. The
wind slippage evidence points the same way against a confidence cut: if weak codes
were drogue-off in disguise they should carry the larger slip, and they do carry it
(312 p90 0.165, 212 p90 0.183 against 313 p90 0.082), yet those are exactly the
codes that score best, while 011 carries the smallest slip of all (p90 0.086) and
scores worst. Slip magnitude is therefore not a contamination proxy in this product.
A defensible cut is a narrow one: dropping 211 alone removes 0.36 percent of kept
rows, dropping 211 and 213 removes 3.05 percent, and both are supported by their own
bootstrap intervals. The wide cuts are not supported: keeping only 313 would discard
48.26 percent of kept rows and 48.31 percent of matchups, most of it code 312, which
scores better than 313 rather than worse.
