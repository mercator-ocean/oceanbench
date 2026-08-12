<!--
SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>

SPDX-License-Identifier: EUPL-1.2
-->

# Old basis versus new basis, class-4 RMSD

OLD = legacy observations2024. NEW = observations2024-v2 at 2024-v2.1.0
(FILTR minus wind slippage, current_test 11 and 211 dropped).
RMSD pooled over depth bins and leads unless a lead is named.
Percent is the change from OLD to NEW; negative means the new basis scores better.

Chunks present per challenger, rung and region:

| challenger | rung | region | chunks_present |
| --- | --- | --- | --- |
| glonet | NEW | global | 13 |
| glonet | NEW | ibi | 13 |
| glonet | OLD | global | 13 |
| glonet | OLD | ibi | 13 |

## region global

### pooled over all leads

| challenger | stream | OLD rmsd | NEW rmsd | delta | % | n OLD | n NEW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glonet | temperature | 0.87216 | 0.87319 | +0.00103 | +0.12% | 63612236 | 63076748 |
| glonet | salinity | 0.14741 | 0.14669 | -0.00073 | -0.49% | 52649339 | 52189847 |
| glonet | sla | 0.05987 | 0.05993 | +0.00006 | +0.10% | 154335529 | 154335811 |
| glonet | uo | 0.21205 | 0.13268 | -0.07937 | -37.43% | 14047004 | 11974208 |
| glonet | vo | 0.19594 | 0.12849 | -0.06745 | -34.43% | 14047004 | 11974208 |

### per lead

| challenger | stream | L1 OLD | L1 NEW | L1 % | L5 OLD | L5 NEW | L5 % | L9 OLD | L9 NEW | L9 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glonet | temperature | 0.81293 | 0.81347 | +0.07% | 0.86585 | 0.86705 | +0.14% | 0.93465 | 0.93631 | +0.18% |
| glonet | salinity | 0.13741 | 0.13666 | -0.54% | 0.14798 | 0.14730 | -0.46% | 0.15766 | 0.15732 | -0.22% |
| glonet | sla | 0.05188 | 0.05193 | +0.08% | 0.05750 | 0.05758 | +0.15% | 0.06754 | 0.06748 | -0.08% |
| glonet | uo | 0.20423 | 0.12221 | -40.16% | 0.21018 | 0.13153 | -37.42% | 0.21700 | 0.13955 | -35.69% |
| glonet | vo | 0.18710 | 0.11853 | -36.65% | 0.19609 | 0.12867 | -34.38% | 0.20110 | 0.13508 | -32.83% |

## region ibi

### pooled over all leads

| challenger | stream | OLD rmsd | NEW rmsd | delta | % | n OLD | n NEW |
| --- | --- | --- | --- | --- | --- | --- | --- |
| glonet | temperature | 0.79768 | 0.79832 | +0.00064 | +0.08% | 1385612 | 1382590 |
| glonet | salinity | 0.18219 | 0.18239 | +0.00021 | +0.11% | 1181401 | 1178380 |
| glonet | sla | 0.05063 | 0.05061 | -0.00002 | -0.04% | 1791498 | 1817402 |
| glonet | uo | 0.20496 | 0.10305 | -0.10191 | -49.72% | 214906 | 183641 |
| glonet | vo | 0.17726 | 0.10229 | -0.07497 | -42.30% | 214906 | 183641 |

### per lead

| challenger | stream | L1 OLD | L1 NEW | L1 % | L5 OLD | L5 NEW | L5 % | L9 OLD | L9 NEW | L9 % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| glonet | temperature | 0.65979 | 0.65979 | +0.00% | 0.79282 | 0.79282 | +0.00% | 0.92430 | 0.92631 | +0.22% |
| glonet | salinity | 0.14375 | 0.14375 | +0.00% | 0.18764 | 0.18764 | +0.00% | 0.21735 | 0.21809 | +0.34% |
| glonet | sla | 0.04555 | 0.04545 | -0.22% | 0.04904 | 0.04905 | +0.04% | 0.05553 | 0.05547 | -0.11% |
| glonet | uo | 0.20177 | 0.09869 | -51.09% | 0.19436 | 0.10124 | -47.91% | 0.21140 | 0.10641 | -49.66% |
| glonet | vo | 0.17414 | 0.09666 | -44.49% | 0.17749 | 0.10183 | -42.63% | 0.18075 | 0.10845 | -40.00% |
