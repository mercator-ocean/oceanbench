.. SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

.. _input-data-conventions-page:

==============================================
Units and time conventions of the input data
==============================================

This page describes the IFS and GLO12 packages that ``oceanbench.datasets.input`` exposes:
what each package covers in time, and the units of each variable.

The script that produced the IFS packages is not available.
The IFS conventions below were established by comparing the packages with IFS operational 6-hourly analyses and forecasts,
on the same grid, for packages from late December 2023 and January 2024.
The mean sea level pressure was also compared with ERA5 daily means, to check which day each lead covers.
The GLO12 conventions come from the metadata of the stores.

IFS forecast packages
*********************

``oceanbench.datasets.input.ifs_forcings()`` opens one package per week, from January 3, 2023 to December 30, 2025.
Each package is a Zarr store with one group per variable:

.. code-block:: text

   https://s3.waw3-1.cloudferro.com/oceanbench-bucket/dev/additionnal-data/IFS/ifs_forcing_rg_forecasts_R{YYYYMMDD}.zarr

R is a Tuesday, the first day of the forecast.
A package has the dimensions ``first_day_datetime`` (1), ``lead_day_index`` (10), ``lat`` (2560) and ``lon`` (5120).
``first_day_datetime`` is day R.
``lead_day_index`` is a plain index with no date attached: lead ``k`` covers the UTC day R + k, from 00:00 to 24:00.
So package R20240102 covers January 2 to January 11, 2024.

The grid is the ECMWF F1280 regular Gaussian grid.
Latitudes go from 89.95 to -89.95 (north to south) and longitudes from 0 to 359.93.

Each value is a mean over its UTC day, except ``cp``, ``ewss`` and ``nsss`` (see the table).
For the instantaneous variables, the daily mean is consistent with an average of the forecast over the day.
Whether the forecast is sampled hourly or every 6 hours is not established.

.. list-table::
   :header-rows: 1
   :widths: 12 18 14 56

   * - Variable
     - ECMWF parameter
     - Units
     - Meaning
   * - ``somslpre``
     - ``msl`` (151)
     - Pa
     - Mean sea level pressure, daily mean.
   * - ``sp``
     - ``sp`` (134)
     - Pa
     - Surface pressure, daily mean.
   * - ``sotemair``
     - ``2t`` (167)
     - K
     - 2 m air temperature, daily mean.
   * - ``sotemhum``
     - ``2d`` (168)
     - K
     - 2 m dewpoint temperature, daily mean. This is a temperature, not a humidity.
   * - ``sohumspe``
     - ``2sh`` (174096)
     - kg kg-1
     - 2 m specific humidity, daily mean.
   * - ``skt``
     - ``skt`` (235)
     - K
     - Skin temperature, daily mean.
   * - ``sowinu10``
     - ``10u`` (165)
     - m s-1
     - 10 m eastward wind, daily mean.
   * - ``sowinv10``
     - ``10v`` (166)
     - m s-1
     - 10 m northward wind, daily mean.
   * - ``sosudosw``
     - ``ssrd`` (169)
     - W m-2
     - Surface downward shortwave radiation, daily mean flux.
   * - ``sosudolw``
     - ``strd`` (175)
     - W m-2
     - Surface downward longwave radiation, daily mean flux.
   * - ``ssr``
     - ``ssr`` (176)
     - W m-2
     - Surface net shortwave radiation, daily mean flux.
   * - ``strd``
     - ``strd`` (175)
     - W m-2
     - Surface downward longwave radiation, daily mean flux. Close to ``sosudolw``, see the note below.
   * - ``sowaprec``
     - ``tp`` (228)
     - kg m-2 s-1
     - Total precipitation, daily mean rate.
   * - ``sosnowfa``
     - ``sf`` (144)
     - kg m-2 s-1
     - Snowfall, in water equivalent, daily mean rate.
   * - ``cp``
     - ``cp`` (143)
     - m
     - Convective precipitation, accumulated since 00:00 UTC of day R. Not a rate: it grows with the lead day. The values are consistent with metres of water.
   * - ``ewss``
     - ``ewss`` (180)
     - unknown
     - Eastward surface stress. Do not use, under investigation.
   * - ``nsss``
     - ``nsss`` (181)
     - unknown
     - Northward surface stress. Do not use, under investigation.

The fluxes do not all come from the same forecast runs.
On the first lead day, ``sosudosw``, ``sosudolw``, ``sowaprec`` and ``sosnowfa`` match fluxes taken 6 hours at a time from the 00:00 and 12:00 UTC runs of that day.
``ssr`` and ``strd`` come from the 00:00 UTC run alone.
This is why ``strd`` and ``sosudolw`` differ slightly, by about 3 W m-2.

GLO12 nowcast packages
**********************

``oceanbench.datasets.input.glo12_nowcasts()`` opens one package per week, from January 4, 2023 to December 31, 2025,
and joins them along ``time``.
Each package is a Zarr store with one group per variable:

.. code-block:: text

   https://s3.waw3-1.cloudferro.com/oceanbench-bucket/dev/additionnal-data/GLO12/glo12_rg_1d-m_nwct_R{YYYYMMDD}.zarr

R is a Wednesday.
A package holds the 7 daily means before R, from day R - 7 to day R - 1.
So package R20240103 covers December 27, 2023 to January 2, 2024.

A package has the dimensions ``time`` (7), ``depth`` (50), ``latitude`` (2041) and ``longitude`` (4320).
The grid is regular at 1/12 degree, from -80 to 90 degrees north and from -180 to 179.92 degrees east.
Each time is 12:00 UTC of its day, the centre of the daily mean.
It is stored in hours since 1950-01-01.

Land cells hold the fill value ``9.96921e+36``, declared as ``_FillValue``.
xarray turns them into NaN when it opens the data.

.. list-table::
   :header-rows: 1
   :widths: 12 18 70

   * - Variable
     - Units
     - Meaning
   * - ``thetao``
     - °C
     - Sea water potential temperature, daily mean.
   * - ``so``
     - 1e-3
     - Sea water salinity, on the practical salinity scale, daily mean.
   * - ``uo``
     - m s-1
     - Eastward sea water velocity, daily mean.
   * - ``vo``
     - m s-1
     - Northward sea water velocity, daily mean.
   * - ``zos``
     - m
     - Sea surface height above the geoid, daily mean.
   * - ``siconc``
     - 1
     - Sea ice area fraction, daily mean.
   * - ``sithick``
     - m
     - Sea ice thickness, daily mean.
   * - ``usi``
     - m s-1
     - Eastward sea ice velocity, daily mean.
   * - ``vsi``
     - m s-1
     - Northward sea ice velocity, daily mean.
