.. SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

.. _evaluation-methods-page:

===================================================
Definitions of evaluation methods
===================================================

Several methods are used to evaluate forecasting system in OceanBench.
Each of them is applied on the production of the forecasting over year 2024, namely the "challenger dataset".

Shape of the challenger dataset
******************************************

For people used to Python, `xarray <https://docs.xarray.dev/en/stable/index.html>`_ and `dask <https://www.dask.org/>`_, the fastest way to get an idea of the needed challenger datacube is to look at `this notebook <https://github.com/mercator-ocean/oceanbench/blob/main/assets/glonet_sample.report.ipynb>`_.

The challenger dataset must contains all 10-days forecasts starting the 52 Wednesdays of year 2024.

Hence, it must be a datacube with at least 5 dimensions and 5 variables as defined in the `Climate Forecast Convention (CF) <https://cfconventions.org>`_.

Dimensions:

- Latitude (standard grid)
- Longitude (standard grid)
- Depth (positive depth level in the ocean)
- Lead day index (from 0 to 9, corresponding to the 10 days of forecasts)
- First day datetime (datetime of the first day of forecast)

Variables:

- Sea surface height above geoid (over all dimensions except depth)
- Sea water potential temperature (over all dimensions)
- Sea water salinity (over all dimensions)
- Northward sea water velocity (over all dimensions)
- Eastward sea water velocity (over all dimensions)

The challenger dataset dimensions and variables must be named after the `Climate Forecast Convention (CF) standard names <https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html>`_ or have a `standard_name` attribute containing the corresponding CF standard name.

The challenger dataset should be opened as an `xarray.Dataset <https://xarray.pydata.org/en/v2023.11.0/generated/xarray.Dataset.html>`_, with explicit `dask chunks <https://docs.dask.org/en/stable/array-chunks.html>`_ for best performances.

Finally, OceanBench supports challenger dataset with 1/12° resolution or with 1/4° resolution.

Reference datasets
**********************************************

OceanBench evaluates challengers against the following reference datasets:

- `2024 GLORYS reanalysis <https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030>`_
- `2024 GLO12 analysis <https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024>`_

Root Mean Square Deviation (RMSD) of variables compared to GLORYS reanalysis
**********************************************************************************************

The `Root Mean Square Deviation (RMSD) <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_ between the challenger dataset and the GLORYS reanalysis dataset, i.e over all dataset variables.

Only 4 depths are used:

- Surface (~4.9 meters)
- 50 m (~4.7 meters)
- 200 m (~223 meters)
- 550 m (~541 meters)

Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS reanalysis
**********************************************************************************************

The `Root Mean Square Deviation (RMSD) <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_ between the two `Mixed Layer Depth (MLD) <https://en.wikipedia.org/wiki/Mixed_layer>`_ computations over the challenger dataset and the GLORYS reanalysis dataset.

The mixed layer depth is computed in meters using all dataset depth levels with a density threshold of 0.03 kg/m³.

Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS reanalysis
**********************************************************************************************

The `Root Mean Square Deviation (RMSD) <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_ between the two `geostrophic current <https://en.wikipedia.org/wiki/Geostrophic_current>`_ computations over the challenger datasets and the GLORYS reanalysis dataset.

The geostrophic currents are computed using sea surface height above geoid with Coriolis parameters Omega of 7.2921e-5, R of 6371000, and a gravity of 9.81 m/s². Equator (latitude betwenn -0.5° and 0.5°) is excluded.

Deviation of Lagrangian trajectories compared to GLORYS reanalysis
**********************************************************************************************

The deviation in kilometers between the two sets of drifting particles computed over the challenger datasets and the GLORYS reanalysis dataset.

The particles are simulated in a small squared area in the Atlantic joining Newyork coasts and Nouadhibou coasts.
