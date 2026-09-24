.. SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

.. _evaluation-methods-page:

===================================================
Definitions of evaluation methods
===================================================

Several methods are used to evaluate forecasting systems in OceanBench.
Each of them is applied to a dataset grouping 52 forecasts in the year 2024.

The following figure provides an overview of the evaluation methodology, illustrating the multifaceted evaluation strategy that captures different aspects of model performance.
This includes (i) observation-based intercomparison, (ii) reference-model benchmarking, and (iii) process-oriented diagnostics derived from physically meaningful variables.
Together, these components provide a holistic view of each model’s ability to reproduce observed ocean dynamics, maintain internal physical consistency, and generalize beyond the training regime.

.. image:: oceanbench-evaluation-overview.png

Reference datasets
**********************************************

OceanBench evaluates challengers against the following reference datasets:

- `2024 GLORYS reanalysis <https://data.marine.copernicus.eu/product/GLOBAL_MULTIYEAR_PHY_001_030>`_
- `2024 GLO12 analysis <https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024>`_
- 2024 in situ and satellite observations (Argo profiles, surface drifters, along track altimetry) from the Copernicus Marine Service, prepared as described in ``helper_scripts/observations2024_v2/README.md``

You can open and explore these datasets by using the :mod:`oceanbench.datasets.reference` module.

The OceanBench ocean mask says which cells of the twelfth of a degree grid OceanBench treats as ocean, at seven levels: the six standard depths listed below and 643.57 m, the first native level below 600 m, used to bracket the deepest Class IV observations. A cell is ocean when it is wet in both official Copernicus Marine static masks, the one of the GLO12 analysis and forecast product and the one of the GLORYS12 reanalysis product, so the scored population never depends on which reference a metric uses. The mask is built once from those two datasets and stored as an artefact pinned by the SHA256 checksum of its array bytes, which is verified every time the mask is loaded.

Class IV scores follow the IV-TT CLASS-4 framework (`Hernandez et al., 2009 <https://doi.org/10.5670/oceanog.2009.71>`_, `Ryan et al., 2015 <https://doi.org/10.1080/1755876X.2015.1022330>`_, `Divakaran et al., 2015 <https://doi.org/10.1080/1755876X.2015.1022333>`_): each forecast is compared with the observations at the observation time, position and depth, and the RMSD is reported per variable, depth bin and lead day. The observation selection and quality control applied when building the observation store are documented in the README above. Temperature and salinity are scored in depth bins down to 600 m. A challenger whose deepest level is shallower than 600 m is still scored on the full range, with its deepest level standing in for the missing depths. Class IV scores the same observations for every challenger and every reference, whatever their grids, and that population is built from the OceanBench ocean mask alone. An observation is dropped when one of its four surrounding twelfth of a degree cells is shallow, wet at the surface but dry at 92 m, unless one of those shallow cells belongs to a shallow region larger than 100,000 km². It is also dropped unless its four surrounding quarter degree cells, each wet only when its nine twelfth of a degree cells are, are wet at its deeper bracketing mask depth. A challenger value missing inside that population is counted in the ``Missing`` column, at the first lead day, and ``Observations`` is also counted at the first lead day.

For gridded RMSD metrics, OceanBench computes an area-weighted spatial mean of squared errors using ``cos(latitude)`` weights, so each grid cell contributes in proportion to the ocean area it represents rather than counting equally, ignoring missing land values during the weighted reduction, then averages the daily RMSE over forecast initialization days.
A cell is scored when the OceanBench ocean mask is wet there, taken at the nearest mask cell for challengers coarser than a twelfth of a degree, and both the challenger and the reference have a value; the ocean cells the challenger leaves empty are counted per variable and depth in the ``Missing`` and ``Missing fraction`` columns, which do not change the RMSD value.

Root Mean Square Deviation (RMSD) of variables compared to GLORYS reanalysis
**********************************************************************************************

The area-weighted (cos latitude) `Root Mean Square Deviation (RMSD) <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_ between the challenger dataset and the GLORYS reanalysis dataset, i.e., over all dataset variables.

Only 6 depths are used:

- Surface (~0.49 meters)
- 50 m (~47 meters)
- 100 m (~92 meters)
- 200 m (~223 meters)
- 300 m (~318 meters)
- 500 m (~541 meters)

Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS reanalysis
**********************************************************************************************

The area-weighted (cos latitude) `Root Mean Square Deviation (RMSD) <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_ between the two `Mixed Layer Depth (MLD) <https://en.wikipedia.org/wiki/Mixed_layer>`_ computations over the challenger dataset and the GLORYS reanalysis dataset.

The mixed layer depth is computed in meters on each dataset's native vertical grid using depth levels up to 600 meters with a density threshold of 0.03 kg/m³.
The reported value is one of the source depth levels, not an interpolated threshold-crossing depth.
If the threshold is not reached within the capped profile, OceanBench reports the deepest finite level available within the cap; in deep-water columns, mixed layers deeper than 600 meters are therefore reported as 600 meters.
This native-grid diagnostic preserves each system's represented vertical structure; vertical resolution therefore affects cross-challenger comparability.

Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS reanalysis
**********************************************************************************************

The area-weighted (cos latitude) `Root Mean Square Deviation (RMSD) <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_ between the two `geostrophic current <https://en.wikipedia.org/wiki/Geostrophic_current>`_ computations over the challenger datasets and the GLORYS reanalysis dataset.

The geostrophic currents are computed using sea surface height above geoid with Coriolis parameters Omega of 7.2921e-5, R of 6371000, and a gravity of 9.81 m/s². The Equator (latitude between -0.5° and 0.5°) is excluded.

Deviation of Lagrangian trajectories compared to GLORYS reanalysis
**********************************************************************************************

The deviation in kilometers between the two sets of drifting particles computed over the challenger datasets and the GLORYS reanalysis dataset.

The particles are seeded by sampling ocean grid points without replacement using ``cos(latitude)``-weighted probabilities, then simulated over the area.

Root Mean Square Deviation (RMSD) of variables compared to GLO12 analysis
**********************************************************************************************

The area-weighted (cos latitude) `Root Mean Square Deviation (RMSD) <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_ between the challenger dataset and the GLO12 analysis dataset, i.e., over all dataset variables.

Only 6 depths are used:

- Surface (~0.49 meters)
- 50 m (~47 meters)
- 100 m (~92 meters)
- 200 m (~223 meters)
- 300 m (~318 meters)
- 500 m (~541 meters)

Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO12 analysis
**********************************************************************************************

The area-weighted (cos latitude) `Root Mean Square Deviation (RMSD) <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_ between the two `Mixed Layer Depth (MLD) <https://en.wikipedia.org/wiki/Mixed_layer>`_ computations over the challenger dataset and the GLO12 analysis dataset.

The mixed layer depth is computed in meters on each dataset's native vertical grid using depth levels up to 600 meters with a density threshold of 0.03 kg/m³.
The reported value is one of the source depth levels, not an interpolated threshold-crossing depth.
If the threshold is not reached within the capped profile, OceanBench reports the deepest finite level available within the cap; in deep-water columns, mixed layers deeper than 600 meters are therefore reported as 600 meters.
This native-grid diagnostic preserves each system's represented vertical structure; vertical resolution therefore affects cross-challenger comparability.

Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLO12 analysis
**********************************************************************************************

The area-weighted (cos latitude) `Root Mean Square Deviation (RMSD) <https://en.wikipedia.org/wiki/Root_mean_square_deviation>`_ between the two `geostrophic current <https://en.wikipedia.org/wiki/Geostrophic_current>`_ computations over the challenger datasets and the GLO12 analysis dataset.

The geostrophic currents are computed using sea surface height above geoid with Coriolis parameters Omega of 7.2921e-5, R of 6371000, and a gravity of 9.81 m/s². The Equator (latitude between -0.5° and 0.5°) is excluded.

Deviation of Lagrangian trajectories compared to GLO12 analysis
**********************************************************************************************

The deviation in kilometers between the two sets of drifting particles computed over the challenger datasets and the GLO12 analysis dataset.

The particles are seeded by sampling ocean grid points without replacement using ``cos(latitude)``-weighted probabilities, then simulated over the area.
