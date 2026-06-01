.. SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

.. _one-degree-track-page:

=====================================================
 1 degree track aloooo
=====================================================

This page documents how OceanBench constructs the 1 degree evaluation track.
It is intended to remove ambiguity for both benchmark users and technical reviewers.
We will go through how reference datasets are selected for the 1 degree track, the 1 degree GLORYS dataset exposed for training and how challenger datasets are interpolated.


How reference datasets are selected for the 1 degree track
**********************************************************

For the 1 degree track, OceanBench uses dedicated 1 degree reference datasets for:

* GLORYS reanalysis
* GLO12 analysis

Then, when the challenger resolution is "1_degree", OceanBench opens the precomputed 1 degree references.
The related public dataset helpers are:

* :func:`oceanbench.datasets.reference.glorys_reanalysis_1_degree`
* :func:`oceanbench.datasets.reference.glo12_analysis`

1 degree GLORYS dataset for training
***********************************************

For training outside the official evaluation workflow, OceanBench also exposes the 1 degree GLORYS dataset through:

* ``oceanbench.datasets.reference.glorys_reanalysis_1_degree()``

This public API is documented in:

* :mod:`oceanbench.datasets.reference`


How challenger datasets are interpolated
****************************************

The 1 degree challengers exposed in ``oceanbench.datasets.challenger`` are:

* ``glo12_1_degree()``
* ``glonet_1_degree()``
* ``wenhai_1_degree()``
* ``xihe_1_degree()``

Those are base challengers on a higher resolution (1/4 degree for GLONET, 1/12 degree for the others) that are interpolated to the 1 degree resolution.
The corresponding public challenger dataset loaders are documented in:

* :mod:`oceanbench.datasets.challenger`

Each of them applies the same interpolation logic:

1. Rename dataset dimensions and variables to the OceanBench standard names.
2. Infer the target 1 degree latitude and longitude grid from the native domain bounds.
3. Build a regular grid with 1.0 degree spacing and cell centres at ``n + 0.5``.
4. Interpolate the dataset with ``xarray.Dataset.interp``.
5. Mark the resulting dataset source with ``resolution=\"one_degree\"`` so downstream staging and caching logic can distinguish it from the native-resolution challenger.
