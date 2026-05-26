.. SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

.. _one-degree-track-interpolation-page:

=====================================================
 1 degree track
=====================================================

This page documents how OceanBench constructs the 1 degree evaluation track.
It is intended to remove ambiguity for both benchmark users and technical reviewers.


Summary
*******


How reference datasets are selected for the 1 degree track

1 degree GLORYS dataset for training

How challenger datasets are interpolated


How reference datasets are selected for the 1 degree track
**********************************************************

For the 1 degree track, OceanBench uses dedicated 1 degree reference datasets for:

* GLORYS reanalysis
* GLO12 analysis

Then, when the challenger resolution is "1_degree", OceanBench opens the precomputed 1 degree references through:

* `oceanbench.core.references.glorys._glorys_reanalysis_dataset_1_degree  <https://oceanbench.readthedocs.io/en/latest/source/oceanbench.datasets.html#oceanbench.datasets.reference.glorys_reanalysis_1_degree>`_
* `oceanbench.core.references.glo12._glo12_analysis_dataset_1_degree <https://oceanbench.readthedocs.io/en/latest/source/oceanbench.datasets.html#oceanbench.datasets.challenger.glo12_1_degree>`_

1 degree GLORYS dataset for training
***********************************************

For training outside the official evaluation workflow, OceanBench also exposes the 1 degree GLORYS dataset through:

* ``oceanbench.datasets.reference.glorys_reanalysis_1_degree()``

This public API is documented in:

* `oceanbench.datasets.reference <https://oceanbench.readthedocs.io/en/latest/source/oceanbench.datasets.html#module-oceanbench.datasets.reference.glorys_reanalysis_1_degree>`_


How challenger datasets are interpolated
****************************************

The 1 degree challengers exposed in ``oceanbench.datasets.challenger`` are:

* ``glo12_1_degree()``
* ``glonet_1_degree()``
* ``wenhai_1_degree()``
* ``xihe_1_degree()``

Those are base challengers on a higher resolution (1/4 degree for GLONET, 1/12 degree for the others) that are interpolated to the 1 degree resolution.
These functions delegate to the core challenger dataset loaders:

* `oceanbench.core.challenger_datasets <https://oceanbench.readthedocs.io/en/latest/source/oceanbench.datasets.html#module-oceanbench.datasets>`_

Each of them applies the same interpolation logic:

1. Rename dataset dimensions and variables to the OceanBench standard names.
2. Infer the target 1 degree latitude and longitude grid from the native domain bounds.
3. Build a regular grid with 1.0 degree spacing and cell centres at ``n + 0.5``.
4. Interpolate the dataset with ``xarray.Dataset.interp``.
5. Mark the resulting dataset source with ``resolution=\"one_degree\"`` so downstream staging and caching logic can distinguish it from the native-resolution challenger.
