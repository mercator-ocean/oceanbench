.. SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

.. _super-resolution-track-page:

=====================================================
 Super-resolution track
=====================================================

This page documents how OceanBench constructs the super-resolution evaluation track.
The track is evaluated only against the GLO36V1 reference and Class IV
observations. GLORYS reanalysis and GLO12 analysis are not used as references
for super-resolution challenger datasets.


GLO36V1 reference period
************************

The super-resolution track uses the GLO36V1 numerical model data published on EDITO:

* ``https://minio.dive.edito.eu/project-moi-glo36-oceanbench/public/``

The public bucket currently exposes weekly Zarr groups from ``20230104.zarr`` to
``20240103.zarr``. OceanBench therefore uses weekly ``first_day_datetime`` values
from 2023-01-04 to 2024-01-03, with 7 lead days per first day.

The corresponding public reference dataset helper is:

* :func:`oceanbench.datasets.reference.glo36v1_reference`


Observation scores
******************

The super-resolution track also keeps the Class IV observation metrics.
OceanBench observations are available from 2024-01-01. If a challenger starts
before that date, OceanBench keeps the challenger windows that overlap the
available observation period instead of dropping observation scoring entirely.


GLO36V1 comparison metrics
**************************

OceanBench exposes the following GLO36V1 comparison metrics:

* ``oceanbench.metrics.rmsd_of_variables_compared_to_glo36v1_reference()``
* ``oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glo36v1_reference()``
* ``oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glo36v1_reference()``
* ``oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glo36v1_reference()``

These metrics are computed only for super-resolution challenger datasets. The
GLONET high-resolution challenger loader uses the 2026 CloudFerro stream:

* ``https://s3.waw3-1.cloudferro.com/moiai-octo-bucket/public/octo/v0/ai-gallery/octo-glonet-hr-p1d/``

The observations used for 2026 challengers are read from:

* ``https://minio.dive.edito.eu/project-oceanbench/public/observations2026/``

Set ``OCEANBENCH_GLONET_HIGH_RESOLUTION_BASE_URI`` and
``OCEANBENCH_OBSERVATIONS_2026_BASE_URI`` to local directories or object-store
prefixes when running against LIR-mounted copies. The challenger loader
discovers dated GLONET high-resolution runs under that base URI.
