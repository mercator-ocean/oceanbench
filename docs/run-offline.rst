.. SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

.. _run-offline-page:

=========================
Run OceanBench offline
=========================

The compute nodes of many HPC clusters have no internet access.
OceanBench can still evaluate a challenger there, using data downloaded beforehand on a node that has internet access.

Two steps
*********

The two nodes must share a file system, for example a scratch or work directory.

1. On a node with internet access, such as a login node, download the data the evaluation needs:

   .. code-block:: bash

      oceanbench stage path/to/challenger.py --stage-dir /shared/oceanbench-stage

   This command only downloads data. It computes no score, so it is light enough for a login node.

2. On the offline compute node, run the evaluation from the downloaded data:

   .. code-block:: bash

      oceanbench evaluate path/to/challenger.py --stage-dir /shared/oceanbench-stage

The evaluation finds everything in ``/shared/oceanbench-stage`` and does not download anything.
It also writes a few small working files there, so the directory must be writable from the compute node.

To evaluate a region, give both commands the same region, with ``--region``, ``--region-file`` or the ``OCEANBENCH_REGION`` environment variable.

What gets staged
****************

``oceanbench stage`` writes these datasets to the stage directory:

* the reference datasets, GLORYS reanalysis and GLO12 analysis
* the observations used by the Class IV scores
* the mean dynamic topography used to compare sea level with observations
* the OceanBench ocean mask
* the challenger forecasts, when the challenger file uses one of the challenger datasets provided by OceanBench

A challenger file that opens its own forecasts must read them from a disk the offline node can see.

The native 1/12 degree references are read with the Copernicus Marine Toolbox.
The node that runs step 1 therefore needs Copernicus Marine Service credentials, as described in the README.
The offline node does not need them.

The stage directory belongs to one challenger
*********************************************

The staged data depends on the challenger: its forecast dates, its number of lead days and its depths.
Step 1 must use the same challenger file as step 2.
If the challenger changes, run step 1 again so that the missing data gets downloaded.
Data already in the stage directory is not downloaded again.

Keep the stage directory
************************

A folder given with ``--stage-dir`` or ``OCEANBENCH_STAGE_DIR`` is kept.
You can evaluate the same challenger again without downloading everything again.
OceanBench only deletes its default temporary folder, used when no folder is given.
It deletes that folder after a successful ``oceanbench evaluate``, unless ``--keep-stage`` is given.

Environment variables
*********************

The same settings can be given as environment variables, for example in a batch script:

* ``OCEANBENCH_STAGE_DIR=/shared/oceanbench-stage`` is the same as ``--stage-dir``. On its own, it stages everything.
* ``OCEANBENCH_STAGE`` selects only some datasets for ``oceanbench evaluate``, like ``--stage``. Several values can be separated by commas, for example ``references,observations``. ``oceanbench stage`` always stages everything.
* ``OCEANBENCH_STAGE_MAX_WORKERS`` is the same as ``--stage-max-workers``, the number of parallel downloads.
* ``OCEANBENCH_REMOTE_RETRIES`` is the same as ``--remote-retries``, the number of attempts for each remote read.
* ``OCEANBENCH_REGION`` is the same as ``--region``.

Command line options take precedence over environment variables.
``--keep-stage`` has no environment variable and must be passed on the command line.
