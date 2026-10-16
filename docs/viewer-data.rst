.. SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

.. _viewer-data-page:

========================
Building the viewer data
========================

The OceanBench viewer shows the forecast maps, the water column, the Class IV match-ups, the eddies and the score summaries of every challenger.
``oceanbench evaluate`` builds these files when it is given a local directory:

.. code-block:: bash

    oceanbench evaluate glonet.py --viewer-artifacts-dir viewer-data
    oceanbench evaluate glonet.py --region ibi --viewer-artifacts-dir viewer-data

The directory can also be set with the ``OCEANBENCH_VIEWER_ARTIFACTS_DIR`` environment variable, or with the ``viewer_artifacts_directory`` argument of ``oceanbench.evaluate_challenger``.
The viewer data is built after the scores file is written, in the same process and from the same opened challenger dataset, and the scores file is the same with or without it.
The Class IV files are written from the match-ups the Class IV scores are computed from.

Layout
******

The directory mirrors the viewer data prefix and can hold several challengers and regions.
Evaluating a challenger and region again replaces its files and its entries in the index files.

``datasets.json``, ``insights.json``, ``scores-summary.json``
    The index files, merged with what the directory already holds.
    ``datasets.json`` lists the challengers with maps, ``insights.json`` the insight files of every challenger and region, and ``scores-summary.json`` one row per gridded RMSD and Class IV score.
    A summary row reports the score of the scores file as ``mean`` and a 95% interval from a seeded bootstrap over the forecast start dates.

``<challenger>.zarr``, ``<challenger>.viewer-manifest.json``
    The map pyramid of the challenger and its manifest, global region only.

``<challenger>.columns.zarr``
    The water-column store, temperature and salinity on the 1 degree grid at every depth, global region only.

``insights/<challenger>/<region>/``
    ``class4-matchups.parquet`` (every Class IV match-up), ``eddies.json`` and one ``eddies-start-<date>-lead-<N>.json`` per start date and lead day (the detected mesoscale eddies), ``rmsd-by-depth.json`` (Class IV RMSD per depth bin, for variables observed at several depths), ``year-error-geography.json`` and ``year-rmsd-by-start.json`` (the yearly error map and the RMSD of every start date, official regions only).

Publishing
**********

The viewer data is not uploaded by ``oceanbench evaluate``.
To publish it, upload the content of the directory to ``dev/benchmark/rebuild-preview/viewer/data/``, keeping its layout.
Upload the index files last, so the viewer never lists files that are not there yet.

Costs
*****

The viewer data adds to the evaluation time and memory, mostly for the maps of a 1/12 degree challenger.
For GLO12 over a year, the viewer data takes about 3 hours more and about 54 GB: 46 GB of maps, 2.7 GB of water column and 5.6 GB of match-ups, eddies and yearly maps.
GLONET takes about 15 GB in total and a 1 degree challenger about 5.4 GB.
The scores summary recomputes the gridded scores one start date at a time, which roughly doubles the time of the gridded metrics.
The match-ups of a 1/12 degree challenger take about 20 GB of memory, so plan that much per challenger when evaluating several challengers in parallel with ``--max-workers``.
