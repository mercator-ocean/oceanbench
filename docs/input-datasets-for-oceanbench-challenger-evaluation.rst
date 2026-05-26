.. SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

.. _input-datasets-for-oceanbench-challenger-evaluation-page:

=============================================================
 Input datasets for OceanBench challenger evaluation
=============================================================

OceanBench exposes the input datasets needed by challengers through the
``oceanbench.datasets.input`` Python API.

For people familiar with Python, `xarray <https://docs.xarray.dev/en/stable/index.html>`_
and `dask <https://www.dask.org/>`_, the fastest way to explore the inputs is
to open them directly:

.. code-block:: python

   import oceanbench

   glo12_nowcasts = oceanbench.datasets.input.glo12_nowcasts()
   ifs_forcings = oceanbench.datasets.input.ifs_forcings()

Available Datasets
**********************************************

``oceanbench.datasets.input.glo12_nowcasts()``
    Weekly GLO12 nowcasts from January 4, 2023 to December 31, 2025.

``oceanbench.datasets.input.ifs_forcings()``
    Weekly IFS forcings from January 3, 2023 to December 30, 2025.

These datasets are backed by weekly Zarr stores. Their storage layout is an
implementation detail of the API and may change, but the current source files
are named like:

- ``glo12_rg_1d-m_nwct_RYYYYMMDD.zarr`` for GLO12 nowcasts.
- ``ifs_forcing_rg_forecasts_RYYYYMMDD.zarr`` for IFS forcings.

Dataset Structure
**********************************************

.. list-table::
   :header-rows: 1
   :widths: 20,25,25,30

   * - Dataset
     - Temporal dimensions
     - Spatial dimensions
     - Variables
   * - GLO12 nowcasts
     - ``time``
     - ``depth``, ``latitude``, ``longitude``
     - ``siconc``, ``sithick``, ``so``, ``thetao``, ``uo``, ``usi``, ``vo``, ``vsi``, ``zos``
   * - IFS forcings
     - ``first_day_datetime``, ``lead_day_index``
     - ``lat``, ``lon``
     - ``cp``, ``ewss``, ``nsss``, ``skt``, ``sohumspe``, ``somslpre``, ``sosnowfa``, ``sosudolw``,
       ``sosudosw``, ``sotemair``, ``sotemhum``, ``sowaprec``, ``sowinu10``, ``sowinv10``, ``sp``

The datasets are opened lazily with dask-backed arrays, so creating the
``xarray.Dataset`` objects reads metadata but does not load the full data into
memory.

For more see
`oceanbench.datasets.input <https://oceanbench.readthedocs.io/en/latest/source/oceanbench.datasets.html#module-oceanbench.datasets.input>`_.
