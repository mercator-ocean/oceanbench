.. SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
..
.. SPDX-License-Identifier: EUPL-1.2

.. _input-datasets-for-oceanbench-challenger-evaluation-page:

=============================================================
 Input datasets for OceanBench challenger evaluation
=============================================================

OceanBench exposes the `input datasets <source/oceanbench.datasets.html#module-oceanbench.datasets.input>`_
needed by challengers through the ``oceanbench.datasets.input`` Python API.


Available Datasets
**********************************************

``oceanbench.datasets.input.glo12_nowcasts()``
    Weekly GLO12 nowcasts from January 4, 2023 to December 31, 2025.

    see more at: `glo12_nowcasts() <source/oceanbench.datasets.html#oceanbench.datasets.input.glo12_nowcasts>`_.

``oceanbench.datasets.input.ifs_forcings()``
    Weekly IFS forcings from January 3, 2023 to December 30, 2025.

    see more at: `ifs_forcings() <source/oceanbench.datasets.html#oceanbench.datasets.input.ifs_forcings>`_.

``oceanbench.datasets.input.ifs_nowcasts()``
    The 52 IFS nowcast files are available at
    ``s3://oceanbench-bucket/public/ifs-nowcasts24/`` and are named for the
    Tuesdays from January 2 to December 24, 2024. Each file is an independent
    Zarr dataset with dimensions ``(time, lat, lon)`` and five UTC timestamps:
    Tuesday 00/06/12/18Z and Wednesday 00Z, on the native ECMWF F1280 grid.

    The 15 variables preserve the IFS forcing names: instantaneous analyses
    ``sotemair``, ``sotemhum``, ``sohumspe``, ``sowinu10``, ``sowinv10``, ``skt``,
    ``somslpre`` and ``sp``; and already-desaccumulated six-hour means or rates
    ``sosudosw``, ``sosudolw``, ``sowaprec``, ``cp``, ``sosnowfa``, ``ewss`` and
    ``nsss``. The last timestamp is intentionally ``NaN`` for the latter
    variables because there is no following six-hour interval.

    The API concatenates the files along ``time`` without re-interpolation or
    daily aggregation. Daily aggregation must treat instantaneous and
    desaccumulated variables separately.

    see more at: `ifs_nowcasts() <source/oceanbench.datasets.html#oceanbench.datasets.input.ifs_nowcasts>`_.
