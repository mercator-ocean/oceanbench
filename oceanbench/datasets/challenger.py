# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# flake8: noqa

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

import xarray
from oceanbench.core import challenger_datasets


def glonet() -> xarray.Dataset:
    """
    Open the GLONET challenger dataset.

    Returns
    -------
    Dataset
        The Dataset containing GLONET forecasts.

    >>> glonet()
    <xarray.Dataset> Size: 342GB
    Dimensions:             (first_day_datetime: 52, lead_day_index: 10, depth: 21,
                             lat: 672, lon: 1440)
    Coordinates:
      * depth               (depth) float32 84B 0.494 47.37 ... 4.833e+03 5.275e+03
      * lat                 (lat) float64 5kB -78.0 -77.75 -77.5 ... 89.5 89.75
      * lon                 (lon) float64 12kB -180.0 -179.8 -179.5 ... 179.5 179.8
      * lead_day_index      (lead_day_index) int64 80B 0 1 2 3 4 5 6 7 8 9
      * first_day_datetime  (first_day_datetime) datetime64[ns] 416B 2024-01-03 ....
    Data variables:
        so                  (first_day_datetime, lead_day_index, depth, lat, lon) float64 85GB dask.array<chunksize=(1, 10, 1, 672, 1440), meta=np.ndarray>
        thetao              (first_day_datetime, lead_day_index, depth, lat, lon) float64 85GB dask.array<chunksize=(1, 10, 1, 672, 1440), meta=np.ndarray>
        uo                  (first_day_datetime, lead_day_index, depth, lat, lon) float64 85GB dask.array<chunksize=(1, 10, 1, 672, 1440), meta=np.ndarray>
        vo                  (first_day_datetime, lead_day_index, depth, lat, lon) float64 85GB dask.array<chunksize=(1, 10, 1, 672, 1440), meta=np.ndarray>
        zos                 (first_day_datetime, lead_day_index, lat, lon) float64 4GB dask.array<chunksize=(1, 10, 672, 1440), meta=np.ndarray>
    Attributes:
        regrid_method:  bilinear
    """

    return challenger_datasets.glonet()
