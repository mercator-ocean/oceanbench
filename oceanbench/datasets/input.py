# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# flake8: noqa

"""
This module exposes the input datasets for OceanBench challengers to produce forecast datasets
to evaluate in the benchmark.
"""

import xarray
from oceanbench.core import input_datasets


def glo12_nowcasts() -> xarray.Dataset:
    """
    Open weekly GLO12 nowcasts from 2023 to 2025.

    Returns
    -------
    Dataset
        The Dataset containing GLO12 nowcasts.

    >>> glo12_nowcasts() # doctest: +SKIP
    <xarray.Dataset> Size: 8TB
    Dimensions:    (time: 1099, latitude: 2041, longitude: 4320, depth: 50)
    Coordinates:
      * time       (time) datetime64[ns] 9kB 2022-12-28T12:00:00 ... 2025-...
      * latitude   (latitude) float32 8kB -80.0 -79.92 -79.83 ... 89.83 89.92 90.0
      * longitude  (longitude) float32 17kB -180.0 -179.9 -179.8 ... 179.8 179.9
      * depth      (depth) float32 200B 0.494 1.541 2.646 ... 5.275e+03 5.728e+03
    Data variables:
        siconc     (time, latitude, longitude) float32 39GB dask.array<chunksize=(1, 640, 1280), meta=np.ndarray>
        sithick    (time, latitude, longitude) float32 39GB dask.array<chunksize=(1, 640, 1280), meta=np.ndarray>
        so         (time, depth, latitude, longitude) float32 2TB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        thetao     (time, depth, latitude, longitude) float32 2TB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        uo         (time, depth, latitude, longitude) float32 2TB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        usi        (time, latitude, longitude) float32 39GB dask.array<chunksize=(1, 640, 1280), meta=np.ndarray>
        vo         (time, depth, latitude, longitude) float32 2TB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        vsi        (time, latitude, longitude) float32 39GB dask.array<chunksize=(1, 640, 1280), meta=np.ndarray>
        zos        (time, latitude, longitude) float32 39GB dask.array<chunksize=(1, 640, 1280), meta=np.ndarray>
    """

    return input_datasets.glo12_nowcasts()


def ifs_forcings() -> xarray.Dataset:
    """
    Open weekly IFS forcings from 2023 to 2025.

    Returns
    -------
    Dataset
        The Dataset containing IFS forcings.

    >>> ifs_forcings() # doctest: +SKIP
    <xarray.Dataset> Size: 1TB
    Dimensions:             (first_day_datetime: 157, lead_day_index: 10,
                             lat: 2560, lon: 5120)
    Coordinates:
      * first_day_datetime  (first_day_datetime) datetime64[ns] 1kB 2023-01-03 ... 2025-12-30
      * lead_day_index      (lead_day_index) int64 80B 0 1 2 3 4 5 6 7 8 9
      * lat                 (lat) float64 20kB 89.95 89.88 89.81 ... -89.88 -89.95
      * lon                 (lon) float64 41kB 0.0 0.07031 0.1406 ... 359.9 359.9
    Data variables:
        cp                  (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        ewss                (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        nsss                (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        skt                 (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        sohumspe            (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        somslpre            (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        sosnowfa            (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        sosudolw            (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        sosudosw            (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        sotemair            (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        sotemhum            (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        sowaprec            (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        sowinu10            (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        sowinv10            (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
        sp                  (first_day_datetime, lead_day_index, lat, lon) float32 82GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
    """

    return input_datasets.ifs_forcings()
