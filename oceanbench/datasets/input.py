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
    Open GLO12 nowcasts.

    Returns
    -------
    Dataset
        The Dataset containing GLO12 nowcasts.

    >>> glo12_nowcasts()
    <xarray.Dataset> Size: 392GB
    Dimensions:    (time: 52, latitude: 2041, longitude: 4320, depth: 50)
    Coordinates:
      * depth      (depth) float32 200B 0.494 1.541 2.646 ... 5.275e+03 5.728e+03
      * latitude   (latitude) float32 8kB -80.0 -79.92 -79.83 ... 89.83 89.92 90.0
      * longitude  (longitude) float32 17kB -180.0 -179.9 -179.8 ... 179.8 179.9
      * time       (time) datetime64[ns] 416B 2024-01-02T12:00:00 ... 2024-12-24T...
    Data variables: (12/18)
        ist        (time, latitude, longitude) float32 2GB dask.array<chunksize=(1, 256, 1080), meta=np.ndarray>
        mlotst     (time, latitude, longitude) float32 2GB dask.array<chunksize=(1, 256, 1080), meta=np.ndarray>
        pbo        (time, latitude, longitude) float32 2GB dask.array<chunksize=(1, 256, 1080), meta=np.ndarray>
        siage      (time, latitude, longitude) float32 2GB dask.array<chunksize=(1, 256, 1080), meta=np.ndarray>
        sialb      (time, latitude, longitude) float32 2GB dask.array<chunksize=(1, 256, 1080), meta=np.ndarray>
        siconc     (time, latitude, longitude) float32 2GB dask.array<chunksize=(1, 256, 1080), meta=np.ndarray>
        ...         ...
        tob        (time, latitude, longitude) float32 2GB dask.array<chunksize=(1, 256, 1080), meta=np.ndarray>
        uo         (time, depth, latitude, longitude) float32 92GB dask.array<chunksize=(1, 4, 256, 540), meta=np.ndarray>
        usi        (time, latitude, longitude) float32 2GB dask.array<chunksize=(1, 256, 1080), meta=np.ndarray>
        vo         (time, depth, latitude, longitude) float32 92GB dask.array<chunksize=(1, 4, 256, 540), meta=np.ndarray>
        vsi        (time, latitude, longitude) float32 2GB dask.array<chunksize=(1, 256, 1080), meta=np.ndarray>
        zos        (time, latitude, longitude) float32 2GB dask.array<chunksize=(1, 256, 1080), meta=np.ndarray>
    Attributes: (12/13)
        Conventions:                CF-1.8
        NCO:                        netCDF Operators version 4.9.7 (Homepage = ht...
        area:                       Global
        contact:                    https://marine.copernicus.eu/contact
        credit:                     E.U. Copernicus Marine Service Information (C...
        history:                    Mon Mar 31 11:20:47 2025: ncks -A /data/rd_ex...
        ...                         ...
        institution:                Mercator Ocean International
        licence:                    http://marine.copernicus.eu/services-portfoli...
        producer:                   CMEMS - Global Monitoring and Forecasting Centre
        references:                 http://marine.copernicus.eu
        source:                     MOI GLO12
        title:                      daily mean fields from Global Ocean Physics A...
    """

    return input_datasets.glo12_nowcasts()


def ifs_forcings() -> xarray.Dataset:
    """
    Open IFS forcings.

    Returns
    -------
    Dataset
        The Dataset containing IFS forcings.

    >>> ifs_forcings()
    True
    """

    return input_datasets.ifs_forcings()
