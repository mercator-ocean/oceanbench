# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# flake8: noqa

"""
This module exposes the reference datasets used in OceanBench for OceanBench challenger to explore.
"""

import xarray
from oceanbench.core.references import glorys
from oceanbench.core.references import glo12


def glorys_reanalysis() -> xarray.Dataset:
    """
    Open GLORYS reanalysis with as an `xarray.Dataset`.

    Returns
    -------
    Dataset
        The Dataset containing GLORYS reanalysis.

    >>> glorys_reanalysis() # doctest: +SKIP
    <xarray.Dataset> Size: 5TB
    Dimensions:    (depth: 50, latitude: 2041, longitude: 4320, time: 366)
    Coordinates:
      * depth      (depth) float32 200B 0.494 1.541 2.646 ... 5.275e+03 5.728e+03
      * latitude   (latitude) float32 8kB -80.0 -79.92 -79.83 ... 89.83 89.92 90.0
      * longitude  (longitude) float32 17kB -180.0 -179.9 -179.8 ... 179.8 179.9
      * time       (time) datetime64[ns] 3kB 2024-01-01 2024-01-02 ... 2024-12-31
    Data variables:
        thetao     (time, depth, latitude, longitude) float64 1TB dask.array<chunksize=(28, 1, 512, 2048), meta=np.ndarray>
        so         (time, depth, latitude, longitude) float64 1TB dask.array<chunksize=(28, 1, 512, 2048), meta=np.ndarray>
        uo         (time, depth, latitude, longitude) float64 1TB dask.array<chunksize=(28, 1, 512, 2048), meta=np.ndarray>
        vo         (time, depth, latitude, longitude) float64 1TB dask.array<chunksize=(28, 1, 512, 2048), meta=np.ndarray>
        zos        (time, latitude, longitude) float64 26GB dask.array<chunksize=(28, 512, 2048), meta=np.ndarray>
    Attributes:
        source:       MERCATOR GLORYS12V1
        institution:  MERCATOR OCEAN
        comment:      CMEMS product
        title:        daily mean fields from Global Ocean Physics Analysis and Fo...
        references:   http://www.mercator-ocean.fr
        history:      2023/06/01 16:20:05 MERCATOR OCEAN Netcdf creation
        Conventions:  CF-1.4
    """

    return glorys.glorys_reanalysis()


def glo12_analysis() -> xarray.Dataset:
    """
    Open GLO12 analysis with as an `xarray.Dataset`.

    Returns
    -------
    Dataset
        The Dataset containing GLO12 analysis.

    >>> glo12_analysis() # doctest: +SKIP
    <xarray.Dataset> Size: 3TB
    Dimensions:    (depth: 50, latitude: 2041, longitude: 4320, time: 366)
    Coordinates:
      * depth      (depth) float32 200B 0.494 1.541 2.646 ... 5.275e+03 5.728e+03
      * latitude   (latitude) float32 8kB -80.0 -79.92 -79.83 ... 89.83 89.92 90.0
      * longitude  (longitude) float32 17kB -180.0 -179.9 -179.8 ... 179.8 179.9
      * time       (time) datetime64[ns] 3kB 2024-01-01 2024-01-02 ... 2024-12-31
    Data variables:
        thetao     (time, depth, latitude, longitude) float32 645GB dask.array<chunksize=(21, 1, 512, 2048), meta=np.ndarray>
        so         (time, depth, latitude, longitude) float32 645GB dask.array<chunksize=(21, 1, 512, 2048), meta=np.ndarray>
        uo         (time, depth, latitude, longitude) float32 645GB dask.array<chunksize=(21, 1, 512, 2048), meta=np.ndarray>
        vo         (time, depth, latitude, longitude) float32 645GB dask.array<chunksize=(21, 1, 512, 2048), meta=np.ndarray>
        zos        (time, latitude, longitude) float32 13GB dask.array<chunksize=(21, 1024, 2048), meta=np.ndarray>
    Attributes:
        credit:       E.U. Copernicus Marine Service Information (CMEMS)
        contact:      https://marine.copernicus.eu/contact
        source:       MOI GLO12
        producer:     CMEMS - Global Monitoring and Forecasting Centre
        references:   http://marine.copernicus.eu
        Conventions:  CF-1.8
        title:        daily mean fields from Global Ocean Physics Analysis and Fo...
        institution:  Mercator Ocean International
    """
    return glo12.glo12_analysis()
