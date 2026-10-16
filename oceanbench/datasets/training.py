# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# flake8: noqa

"""
This module exposes the datasets to train ocean forecasting systems.
They cover the years 1993 to 2019, and never the year evaluated by the benchmark.
"""

import xarray
from oceanbench.core import training_datasets


def glorys_reanalysis() -> xarray.Dataset:
    """
    Open the 1/12 degree GLORYS reanalysis from 1993 to 2019 (included) as an `xarray.Dataset`.

    It is opened lazily with the Copernicus Marine Toolbox, and therefore requires
    authentication with the Copernicus Marine Service.

    Returns
    -------
    Dataset
        The Dataset containing the GLORYS reanalysis from 1993-01-01 to 2019-12-31.

    Examples
    --------
    >>> glorys_reanalysis() # doctest: +SKIP
    <xarray.Dataset> Size: 140TB
    Dimensions:    (depth: 50, latitude: 2041, longitude: 4320, time: 9861)
    Coordinates:
      * depth      (depth) float32 200B 0.494 1.541 2.646 ... 5.275e+03 5.728e+03
      * latitude   (latitude) float32 8kB -80.0 -79.92 -79.83 ... 89.83 89.92 90.0
      * longitude  (longitude) float32 17kB -180.0 -179.9 -179.8 ... 179.8 179.9
      * time       (time) datetime64[ns] 79kB 1993-01-01 1993-01-02 ... 2019-12-31
    Data variables:
        thetao     (time, depth, latitude, longitude) float64 35TB dask.array<chunksize=(50, 1, 512, 2048), meta=np.ndarray>
        so         (time, depth, latitude, longitude) float64 35TB dask.array<chunksize=(50, 1, 512, 2048), meta=np.ndarray>
        uo         (time, depth, latitude, longitude) float64 35TB dask.array<chunksize=(50, 1, 512, 2048), meta=np.ndarray>
        vo         (time, depth, latitude, longitude) float64 35TB dask.array<chunksize=(50, 1, 512, 2048), meta=np.ndarray>
        zos        (time, latitude, longitude) float64 696GB dask.array<chunksize=(50, 512, 2048), meta=np.ndarray>
    Attributes:
        ...
    """

    return training_datasets.glorys_reanalysis()


def glorys_reanalysis_1_degree() -> xarray.Dataset:
    """
    Open the 1 degree GLORYS reanalysis from 1993 to 2019 (included) as an `xarray.Dataset`.

    This dataset is intended to support training on the 1 degree OceanBench track.
    It exposes the historical GLORYS reanalysis published at 1 degree resolution
    in the OceanBench public EDITO bucket.

    Returns
    -------
    Dataset
        The Dataset containing the 1 degree GLORYS reanalysis from 1993-01-01 to 2019-12-31.

    Examples
    --------
    >>> glorys_reanalysis_1_degree() # doctest: +SKIP
    <xarray.Dataset> Size: 970GB
    Dimensions:                          (time: 9861, depth: 50, latitude: 170,
                                          longitude: 360)
    Coordinates:
      * depth                            (depth) float32 200B 0.494 ... 5.728e+03
      * latitude                         (latitude) float64 1kB -79.5 -78.5 ... 89.5
      * longitude                        (longitude) float64 3kB -179.5 ... 179.5
      * time                             (time) datetime64[ns] 79kB 1993-01-01 .....
    Data variables:
        eastward_sea_water_velocity      (time, depth, latitude, longitude) float64 241GB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
        northward_sea_water_velocity     (time, depth, latitude, longitude) float64 241GB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
        sea_surface_height_above_geoid   (time, latitude, longitude) float64 5GB dask.array<chunksize=(1, 170, 360), meta=np.ndarray>
        sea_water_potential_temperature  (time, depth, latitude, longitude) float64 241GB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
        sea_water_salinity               (time, depth, latitude, longitude) float64 241GB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
    Attributes:
        ...
    """

    return training_datasets.glorys_reanalysis_1_degree()
