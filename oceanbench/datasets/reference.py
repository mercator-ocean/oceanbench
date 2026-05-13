# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# flake8: noqa

"""
This module exposes the reference datasets used in OceanBench for OceanBench challenger to explore.
"""

import pandas
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


def glorys_reanalysis_1_degree_historical() -> xarray.Dataset:
    """
    Open the historical 1 degree GLORYS reanalysis as an `xarray.Dataset`.

    This dataset is intended to support training on the 1 degree OceanBench track.
    It exposes the historical GLORYS reanalysis published at 1 degree resolution
    in the OceanBench public EDITO bucket.

    Returns
    -------
    Dataset
        The Dataset containing the historical 1 degree GLORYS reanalysis.

    >>> glorys_reanalysis_1_degree_historical() # doctest: +SKIP
    <xarray.Dataset> Size: 970GB
    Dimensions:                          (time: 9861, depth: 50, latitude: 170,
                                          longitude: 360)
    Coordinates:
      * time                             (time) datetime64[ns] 79kB 1993-01-01 .....
      * depth                            (depth) float32 200B 0.494 ... 5.728e+03
      * latitude                         (latitude) float64 1kB -79.5 -78.5 ... 89.5
      * longitude                        (longitude) float64 3kB -179.5 ... 179.5
    Data variables:
        eastward_sea_water_velocity      (time, depth, latitude, longitude) float64 241GB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
        northward_sea_water_velocity     (time, depth, latitude, longitude) float64 241GB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
        sea_surface_height_above_geoid   (time, latitude, longitude) float64 5GB dask.array<chunksize=(1, 170, 360), meta=np.ndarray>
        sea_water_potential_temperature  (time, depth, latitude, longitude) float64 241GB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
        sea_water_salinity               (time, depth, latitude, longitude) float64 241GB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
    Attributes:
        Conventions:  CF-1.4
        comment:      CMEMS product
        history:      2023/06/01 16:20:05 MERCATOR OCEAN Netcdf creation
        institution:  MERCATOR OCEAN
        references:   http://www.mercator-ocean.fr
        source:       MERCATOR GLORYS12V1
        title:        daily mean fields from Global Ocean Physics Analysis and Fo...
    """

    monthly_dates = pandas.date_range("1993-01-01", "2019-12-01", freq="MS")
    dataset_paths = [
        "https://minio.dive.edito.eu/project-oceanbench/public/"
        f"glorys_1degree_1993_2019/{monthly_date.strftime('%Y%m')}.zarr"
        for monthly_date in monthly_dates
    ]

    return xarray.open_mfdataset(
        dataset_paths,
        engine="zarr",
        parallel=False,
    )


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
