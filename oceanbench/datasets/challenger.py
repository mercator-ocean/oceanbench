# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# flake8: noqa

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

import xarray
from oceanbench.core import challenger_datasets


def glo12() -> xarray.Dataset:
    """
    Open the GLO12 challenger dataset.

    Returns
    -------
    Dataset
        The Dataset containing GLO12 forecasts.

    >>> glo12()
    <xarray.Dataset> Size: 2TB
    Dimensions:             (first_day_datetime: 52, lead_day_index: 10, depth: 21,
                             latitude: 2041, longitude: 4320)
    Coordinates:
      * depth               (depth) float32 84B 0.494 47.37 ... 4.833e+03 5.275e+03
      * latitude            (latitude) float32 8kB -80.0 -79.92 ... 89.92 90.0
      * longitude           (longitude) float32 17kB -180.0 -179.9 ... 179.8 179.9
      * lead_day_index      (lead_day_index) int64 80B 0 1 2 3 4 5 6 7 8 9
      * first_day_datetime  (first_day_datetime) datetime64[us] 416B 2024-01-03 ....
    Data variables:
        so                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 385GB dask.array<chunksize=(1, 1, 1, 75, 4320), meta=np.ndarray>
        thetao              (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 385GB dask.array<chunksize=(1, 1, 1, 75, 4320), meta=np.ndarray>
        uo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 385GB dask.array<chunksize=(1, 1, 1, 75, 4320), meta=np.ndarray>
        vo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 385GB dask.array<chunksize=(1, 1, 1, 75, 4320), meta=np.ndarray>
        zos                 (first_day_datetime, lead_day_index, latitude, longitude) float32 18GB dask.array<chunksize=(1, 1, 75, 4320), meta=np.ndarray>
    Attributes:
        Conventions:  CF-1.8
        area:         Global
        contact:      https://marine.copernicus.eu/contact
        credit:       E.U. Copernicus Marine Service Information (CMEMS)
        institution:  Mercator Ocean International
        licence:      http://marine.copernicus.eu/services-portfolio/service-comm...
        producer:     CMEMS - Global Monitoring and Forecasting Centre
        references:   http://marine.copernicus.eu
        source:       MOI GLO12
        title:        daily mean fields from Global Ocean Physics Analysis and Fo...
    """

    return challenger_datasets.glo12()


def glo36v1() -> xarray.Dataset:
    """
    Open the GLO36V1 challenger dataset.

    Returns
    -------
    Dataset
        The Dataset containing GLO36V1 forecasts.

    >>> glo36v1()
    <xarray.Dataset> Size: 3TB
    Dimensions:             (first_day_datetime: 52, lead_day_index: 7, depth: 50,
                             lat: 2041, lon: 4320)
    Coordinates:
      * depth               (depth) float32 200B 0.494 1.541 ... 5.275e+03 5.728e+03
      * lat                 (lat) float32 8kB -80.0 -79.92 -79.83 ... 89.92 90.0
      * lead_day_index      (lead_day_index) int64 56B 0 1 2 3 4 5 6
      * lon                 (lon) float32 17kB -180.0 -179.9 -179.8 ... 179.8 179.9
      * first_day_datetime  (first_day_datetime) datetime64[us] 416B 2023-01-04 ...
    Data variables:
        so                  (first_day_datetime, lead_day_index, depth, lat, lon) float32 642GB dask.array<chunksize=(1, 1, 7, 256, 540), meta=np.ndarray>
        thetao              (first_day_datetime, lead_day_index, depth, lat, lon) float32 642GB dask.array<chunksize=(1, 1, 7, 256, 540), meta=np.ndarray>
        uo                  (first_day_datetime, lead_day_index, depth, lat, lon) float32 642GB dask.array<chunksize=(1, 1, 7, 256, 540), meta=np.ndarray>
        vo                  (first_day_datetime, lead_day_index, depth, lat, lon) float32 642GB dask.array<chunksize=(1, 1, 7, 256, 540), meta=np.ndarray>
        zos                 (first_day_datetime, lead_day_index, lat, lon) float32 13GB dask.array<chunksize=(1, 1, 256, 1080), meta=np.ndarray>
    """

    return challenger_datasets.glo36v1()


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
      * first_day_datetime  (first_day_datetime) datetime64[us] 416B 2024-01-03 ....
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


def xihe() -> xarray.Dataset:
    """
    Open the XiHe challenger dataset.

    Returns
    -------
    Dataset
        The Dataset containing XiHe forecasts.

    >>> xihe()
    <xarray.Dataset> Size: 2TB
    Dimensions:             (first_day_datetime: 52, lead_day_index: 10, depth: 23,
                             latitude: 2041, longitude: 4320)
    Coordinates:
      * depth               (depth) float64 184B 0.494 2.646 5.078 ... 541.1 643.6
      * latitude            (latitude) float32 8kB -80.0 -79.92 ... 89.92 90.0
      * longitude           (longitude) float32 17kB -180.0 -179.9 ... 179.8 179.9
      * lead_day_index      (lead_day_index) int64 80B 0 1 2 3 4 5 6 7 8 9
      * first_day_datetime  (first_day_datetime) datetime64[us] 416B 2024-01-03 ....
    Data variables:
        so                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 75, 4320), meta=np.ndarray>
        thetao              (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 75, 4320), meta=np.ndarray>
        uo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 75, 4320), meta=np.ndarray>
        vo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 75, 4320), meta=np.ndarray>
        zos                 (first_day_datetime, lead_day_index, latitude, longitude) float32 18GB dask.array<chunksize=(1, 1, 75, 4320), meta=np.ndarray>
    """

    return challenger_datasets.xihe()


def wenhai() -> xarray.Dataset:
    """
    Open the WenHai challenger dataset.

    Returns
    -------
    Dataset
        The Dataset containing WenHai forecasts.

    >>> wenhai()
    <xarray.Dataset> Size: 2TB
    Dimensions:             (first_day_datetime: 52, lead_day_index: 10, depth: 23,
                             latitude: 2041, longitude: 4320)
    Coordinates:
      * depth               (depth) float32 92B 0.494 2.646 5.078 ... 541.1 643.6
      * latitude            (latitude) float32 8kB -80.0 -79.92 ... 89.92 90.0
      * longitude           (longitude) float32 17kB -180.0 -179.9 ... 179.8 179.9
      * lead_day_index      (lead_day_index) int64 80B 0 1 2 3 4 5 6 7 8 9
      * first_day_datetime  (first_day_datetime) datetime64[us] 416B 2024-01-03 ....
    Data variables:
        so                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 75, 4320), meta=np.ndarray>
        thetao              (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 75, 4320), meta=np.ndarray>
        uo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 75, 4320), meta=np.ndarray>
        vo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 75, 4320), meta=np.ndarray>
        zos                 (first_day_datetime, lead_day_index, latitude, longitude) float32 18GB dask.array<chunksize=(1, 1, 75, 4320), meta=np.ndarray>
    """

    return challenger_datasets.wenhai()
