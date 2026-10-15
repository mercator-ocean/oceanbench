# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# flake8: noqa

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

from datetime import datetime

import xarray
from oceanbench.core import challenger_datasets


def glo12(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """
    Open the GLO12 challenger dataset.

    Parameters
    ----------
    first_day_datetimes : list[datetime], optional
        Forecast start dates to open, all benchmark start dates by default.
    evaluation_year : int or str, optional
        Evaluation year used when ``first_day_datetimes`` is not provided.

    Returns
    -------
    Dataset
        The Dataset containing GLO12 forecasts.

    >>> glo12() # doctest: +ELLIPSIS
    <xarray.Dataset> Size: 4TB
    Dimensions:             (first_day_datetime: N, lead_day_index: 10, depth: 50,
                             latitude: 2041, longitude: 4320)
    Coordinates:
      * depth               (depth) float32 200B 0.494 1.541 ... 5.275e+03 5.728e+03
      * latitude            (latitude) float32 8kB -80.0 -79.92 ... 89.92 90.0
      * longitude           (longitude) float32 17kB -180.0 -179.9 ... 179.8 179.9
      * lead_day_index      (lead_day_index) int64 80B 0 1 2 3 4 5 6 7 8 9
      * first_day_datetime  (first_day_datetime) datetime64[...] 416B first Wednesday of the selected year ....
    Data variables:
        so                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 917GB dask.array<chunksize=(1, 1, 1, 640, 1280), meta=np.ndarray>
        thetao              (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 917GB dask.array<chunksize=(1, 1, 1, 640, 1280), meta=np.ndarray>
        uo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 917GB dask.array<chunksize=(1, 1, 1, 640, 1280), meta=np.ndarray>
        vo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 917GB dask.array<chunksize=(1, 1, 1, 640, 1280), meta=np.ndarray>
        zos                 (first_day_datetime, lead_day_index, latitude, longitude) float32 18GB dask.array<chunksize=(1, 1, 640, 1280), meta=np.ndarray>
    """

    return challenger_datasets.glo12(first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year)


def glo12_1_degree(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """
    Open the GLO12 challenger dataset interpolated to the 1 degree resolution.

    Parameters
    ----------
    first_day_datetimes : list[datetime], optional
        Forecast start dates to open, all benchmark start dates by default.
    evaluation_year : int or str, optional
        Evaluation year used when ``first_day_datetimes`` is not provided.

    Returns
    -------
    Dataset
        The Dataset containing GLO12 forecasts interpolated to 1 degree resolution.

    >>> glo12_1_degree() # doctest: +ELLIPSIS
    <xarray.Dataset> Size: 26GB
    Dimensions:                          (first_day_datetime: N,
                                          lead_day_index: 10, depth: 50,
                                          latitude: 170, longitude: 360)
    Coordinates:
      * depth                            (depth) float32 200B 0.494 ... 5.728e+03
      * lead_day_index                   (lead_day_index) int64 80B 0 1 2 ... 7 8 9
      * first_day_datetime               (first_day_datetime) datetime64[...] 416B ...
      * latitude                         (latitude) float64 1kB -79.5 -78.5 ... 89.5
      * longitude                        (longitude) float64 3kB -179.5 ... 179.5
    Data variables:
        sea_water_salinity               (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 6GB dask.array<chunksize=(1, 1, 1, 170, 360), meta=np.ndarray>
        sea_water_potential_temperature  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 6GB dask.array<chunksize=(1, 1, 1, 170, 360), meta=np.ndarray>
        eastward_sea_water_velocity      (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 6GB dask.array<chunksize=(1, 1, 1, 170, 360), meta=np.ndarray>
        northward_sea_water_velocity     (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 6GB dask.array<chunksize=(1, 1, 1, 170, 360), meta=np.ndarray>
        sea_surface_height_above_geoid   (first_day_datetime, lead_day_index, latitude, longitude) float32 127MB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
    """

    return challenger_datasets.glo12_1_degree(first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year)


def glo36v1(first_day_datetimes: list[datetime] | None = None) -> xarray.Dataset:
    """
    Open the GLO36V1 challenger dataset.

    Parameters
    ----------
    first_day_datetimes : list[datetime], optional
        Forecast start dates to open, all benchmark start dates by default.

    Returns
    -------
    Dataset
        The Dataset containing GLO36V1 forecasts.

    >>> glo36v1() # doctest: +ELLIPSIS
    <xarray.Dataset> Size: 3TB
    Dimensions:             (first_day_datetime: N, lead_day_index: 7, depth: 50,
                             latitude: 2041, longitude: 4320)
    Coordinates:
      * depth               (depth) float32 200B 0.494 1.541 ... 5.275e+03 5.728e+03
      * latitude            (latitude) float32 8kB -80.0 -79.92 ... 89.92 90.0
      * lead_day_index      (lead_day_index) int64 56B 0 1 2 3 4 5 6
      * longitude           (longitude) float32 17kB -180.0 -179.9 ... 179.8 179.9
      * first_day_datetime  (first_day_datetime) datetime64[...] 416B first Wednesday of the selected year ....
    Data variables:
        so                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 642GB dask.array<chunksize=(1, 1, 7, 256, 540), meta=np.ndarray>
        thetao              (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 642GB dask.array<chunksize=(1, 1, 7, 256, 540), meta=np.ndarray>
        uo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 642GB dask.array<chunksize=(1, 1, 7, 256, 540), meta=np.ndarray>
        vo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 642GB dask.array<chunksize=(1, 1, 7, 256, 540), meta=np.ndarray>
        zos                 (first_day_datetime, lead_day_index, latitude, longitude) float32 13GB dask.array<chunksize=(1, 1, 256, 1080), meta=np.ndarray>
    """

    return challenger_datasets.glo36v1(first_day_datetimes)


def glonet(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """
    Open the GLONET challenger dataset.

    Parameters
    ----------
    first_day_datetimes : list[datetime], optional
        Forecast start dates to open, all benchmark start dates by default.
    evaluation_year : int or str, optional
        Evaluation year used when ``first_day_datetimes`` is not provided.

    Returns
    -------
    Dataset
        The Dataset containing GLONET forecasts.

    >>> glonet() # doctest: +ELLIPSIS
    <xarray.Dataset> Size: 342GB
    Dimensions:             (first_day_datetime: N, lead_day_index: 10, depth: 21,
                             latitude: 672, longitude: 1440)
    Coordinates:
      * depth               (depth) float32 84B 0.494 47.37 ... 4.833e+03 5.275e+03
      * latitude            (latitude) float64 5kB -78.0 -77.75 -77.5 ... 89.5 89.75
      * longitude           (longitude) float64 12kB -180.0 -179.8 ... 179.5 179.8
      * lead_day_index      (lead_day_index) int64 80B 0 1 2 3 4 5 6 7 8 9
      * first_day_datetime  (first_day_datetime) datetime64[...] 416B first Wednesday of the selected year ....
    Data variables:
        so                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float64 85GB dask.array<chunksize=(1, 2, 3, 168, 360), meta=np.ndarray>
        thetao              (first_day_datetime, lead_day_index, depth, latitude, longitude) float64 85GB dask.array<chunksize=(1, 2, 3, 168, 360), meta=np.ndarray>
        uo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float64 85GB dask.array<chunksize=(1, 2, 3, 168, 360), meta=np.ndarray>
        vo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float64 85GB dask.array<chunksize=(1, 2, 3, 168, 360), meta=np.ndarray>
        zos                 (first_day_datetime, lead_day_index, latitude, longitude) float64 4GB dask.array<chunksize=(1, 2, 168, 360), meta=np.ndarray>
    Attributes:
        Conventions:              CF-1.8
        area:                     Global
        challenger:               glonet
        contact:                  glonet@mercator-ocean.eu
        forecast_reference_time:  day before the first Wednesday of the selected year
        institution:              Mercator Ocean International
        references:               www.edito.eu
        source:                   MOI GLONET
        title:                    Daily mean fields from GLONET 1/4 degree resolu...
    """

    return challenger_datasets.glonet(first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year)


def glonet_1_degree(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """
    Open the GLONET challenger dataset interpolated to the 1 degree resolution.

    Parameters
    ----------
    first_day_datetimes : list[datetime], optional
        Forecast start dates to open, all benchmark start dates by default.
    evaluation_year : int or str, optional
        Evaluation year used when ``first_day_datetimes`` is not provided.

    Returns
    -------
    Dataset
        The Dataset containing GLONET forecasts interpolated to 1 degree resolution.

    >>> glonet_1_degree() # doctest: +ELLIPSIS
    <xarray.Dataset> Size: 21GB
    Dimensions:                          (first_day_datetime: N,
                                          lead_day_index: 10, depth: 21,
                                          latitude: 168, longitude: 360)
    Coordinates:
      * depth                            (depth) float32 84B 0.494 ... 5.275e+03
      * lead_day_index                   (lead_day_index) int64 80B 0 1 2 ... 7 8 9
      * first_day_datetime               (first_day_datetime) datetime64[...] 416B ...
      * latitude                         (latitude) float64 1kB -77.5 -76.5 ... 89.5
      * longitude                        (longitude) float64 3kB -179.5 ... 179.5
    Data variables:
        sea_water_salinity               (first_day_datetime, lead_day_index, depth, latitude, longitude) float64 5GB dask.array<chunksize=(1, 2, 1, 168, 360), meta=np.ndarray>
        sea_water_potential_temperature  (first_day_datetime, lead_day_index, depth, latitude, longitude) float64 5GB dask.array<chunksize=(1, 2, 1, 168, 360), meta=np.ndarray>
        eastward_sea_water_velocity      (first_day_datetime, lead_day_index, depth, latitude, longitude) float64 5GB dask.array<chunksize=(1, 2, 1, 168, 360), meta=np.ndarray>
        northward_sea_water_velocity     (first_day_datetime, lead_day_index, depth, latitude, longitude) float64 5GB dask.array<chunksize=(1, 2, 1, 168, 360), meta=np.ndarray>
        sea_surface_height_above_geoid   (first_day_datetime, lead_day_index, latitude, longitude) float64 252MB dask.array<chunksize=(1, 2, 168, 360), meta=np.ndarray>
    Attributes:
        Conventions:              CF-1.8
        area:                     Global
        challenger:               glonet
        contact:                  glonet@mercator-ocean.eu
        forecast_reference_time:  day before the first Wednesday of the selected year
        institution:              Mercator Ocean International
        references:               www.edito.eu
        source:                   MOI GLONET
        title:                    Daily mean fields from GLONET 1/4 degree resolu...
    """

    return challenger_datasets.glonet_1_degree(first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year)


def langya(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """
    Open the LangYa challenger dataset.

    Parameters
    ----------
    first_day_datetimes : list[datetime], optional
        Forecast start dates to open, all benchmark start dates by default.
    evaluation_year : int or str, optional
        Evaluation year used when ``first_day_datetimes`` is not provided.

    Returns
    -------
    Dataset
        The Dataset containing LangYa forecasts.
    """

    return challenger_datasets.langya(first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year)


def langya_1_degree(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """
    Open the LangYa challenger dataset interpolated to the 1 degree resolution.

    Parameters
    ----------
    first_day_datetimes : list[datetime], optional
        Forecast start dates to open, all benchmark start dates by default.
    evaluation_year : int or str, optional
        Evaluation year used when ``first_day_datetimes`` is not provided.

    Returns
    -------
    Dataset
        The Dataset containing LangYa forecasts interpolated to 1 degree resolution.
    """

    return challenger_datasets.langya_1_degree(first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year)


def xihe(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """
    Open the XiHe challenger dataset.

    Parameters
    ----------
    first_day_datetimes : list[datetime], optional
        Forecast start dates to open, all benchmark start dates by default.
    evaluation_year : int or str, optional
        Evaluation year used when ``first_day_datetimes`` is not provided.

    Returns
    -------
    Dataset
        The Dataset containing XiHe forecasts.

    >>> xihe() # doctest: +ELLIPSIS
    <xarray.Dataset> Size: 2TB
    Dimensions:             (first_day_datetime: N, lead_day_index: 10, depth: 23,
                             latitude: 2041, longitude: 4320)
    Coordinates:
      * depth               (depth) float32 92B 0.494 2.646 5.078 ... 541.1 643.6
      * latitude            (latitude) float32 8kB -80.0 -79.92 ... 89.92 90.0
      * longitude           (longitude) float32 17kB -180.0 -179.9 ... 179.8 179.9
      * lead_day_index      (lead_day_index) int64 80B 0 1 2 3 4 5 6 7 8 9
      * first_day_datetime  (first_day_datetime) datetime64[...] 416B first Wednesday of the selected year ....
    Data variables:
        so                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 256, 512), meta=np.ndarray>
        thetao              (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 256, 512), meta=np.ndarray>
        uo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 256, 512), meta=np.ndarray>
        vo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 256, 512), meta=np.ndarray>
        zos                 (first_day_datetime, lead_day_index, latitude, longitude) float32 18GB dask.array<chunksize=(1, 1, 256, 512), meta=np.ndarray>
    Attributes:
        Conventions:              CF-1.8
        challenger:               xihe
        forecast_reference_time:  day before the first Wednesday of the selected year
    """

    return challenger_datasets.xihe(first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year)


def xihe_1_degree(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """
    Open the XiHe challenger dataset interpolated to the 1 degree resolution.

    Parameters
    ----------
    first_day_datetimes : list[datetime], optional
        Forecast start dates to open, all benchmark start dates by default.
    evaluation_year : int or str, optional
        Evaluation year used when ``first_day_datetimes`` is not provided.

    Returns
    -------
    Dataset
        The Dataset containing XiHe forecasts interpolated to 1 degree resolution.

    >>> xihe_1_degree() # doctest: +ELLIPSIS
    <xarray.Dataset> Size: 12GB
    Dimensions:                          (first_day_datetime: N,
                                          lead_day_index: 10, depth: 23,
                                          latitude: 170, longitude: 360)
    Coordinates:
      * depth                            (depth) float32 92B 0.494 2.646 ... 643.6
      * lead_day_index                   (lead_day_index) int64 80B 0 1 2 ... 7 8 9
      * first_day_datetime               (first_day_datetime) datetime64[...] 416B ...
      * latitude                         (latitude) float64 1kB -79.5 -78.5 ... 89.5
      * longitude                        (longitude) float64 3kB -179.5 ... 179.5
    Data variables:
        sea_water_salinity               (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 3GB dask.array<chunksize=(1, 1, 1, 170, 360), meta=np.ndarray>
        sea_water_potential_temperature  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 3GB dask.array<chunksize=(1, 1, 1, 170, 360), meta=np.ndarray>
        eastward_sea_water_velocity      (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 3GB dask.array<chunksize=(1, 1, 1, 170, 360), meta=np.ndarray>
        northward_sea_water_velocity     (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 3GB dask.array<chunksize=(1, 1, 1, 170, 360), meta=np.ndarray>
        sea_surface_height_above_geoid   (first_day_datetime, lead_day_index, latitude, longitude) float32 127MB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
    Attributes:
        Conventions:              CF-1.8
        challenger:               xihe
        forecast_reference_time:  day before the first Wednesday of the selected year
    """

    return challenger_datasets.xihe_1_degree(first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year)


def wenhai(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """
    Open the WenHai challenger dataset.

    Parameters
    ----------
    first_day_datetimes : list[datetime], optional
        Forecast start dates to open, all benchmark start dates by default.
    evaluation_year : int or str, optional
        Evaluation year used when ``first_day_datetimes`` is not provided.

    Returns
    -------
    Dataset
        The Dataset containing WenHai forecasts.

    >>> wenhai() # doctest: +ELLIPSIS
    <xarray.Dataset> Size: 2TB
    Dimensions:             (first_day_datetime: N, lead_day_index: 10, depth: 23,
                             latitude: 2041, longitude: 4320)
    Coordinates:
      * depth               (depth) float32 92B 0.494 2.646 5.078 ... 541.1 643.6
      * latitude            (latitude) float32 8kB -80.0 -79.92 ... 89.92 90.0
      * longitude           (longitude) float32 17kB -180.0 -179.9 ... 179.8 179.9
      * lead_day_index      (lead_day_index) int64 80B 0 1 2 3 4 5 6 7 8 9
      * first_day_datetime  (first_day_datetime) datetime64[...] 416B first Wednesday of the selected year ....
    Data variables:
        so                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 256, 512), meta=np.ndarray>
        thetao              (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 256, 512), meta=np.ndarray>
        uo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 256, 512), meta=np.ndarray>
        vo                  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 422GB dask.array<chunksize=(1, 1, 1, 256, 512), meta=np.ndarray>
        zos                 (first_day_datetime, lead_day_index, latitude, longitude) float32 18GB dask.array<chunksize=(1, 1, 256, 512), meta=np.ndarray>
    Attributes:
        Conventions:              CF-1.8
        challenger:               wenhai
        forecast_reference_time:  day before the first Wednesday of the selected year
    """

    return challenger_datasets.wenhai(first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year)


def wenhai_1_degree(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """
    Open the WenHai challenger dataset interpolated to the 1 degree resolution.

    Parameters
    ----------
    first_day_datetimes : list[datetime], optional
        Forecast start dates to open, all benchmark start dates by default.
    evaluation_year : int or str, optional
        Evaluation year used when ``first_day_datetimes`` is not provided.

    Returns
    -------
    Dataset
        The Dataset containing WenHai forecasts interpolated to 1 degree resolution.

    >>> wenhai_1_degree() # doctest: +ELLIPSIS
    <xarray.Dataset> Size: 12GB
    Dimensions:                          (first_day_datetime: N,
                                          lead_day_index: 10, depth: 23,
                                          latitude: 170, longitude: 360)
    Coordinates:
      * depth                            (depth) float32 92B 0.494 2.646 ... 643.6
      * lead_day_index                   (lead_day_index) int64 80B 0 1 2 ... 7 8 9
      * first_day_datetime               (first_day_datetime) datetime64[...] 416B ...
      * latitude                         (latitude) float64 1kB -79.5 -78.5 ... 89.5
      * longitude                        (longitude) float64 3kB -179.5 ... 179.5
    Data variables:
        sea_water_salinity               (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 3GB dask.array<chunksize=(1, 1, 1, 170, 360), meta=np.ndarray>
        sea_water_potential_temperature  (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 3GB dask.array<chunksize=(1, 1, 1, 170, 360), meta=np.ndarray>
        eastward_sea_water_velocity      (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 3GB dask.array<chunksize=(1, 1, 1, 170, 360), meta=np.ndarray>
        northward_sea_water_velocity     (first_day_datetime, lead_day_index, depth, latitude, longitude) float32 3GB dask.array<chunksize=(1, 1, 1, 170, 360), meta=np.ndarray>
        sea_surface_height_above_geoid   (first_day_datetime, lead_day_index, latitude, longitude) float32 127MB dask.array<chunksize=(1, 1, 170, 360), meta=np.ndarray>
    Attributes:
        Conventions:              CF-1.8
        challenger:               wenhai
        forecast_reference_time:  day before the first Wednesday of the selected year
    """

    return challenger_datasets.wenhai_1_degree(first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year)


def glo12_persistence(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """Open the GLO12 persistence challenger dataset."""
    return challenger_datasets.glo12_persistence(
        first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year
    )


def glo12_persistence_1_degree(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """Open the GLO12 persistence challenger at one-degree resolution."""
    return challenger_datasets.glo12_persistence_1_degree(
        first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year
    )


def hclimrep(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """Open the HCLIMREP challenger dataset."""
    return challenger_datasets.hclimrep(first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year)


def hclimrep_1_degree(
    first_day_datetimes: list[datetime] | None = None,
    evaluation_year: int | str | None = None,
) -> xarray.Dataset:
    """Open the HCLIMREP challenger at one-degree resolution."""
    return challenger_datasets.hclimrep_1_degree(
        first_day_datetimes=first_day_datetimes, evaluation_year=evaluation_year
    )
