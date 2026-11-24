# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
from enum import Enum

import numpy

import xarray
import zarr

LATITUDE_NAMES = ("latitude", "lat")
LONGITUDE_NAMES = ("longitude", "lon")
ANTIMERIDIAN_LONGITUDE = 180.0
FULL_TURN_DEGREES = 360.0


@dataclass(frozen=True)
class SpatialExtent:
    west: float
    south: float
    east: float
    north: float


GLOBAL_SPATIAL_EXTENT = SpatialExtent(west=-180.0, south=-90.0, east=180.0, north=90.0)


@dataclass(frozen=True)
class Variable:
    name: str
    dimensions: tuple[str, ...]
    units: str | None
    long_name: str | None
    standard_name: str | None


class DimensionKind(Enum):
    LONGITUDE = "longitude"
    LATITUDE = "latitude"
    TEMPORAL = "temporal"
    OTHER = "other"


@dataclass(frozen=True)
class Dimension:
    name: str
    kind: DimensionKind
    extent: tuple[float | str | None, float | str | None]


@dataclass(frozen=True)
class StoreMetadata:
    spatial_extent: SpatialExtent
    variables: tuple[Variable, ...]
    groups: tuple[str, ...]
    dimensions: tuple[Dimension, ...] = ()


def store_metadata(url: str) -> StoreMetadata:
    groups = _groups(url)
    dataset = _opened_dataset(url, groups)
    return StoreMetadata(
        spatial_extent=_spatial_extent(dataset),
        variables=tuple(map(_variable, dataset.data_vars.values())),
        groups=groups,
        dimensions=tuple(_dimension(dataset, str(name)) for name in dataset.sizes),
    )


def _groups(url: str) -> tuple[str, ...]:
    return tuple(zarr.open_consolidated(url, mode="r").group_keys())


def _opened_dataset(url: str, groups: tuple[str, ...]) -> xarray.Dataset:
    match groups:
        case ():
            return xarray.open_zarr(url, consolidated=None)
        case _:
            return xarray.merge(
                [xarray.open_zarr(url, group=group) for group in groups], compat="override", join="override"
            )


def _coordinate_name(dataset: xarray.Dataset, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in dataset.coords), None)


def _spatial_extent(dataset: xarray.Dataset) -> SpatialExtent:
    latitude = _coordinate_name(dataset, LATITUDE_NAMES)
    longitude = _coordinate_name(dataset, LONGITUDE_NAMES)
    match (latitude, longitude):
        case (str(), str()):
            return SpatialExtent(
                west=float(_normalized_longitude(dataset[longitude]).min()),
                south=float(dataset[latitude].min()),
                east=float(_normalized_longitude(dataset[longitude]).max()),
                north=float(dataset[latitude].max()),
            )
        case _:
            return GLOBAL_SPATIAL_EXTENT


def _normalized_longitude(longitude: xarray.DataArray) -> numpy.ndarray:
    return (longitude.values + ANTIMERIDIAN_LONGITUDE) % FULL_TURN_DEGREES - ANTIMERIDIAN_LONGITUDE


def _dimension_kind(dataset: xarray.Dataset, name: str) -> DimensionKind:
    match name:
        case _ if name in LONGITUDE_NAMES:
            return DimensionKind.LONGITUDE
        case _ if name in LATITUDE_NAMES:
            return DimensionKind.LATITUDE
        case _ if name in dataset.coords and numpy.issubdtype(dataset[name].dtype, numpy.datetime64):
            return DimensionKind.TEMPORAL
        case _:
            return DimensionKind.OTHER


def _dimension(dataset: xarray.Dataset, name: str) -> Dimension:
    kind = _dimension_kind(dataset, name)
    match kind:
        case DimensionKind.LONGITUDE:
            longitude = _normalized_longitude(dataset[name])
            return Dimension(name, kind, (float(longitude.min()), float(longitude.max())))
        case DimensionKind.TEMPORAL:
            return Dimension(name, kind, (_iso_time(dataset[name].min()), _iso_time(dataset[name].max())))
        case _ if name in dataset.coords and numpy.issubdtype(dataset[name].dtype, numpy.number):
            return Dimension(name, kind, (float(dataset[name].min()), float(dataset[name].max())))
        case _:
            return Dimension(name, kind, (None, None))


def _iso_time(value: xarray.DataArray) -> str:
    return f"{numpy.datetime_as_string(value.values, unit='s')}Z"


def _variable(data_array: xarray.DataArray) -> Variable:
    return Variable(
        name=str(data_array.name),
        dimensions=tuple(map(str, data_array.dims)),
        units=data_array.attrs.get("units"),
        long_name=data_array.attrs.get("long_name"),
        standard_name=data_array.attrs.get("standard_name"),
    )
