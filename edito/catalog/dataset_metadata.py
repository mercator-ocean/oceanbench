# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import threading
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import reduce

import copernicusmarine
import numpy
import pandas
import xarray
import zarr

from bucket import BucketObject, ObjectKind, public_url
from oceanbench.core.remote_http import open_remote_zarr, remote_zarr_store

NETCDF_SUFFIX = ".nc"
NETCDF_BYTE_RANGE_MODE = "#mode=bytes"
COPERNICUS_MARINE_ARCO_SERVICE = "arco-geo-series"
COPERNICUS_MARINE_TIME_UNIT_MILLISECONDS = "ms"
LONGITUDE_NAMES = ("longitude", "lon", "nav_lon")
LATITUDE_NAMES = ("latitude", "lat", "nav_lat")
DEPTH_NAMES = ("depth", "deptht", "lev")
TIME_NAMES = ("time", "first_day_datetime", "time_ns", "juld")
DATETIME_OR_TEXT_KINDS = ("M", "U", "S", "O")
FIRST_DAY_DATETIME = "first_day_datetime"
LEAD_DAY_INDEX = "lead_day_index"
MAXIMUM_LONGITUDE = 180.0
_NETCDF_LIBRARY_LOCK = threading.Lock()


class DimensionKind(Enum):
    LONGITUDE = "longitude"
    LATITUDE = "latitude"
    DEPTH = "depth"
    TIME = "time"
    OTHER = "other"


@dataclass(frozen=True)
class SpatialExtent:
    west: float
    south: float
    east: float
    north: float


@dataclass(frozen=True)
class TemporalExtent:
    start: datetime
    end: datetime


@dataclass(frozen=True)
class Dimension:
    name: str
    kind: DimensionKind
    size: int
    minimum: float | str | None
    maximum: float | str | None
    step: float | None


@dataclass(frozen=True)
class Variable:
    name: str
    standard_name: str | None
    long_name: str | None
    units: str | None
    dimensions: tuple[str, ...]


@dataclass(frozen=True)
class DatasetMetadata:
    spatial_extent: SpatialExtent | None
    temporal_extent: TemporalExtent | None
    dimensions: tuple[Dimension, ...]
    variables: tuple[Variable, ...]
    groups: tuple[str, ...]


class TimeDecoding(Enum):
    DECODED = "decoded"
    RAW = "raw"


class UnavailabilityCause(Enum):
    UNREADABLE_FORMAT = "unreadable_format"
    READ_FAILURE = "read_failure"


@dataclass(frozen=True)
class MetadataUnavailable:
    cause: UnavailabilityCause
    reason: str


@dataclass(frozen=True)
class CopernicusMarineDataset:
    dataset_identifier: str
    product_identifier: str
    product_title: str
    product_description: str
    digital_object_identifier: str | None
    thumbnail_url: str | None
    arco_url: str | None
    metadata: DatasetMetadata


def bucket_objects_metadata(bucket_objects: tuple[BucketObject, ...]) -> DatasetMetadata | MetadataUnavailable:
    first_and_last_objects = tuple(dict.fromkeys((bucket_objects[0], bucket_objects[-1])))
    metadata = tuple(map(_bucket_object_metadata, first_and_last_objects))
    unavailable = next((result for result in metadata if isinstance(result, MetadataUnavailable)), None)
    return unavailable if unavailable is not None else reduce(merged_metadata, metadata)


def merged_metadata(first: DatasetMetadata, last: DatasetMetadata) -> DatasetMetadata:
    return replace(
        last,
        spatial_extent=_merged_spatial_extent(first.spatial_extent, last.spatial_extent),
        temporal_extent=_merged_temporal_extent(first.temporal_extent, last.temporal_extent),
    )


def merged_spatial_extents(extents: tuple[SpatialExtent | None, ...]) -> SpatialExtent | None:
    return reduce(_merged_spatial_extent, extents, None)


def merged_temporal_extents(extents: tuple[TemporalExtent | None, ...]) -> TemporalExtent | None:
    return reduce(_merged_temporal_extent, extents, None)


def _merged_spatial_extent(first: SpatialExtent | None, second: SpatialExtent | None) -> SpatialExtent | None:
    match (first, second):
        case (None, _):
            return second
        case (_, None):
            return first
        case _:
            return SpatialExtent(
                west=min(first.west, second.west),
                south=min(first.south, second.south),
                east=max(first.east, second.east),
                north=max(first.north, second.north),
            )


def _merged_temporal_extent(first: TemporalExtent | None, second: TemporalExtent | None) -> TemporalExtent | None:
    match (first, second):
        case (None, _):
            return second
        case (_, None):
            return first
        case _:
            return TemporalExtent(start=min(first.start, second.start), end=max(first.end, second.end))


def _bucket_object_metadata(bucket_object: BucketObject) -> DatasetMetadata | MetadataUnavailable:
    try:
        return _opened_bucket_object_metadata(bucket_object)
    except Exception as error:
        return MetadataUnavailable(
            UnavailabilityCause.READ_FAILURE, f"{bucket_object.key}: {type(error).__name__}: {error}"
        )


def _opened_bucket_object_metadata(bucket_object: BucketObject) -> DatasetMetadata | MetadataUnavailable:
    match bucket_object:
        case BucketObject(kind=ObjectKind.ZARR_STORE):
            return _zarr_store_metadata(public_url(bucket_object.key))
        case BucketObject(key=key) if key.endswith(NETCDF_SUFFIX):
            return _netcdf_file_metadata(public_url(key))
        case _:
            return MetadataUnavailable(
                UnavailabilityCause.UNREADABLE_FORMAT, f"{bucket_object.key}: format without readable metadata"
            )


def _zarr_store_metadata(url: str) -> DatasetMetadata:
    try:
        return _decoded_zarr_store_metadata(url, TimeDecoding.DECODED)
    except (ValueError, OverflowError):
        return _decoded_zarr_store_metadata(url, TimeDecoding.RAW)


def _decoded_zarr_store_metadata(url: str, time_decoding: TimeDecoding) -> DatasetMetadata:
    decode_times = time_decoding == TimeDecoding.DECODED
    groups = tuple(sorted(zarr.open_consolidated(remote_zarr_store(url), mode="r").group_keys()))
    datasets = (
        tuple(open_remote_zarr(url, group=group, chunks={}, decode_times=decode_times) for group in groups)
        if groups
        else (open_remote_zarr(url, chunks={}, decode_times=decode_times),)
    )
    return replace(reduce(_combined_metadata, (dataset_metadata(dataset, ()) for dataset in datasets)), groups=groups)


def _combined_metadata(first: DatasetMetadata, second: DatasetMetadata) -> DatasetMetadata:
    return DatasetMetadata(
        spatial_extent=_merged_spatial_extent(first.spatial_extent, second.spatial_extent),
        temporal_extent=_merged_temporal_extent(first.temporal_extent, second.temporal_extent),
        dimensions=tuple({dimension.name: dimension for dimension in (*second.dimensions, *first.dimensions)}.values()),
        variables=tuple({variable.name: variable for variable in (*first.variables, *second.variables)}.values()),
        groups=(),
    )


def _netcdf_file_metadata(url: str) -> DatasetMetadata:
    with _NETCDF_LIBRARY_LOCK:
        with xarray.open_dataset(f"{url}{NETCDF_BYTE_RANGE_MODE}", engine="netcdf4") as dataset:
            return dataset_metadata(dataset, ())


def dataset_metadata(dataset: xarray.Dataset, groups: tuple[str, ...]) -> DatasetMetadata:
    coordinate_like_names = (*LONGITUDE_NAMES, *LATITUDE_NAMES, *TIME_NAMES)
    return DatasetMetadata(
        spatial_extent=_spatial_extent(dataset),
        temporal_extent=_temporal_extent(dataset),
        dimensions=tuple(_dimension(dataset, name, size) for name, size in dataset.sizes.items()),
        variables=tuple(
            _variable(name, dataset[name]) for name in dataset.data_vars if name not in coordinate_like_names
        ),
        groups=groups,
    )


def _first_present(dataset: xarray.Dataset, names: tuple[str, ...]) -> str | None:
    return next((name for name in names if name in dataset.variables), None)


def _spatial_extent(dataset: xarray.Dataset) -> SpatialExtent | None:
    longitude_name = _first_present(dataset, LONGITUDE_NAMES)
    latitude_name = _first_present(dataset, LATITUDE_NAMES)
    if longitude_name is None or latitude_name is None:
        return None
    west, east = _finite_range(dataset[longitude_name])
    south, north = _finite_range(dataset[latitude_name])
    return (
        SpatialExtent(west=west, south=south, east=east, north=north)
        if east <= MAXIMUM_LONGITUDE
        else SpatialExtent(west=-MAXIMUM_LONGITUDE, south=south, east=MAXIMUM_LONGITUDE, north=north)
    )


def _finite_range(data_array: xarray.DataArray) -> tuple[float, float]:
    return float(data_array.min(skipna=True).compute()), float(data_array.max(skipna=True).compute())


def _temporal_extent(dataset: xarray.Dataset) -> TemporalExtent | None:
    time_name = _first_present(dataset, TIME_NAMES)
    if time_name is None or dataset[time_name].dtype.kind not in DATETIME_OR_TEXT_KINDS:
        return None
    times = pandas.to_datetime(dataset[time_name].values.ravel(), errors="coerce")
    if times.isna().all():
        return None
    return TemporalExtent(
        start=_utc_datetime(times.min()),
        end=_utc_datetime(times.max()) + _lead_time(dataset, time_name),
    )


def _lead_time(dataset: xarray.Dataset, time_name: str) -> timedelta:
    return (
        timedelta(days=int(dataset[LEAD_DAY_INDEX].max()))
        if time_name == FIRST_DAY_DATETIME and LEAD_DAY_INDEX in dataset.variables
        else timedelta(0)
    )


def _utc_datetime(value: numpy.datetime64) -> datetime:
    return pandas.Timestamp(value).tz_localize(timezone.utc).to_pydatetime()


def _dimension_kind(name: str) -> DimensionKind:
    match name:
        case _ if name in LONGITUDE_NAMES:
            return DimensionKind.LONGITUDE
        case _ if name in LATITUDE_NAMES:
            return DimensionKind.LATITUDE
        case _ if name in DEPTH_NAMES:
            return DimensionKind.DEPTH
        case _ if name in TIME_NAMES:
            return DimensionKind.TIME
        case _:
            return DimensionKind.OTHER


def _dimension(dataset: xarray.Dataset, name: str, size: int) -> Dimension:
    coordinate = dataset.coords.get(name)
    if coordinate is None or coordinate.ndim != 1 or size == 0 or not _is_orderable(coordinate.dtype):
        return Dimension(name, _dimension_kind(name), size, None, None, None)
    values = coordinate.values
    return Dimension(
        name=name,
        kind=_dimension_kind(name),
        size=size,
        minimum=_serializable(numpy.nanmin(values)),
        maximum=_serializable(numpy.nanmax(values)),
        step=_regular_step(values),
    )


def _is_orderable(dtype: numpy.dtype) -> bool:
    return numpy.issubdtype(dtype, numpy.number) or numpy.issubdtype(dtype, numpy.datetime64)


def _serializable(value: numpy.generic) -> float | str | None:
    if _is_missing(value):
        return None
    return _utc_datetime(value).isoformat() if _is_datetime(value) else float(value)


def _is_datetime(value: numpy.generic) -> bool:
    return numpy.issubdtype(value.dtype, numpy.datetime64)


def _is_missing(value: numpy.generic) -> bool:
    return bool(numpy.isnat(value)) if _is_datetime(value) else not numpy.isfinite(value)


def _regular_step(values: numpy.ndarray) -> float | None:
    if not numpy.issubdtype(values.dtype, numpy.number) or values.size < 2:
        return None
    steps = numpy.diff(values.astype(float))
    return float(steps[0]) if numpy.isfinite(steps).all() and numpy.allclose(steps, steps[0], rtol=1e-3) else None


def _variable(name: str, data_array: xarray.DataArray) -> Variable:
    return Variable(
        name=str(name),
        standard_name=data_array.attrs.get("standard_name"),
        long_name=data_array.attrs.get("long_name"),
        units=data_array.attrs.get("units"),
        dimensions=tuple(map(str, data_array.dims)),
    )


def copernicus_marine_datasets(
    dataset_identifiers: tuple[str, ...],
) -> tuple[CopernicusMarineDataset, ...] | MetadataUnavailable:
    try:
        return tuple(map(_copernicus_marine_dataset, dataset_identifiers))
    except Exception as error:
        return MetadataUnavailable(
            UnavailabilityCause.READ_FAILURE, f"{', '.join(dataset_identifiers)}: {type(error).__name__}: {error}"
        )


def _copernicus_marine_dataset(dataset_identifier: str) -> CopernicusMarineDataset:
    product = copernicusmarine.describe(dataset_id=dataset_identifier, disable_progress_bar=True).products[0]
    dataset = next(dataset for dataset in product.datasets if dataset.dataset_id == dataset_identifier)
    service = next(
        service
        for service in dataset.versions[0].parts[0].services
        if service.service_name == COPERNICUS_MARINE_ARCO_SERVICE
    )
    return CopernicusMarineDataset(
        dataset_identifier=dataset_identifier,
        product_identifier=product.product_id,
        product_title=product.title,
        product_description=product.description,
        digital_object_identifier=product.digital_object_identifier,
        thumbnail_url=product.thumbnail_url,
        arco_url=service.uri,
        metadata=_copernicus_marine_metadata(service.variables),
    )


def _copernicus_marine_metadata(variables: list) -> DatasetMetadata:
    coordinates = {
        coordinate.coordinate_id: coordinate for variable in variables for coordinate in variable.coordinates
    }
    return DatasetMetadata(
        spatial_extent=_copernicus_marine_spatial_extent(coordinates),
        temporal_extent=_copernicus_marine_temporal_extent(coordinates),
        dimensions=tuple(_copernicus_marine_dimension(coordinate) for coordinate in coordinates.values()),
        variables=tuple(
            Variable(
                name=variable.short_name,
                standard_name=variable.standard_name,
                long_name=None,
                units=variable.units,
                dimensions=tuple(coordinate.coordinate_id for coordinate in variable.coordinates),
            )
            for variable in variables
        ),
        groups=(),
    )


def _copernicus_marine_spatial_extent(coordinates: dict) -> SpatialExtent | None:
    longitude = next((coordinates[name] for name in LONGITUDE_NAMES if name in coordinates), None)
    latitude = next((coordinates[name] for name in LATITUDE_NAMES if name in coordinates), None)
    if longitude is None or latitude is None:
        return None
    return SpatialExtent(
        west=float(longitude.minimum_value),
        south=float(latitude.minimum_value),
        east=float(longitude.maximum_value),
        north=float(latitude.maximum_value),
    )


def _copernicus_marine_temporal_extent(coordinates: dict) -> TemporalExtent | None:
    time = coordinates.get("time")
    if time is None or time.minimum_value is None:
        return None
    return TemporalExtent(
        start=_milliseconds_datetime(time.minimum_value), end=_milliseconds_datetime(time.maximum_value)
    )


def _milliseconds_datetime(milliseconds: float) -> datetime:
    return pandas.Timestamp(
        milliseconds, unit=COPERNICUS_MARINE_TIME_UNIT_MILLISECONDS, tz=timezone.utc
    ).to_pydatetime()


def _copernicus_marine_dimension(coordinate) -> Dimension:
    kind = _dimension_kind(coordinate.coordinate_id)
    values = coordinate.values or []
    return Dimension(
        name=coordinate.coordinate_id,
        kind=kind,
        size=len(values) if values else _regular_coordinate_size(coordinate),
        minimum=_copernicus_marine_bound(kind, coordinate.minimum_value, values, min),
        maximum=_copernicus_marine_bound(kind, coordinate.maximum_value, values, max),
        step=coordinate.step if kind != DimensionKind.TIME else None,
    )


def _regular_coordinate_size(coordinate) -> int:
    bounds_and_step = (coordinate.minimum_value, coordinate.maximum_value, coordinate.step)
    return (
        round((coordinate.maximum_value - coordinate.minimum_value) / coordinate.step) + 1
        if None not in bounds_and_step and coordinate.step
        else 0
    )


def _copernicus_marine_bound(kind: DimensionKind, bound: float | None, values: list, select) -> float | str | None:
    value = bound if bound is not None else (select(values) if values else None)
    match (kind, value):
        case (_, None):
            return None
        case (DimensionKind.TIME, _):
            return _milliseconds_datetime(value).isoformat()
        case _:
            return float(value)
