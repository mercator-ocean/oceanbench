# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass

import xarray

from oceanbench.core.dataset_utils import Dimension


@dataclass(frozen=True)
class GeographicSubRegion:
    identifier: str
    display_name: str
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float


IBI = GeographicSubRegion(
    identifier="ibi",
    display_name="IBI",
    minimum_latitude=26.17,
    maximum_latitude=56.08,
    minimum_longitude=-19.08,
    maximum_longitude=5.08,
)

PRE_DEFINED_SUB_REGIONS = {
    sub_region.identifier: sub_region
    for sub_region in [
        IBI,
    ]
}


def get_pre_defined_sub_region_names() -> list[str]:
    return list(PRE_DEFINED_SUB_REGIONS.keys())


def resolve_sub_region(sub_region_name: str | None) -> GeographicSubRegion | None:
    if sub_region_name in (None, ""):
        return None

    normalized_sub_region_name = sub_region_name.strip().lower()
    if normalized_sub_region_name not in PRE_DEFINED_SUB_REGIONS:
        supported_sub_regions = ", ".join(get_pre_defined_sub_region_names())
        raise ValueError(f"Unsupported sub-region: {sub_region_name}. Supported values are: {supported_sub_regions}.")
    return PRE_DEFINED_SUB_REGIONS[normalized_sub_region_name]


def subset_dataset_to_sub_region(
    dataset: xarray.Dataset,
    sub_region_name: str | None,
) -> xarray.Dataset:
    resolved_sub_region = resolve_sub_region(sub_region_name)
    if resolved_sub_region is None:
        return dataset

    latitude_key = _resolve_coordinate_key(dataset, Dimension.LATITUDE.key())
    longitude_key = _resolve_coordinate_key(dataset, Dimension.LONGITUDE.key())
    latitude_values = dataset[latitude_key]
    longitude_values = dataset[longitude_key]

    if latitude_key in dataset.dims and longitude_key in dataset.dims:
        latitude_mask = _get_latitude_mask(latitude_values, resolved_sub_region)
        longitude_mask = _get_longitude_mask(longitude_values, resolved_sub_region)
        _validate_mask(latitude_mask, latitude_key, resolved_sub_region)
        _validate_mask(longitude_mask, longitude_key, resolved_sub_region)
        return dataset.isel(
            {
                latitude_key: latitude_mask,
                longitude_key: longitude_mask,
            }
        )

    shared_dimensions = [dimension for dimension in latitude_values.dims if dimension in longitude_values.dims]
    if len(shared_dimensions) != 1:
        raise ValueError(
            "Sub-region selection requires latitude and longitude to either be dimensions, "
            + "or point-like variables sharing a single dimension."
        )

    shared_dimension = shared_dimensions[0]
    point_mask = _get_latitude_mask(latitude_values, resolved_sub_region) & _get_longitude_mask(
        longitude_values, resolved_sub_region
    )
    _validate_mask(point_mask, shared_dimension, resolved_sub_region)
    return dataset.isel({shared_dimension: point_mask})


def _resolve_coordinate_key(
    dataset: xarray.Dataset,
    standard_name: str,
) -> str:
    if standard_name in dataset.variables:
        return standard_name

    matching_variable_names = [
        variable_name
        for variable_name in dataset.variables
        if getattr(dataset[variable_name], "standard_name", None) == standard_name
    ]
    if len(matching_variable_names) == 1:
        return matching_variable_names[0]
    raise ValueError(
        f"Dataset does not expose a unique variable with standard_name={standard_name!r} required for sub-region."
    )


def _get_latitude_mask(
    latitude_values: xarray.DataArray,
    sub_region: GeographicSubRegion,
) -> xarray.DataArray:
    return (latitude_values >= sub_region.minimum_latitude) & (latitude_values <= sub_region.maximum_latitude)


def _get_longitude_mask(
    longitude_values: xarray.DataArray,
    sub_region: GeographicSubRegion,
) -> xarray.DataArray:
    dataset_uses_positive_longitudes = (
        float(longitude_values.min().values) >= 0 and float(longitude_values.max().values) > 180
    )
    minimum_longitude = sub_region.minimum_longitude
    maximum_longitude = sub_region.maximum_longitude

    if dataset_uses_positive_longitudes:
        minimum_longitude = minimum_longitude % 360
        maximum_longitude = maximum_longitude % 360

    if minimum_longitude <= maximum_longitude:
        return (longitude_values >= minimum_longitude) & (longitude_values <= maximum_longitude)
    return (longitude_values >= minimum_longitude) | (longitude_values <= maximum_longitude)


def _validate_mask(
    mask: xarray.DataArray,
    dimension_name: str,
    sub_region: GeographicSubRegion,
) -> None:
    if bool(mask.any().item()):
        return
    raise ValueError(f"Sub-region {sub_region.display_name} does not overlap dataset {dimension_name} coordinates.")
