# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import xarray

from oceanbench.core.dataset_utils import Dimension


GLOBAL_REGION_NAME = "global"
REGION_IDENTIFIER_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class BoundingBox:
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float


@dataclass(frozen=True)
class RegionSpec:
    id: str
    display_name: str
    bounds: BoundingBox | None

    @property
    def official(self) -> bool:
        return self.id in OFFICIAL_REGIONS


GLOBAL = RegionSpec(
    id=GLOBAL_REGION_NAME,
    display_name="Global",
    bounds=None,
)

IBI = RegionSpec(
    id="ibi",
    display_name="IBI",
    bounds=BoundingBox(
        minimum_latitude=26.17,
        maximum_latitude=56.08,
        minimum_longitude=-19.08,
        maximum_longitude=5.08,
    ),
)

OFFICIAL_REGIONS = {
    region.id: region
    for region in [
        GLOBAL,
        IBI,
    ]
}

RegionLike = str | RegionSpec | None


def _normalize_region_identifier(region_id: str) -> str:
    normalized_region_id = region_id.strip().lower()
    if not REGION_IDENTIFIER_PATTERN.fullmatch(normalized_region_id):
        raise ValueError(
            "Region identifiers must match the pattern "
            + f"{REGION_IDENTIFIER_PATTERN.pattern!r}. Received: {region_id!r}."
        )
    return normalized_region_id


def _validate_bounding_box(bounds: BoundingBox) -> BoundingBox:
    if bounds.minimum_latitude >= bounds.maximum_latitude:
        raise ValueError("Region minimum latitude must be strictly smaller than maximum latitude.")
    if bounds.minimum_longitude == bounds.maximum_longitude:
        raise ValueError("Region longitude bounds must span a non-zero interval.")
    if bounds.minimum_latitude < -90 or bounds.maximum_latitude > 90:
        raise ValueError("Region latitude bounds must stay within [-90, 90].")
    if bounds.minimum_longitude < -360 or bounds.maximum_longitude > 360:
        raise ValueError("Region longitude bounds must stay within [-360, 360].")
    return bounds


def _validate_display_name(display_name: str) -> str:
    normalized_display_name = display_name.strip()
    if not normalized_display_name:
        raise ValueError("Region display_name cannot be empty.")
    return normalized_display_name


def official_region_ids() -> list[str]:
    return list(OFFICIAL_REGIONS.keys())


def official_regions() -> list[RegionSpec]:
    return list(OFFICIAL_REGIONS.values())


def get_pre_defined_region_names() -> list[str]:
    return official_region_ids()


def normalize_region_name(region_name: str | None) -> str:
    if region_name in (None, ""):
        return GLOBAL_REGION_NAME

    normalized_region_name = _normalize_region_identifier(region_name)
    if normalized_region_name not in OFFICIAL_REGIONS:
        supported_regions = ", ".join(official_region_ids())
        raise ValueError(f"Unsupported region: {region_name}. Supported values are: {supported_regions}.")
    return normalized_region_name


def resolve_region(region: RegionLike) -> RegionSpec:
    if not isinstance(region, RegionSpec):
        return OFFICIAL_REGIONS[normalize_region_name(region)]

    region_id = _normalize_region_identifier(region.id)
    official_region = OFFICIAL_REGIONS.get(region_id)
    if official_region is not None:
        if region != official_region:
            raise ValueError(
                f"Region id {region_id!r} is reserved for the built-in official region. "
                + "Use the named region directly instead of redefining it."
            )
        return official_region

    if region.bounds is None:
        raise ValueError("Custom regions must define bounds.")

    return RegionSpec(
        id=region_id,
        display_name=_validate_display_name(region.display_name),
        bounds=_validate_bounding_box(region.bounds),
    )


def custom_region(
    identifier: str,
    display_name: str,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
) -> RegionSpec:
    region_id = _normalize_region_identifier(identifier)
    if region_id in OFFICIAL_REGIONS:
        raise ValueError(
            f"Region id {region_id!r} is reserved for a built-in official region. "
            + "Use --region for official regions and a different id for custom regions."
        )
    return RegionSpec(
        id=region_id,
        display_name=_validate_display_name(display_name),
        bounds=_validate_bounding_box(
            BoundingBox(
                minimum_latitude=minimum_latitude,
                maximum_latitude=maximum_latitude,
                minimum_longitude=minimum_longitude,
                maximum_longitude=maximum_longitude,
            )
        ),
    )


def _bounds_to_dict(bounds: BoundingBox | None) -> dict[str, float] | None:
    if bounds is None:
        return None

    return {
        "minimum_latitude": bounds.minimum_latitude,
        "maximum_latitude": bounds.maximum_latitude,
        "minimum_longitude": bounds.minimum_longitude,
        "maximum_longitude": bounds.maximum_longitude,
    }


def region_to_dict(region: RegionLike) -> dict[str, Any]:
    resolved_region = resolve_region(region)
    return {
        "id": resolved_region.id,
        "display_name": resolved_region.display_name,
        "official": resolved_region.official,
        "bounds": _bounds_to_dict(resolved_region.bounds),
    }


def region_from_dict(data: dict[str, Any]) -> RegionSpec:
    if not isinstance(data, dict):
        raise ValueError("Region configuration must be a mapping.")

    if "id" not in data:
        raise ValueError("Region configuration must define an 'id'.")

    region_id = _normalize_region_identifier(str(data["id"]))
    official_region = OFFICIAL_REGIONS.get(region_id)

    if official_region is not None:
        allowed_display_name = data.get("display_name", official_region.display_name)
        allowed_official = data.get("official", True)
        allowed_bounds = data.get("bounds", _bounds_to_dict(official_region.bounds))
        if (
            _validate_display_name(str(allowed_display_name)) != official_region.display_name
            or bool(allowed_official) is not True
            or allowed_bounds != _bounds_to_dict(official_region.bounds)
        ):
            raise ValueError(
                f"Region id {region_id!r} is reserved for the built-in official region. "
                + "Use the named region directly instead of redefining it."
            )
        return official_region

    display_name = _validate_display_name(str(data.get("display_name", region_id.replace("_", " ").title())))
    if bool(data.get("official", False)):
        raise ValueError("Custom regions cannot be marked as official.")

    bounds_data = data.get("bounds")
    if not isinstance(bounds_data, dict):
        raise ValueError("Custom regions must define a 'bounds' mapping.")

    try:
        minimum_latitude = float(bounds_data["minimum_latitude"])
        maximum_latitude = float(bounds_data["maximum_latitude"])
        minimum_longitude = float(bounds_data["minimum_longitude"])
        maximum_longitude = float(bounds_data["maximum_longitude"])
    except KeyError as error:
        raise ValueError(f"Missing custom region bound: {error.args[0]}") from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid custom region bounds: {bounds_data!r}") from error

    return custom_region(
        identifier=region_id,
        display_name=display_name,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
    )


def load_region_file(path: str | Path) -> RegionSpec:
    region_path = Path(path)
    try:
        with region_path.open("r", encoding="utf8") as file:
            region_data = json.load(file)
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid region JSON in {region_path}: {error}") from error
    except OSError as error:
        raise ValueError(f"Unable to read region file {region_path}: {error.strerror or error}") from error
    return region_from_dict(region_data)


def subset_dataset_to_region(
    dataset: xarray.Dataset,
    region: RegionLike,
) -> xarray.Dataset:
    resolved_region = resolve_region(region)
    if resolved_region.bounds is None:
        return dataset

    latitude_key = _resolve_coordinate_key(dataset, Dimension.LATITUDE.key())
    longitude_key = _resolve_coordinate_key(dataset, Dimension.LONGITUDE.key())
    latitude_values = dataset[latitude_key]
    longitude_values = dataset[longitude_key]

    if latitude_key in dataset.dims and longitude_key in dataset.dims:
        latitude_mask = _validated_mask(
            _get_latitude_mask(latitude_values, resolved_region.bounds),
            latitude_key,
            resolved_region,
        )
        longitude_mask = _validated_mask(
            _get_longitude_mask(longitude_values, resolved_region.bounds),
            longitude_key,
            resolved_region,
        )
        return dataset.isel(
            {
                latitude_key: latitude_mask,
                longitude_key: longitude_mask,
            }
        )

    shared_dimensions = [dimension for dimension in latitude_values.dims if dimension in longitude_values.dims]
    if len(shared_dimensions) != 1:
        raise ValueError(
            "Region selection requires latitude and longitude to either be dimensions, "
            + "or point-like variables sharing a single dimension."
        )

    shared_dimension = shared_dimensions[0]
    point_mask = _validated_mask(
        _get_latitude_mask(latitude_values, resolved_region.bounds)
        & _get_longitude_mask(longitude_values, resolved_region.bounds),
        shared_dimension,
        resolved_region,
    )
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
        f"Dataset does not expose a unique variable with standard_name={standard_name!r} required for region."
    )


def _get_latitude_mask(
    latitude_values: xarray.DataArray,
    bounds: BoundingBox,
) -> xarray.DataArray:
    return (latitude_values >= bounds.minimum_latitude) & (latitude_values <= bounds.maximum_latitude)


def _get_longitude_mask(
    longitude_values: xarray.DataArray,
    bounds: BoundingBox,
) -> xarray.DataArray:
    dataset_uses_positive_longitudes = (
        float(longitude_values.min().values) >= 0 and float(longitude_values.max().values) > 180
    )
    minimum_longitude = bounds.minimum_longitude
    maximum_longitude = bounds.maximum_longitude

    if dataset_uses_positive_longitudes:
        minimum_longitude = minimum_longitude % 360
        maximum_longitude = maximum_longitude % 360

    if minimum_longitude <= maximum_longitude:
        return (longitude_values >= minimum_longitude) & (longitude_values <= maximum_longitude)
    return (longitude_values >= minimum_longitude) | (longitude_values <= maximum_longitude)


def _validated_mask(
    mask: xarray.DataArray,
    dimension_name: str,
    region: RegionSpec,
) -> xarray.DataArray:
    computed_mask = mask.compute()
    if bool(computed_mask.any().item()):
        return computed_mask
    raise ValueError(f"Region {region.display_name} does not overlap dataset {dimension_name} coordinates.")
