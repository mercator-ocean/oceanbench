# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the OceanBench region API.
"""

from os import environ

import xarray

from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.regions import BoundingBox
from oceanbench.core.regions import RegionLike
from oceanbench.core.regions import RegionSpec
from oceanbench.core.regions import get_pre_defined_region_names
from oceanbench.core.regions import load_region_file
from oceanbench.core.regions import normalize_region_name
from oceanbench.core.regions import official_region_ids
from oceanbench.core.regions import official_regions
from oceanbench.core.regions import region_from_dict
from oceanbench.core.regions import region_to_dict
from oceanbench.core.regions import resolve_region
from oceanbench.core.regions import subset_dataset_to_region
from oceanbench.core.regions import custom_region as custom


__all__ = [
    "BoundingBox",
    "RegionLike",
    "RegionSpec",
    "available_regions",
    "custom",
    "get_pre_defined_region_names",
    "load_region_file",
    "normalize_region_name",
    "official_region_ids",
    "official_regions",
    "region_from_dict",
    "region_to_dict",
    "resolve_region",
    "selected_region_name_from_environment",
    "subset",
]


available_regions = get_pre_defined_region_names


def selected_region_name_from_environment() -> str:
    return normalize_region_name(environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_REGION.value))


def subset(dataset: xarray.Dataset, region: RegionLike) -> xarray.Dataset:
    return subset_dataset_to_region(dataset, region)
