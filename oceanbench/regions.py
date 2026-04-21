# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the OceanBench region API.
"""

from pathlib import Path

import xarray

from oceanbench.core.regions import RegionLike
from oceanbench.core.regions import RegionSpec
from oceanbench.core.regions import custom_region as _custom_region
from oceanbench.core.regions import get_pre_defined_region_names as _get_pre_defined_region_names
from oceanbench.core.regions import load_region_file as _load_region_file
from oceanbench.core.regions import subset_dataset_to_region


__all__ = [
    "RegionLike",
    "RegionSpec",
    "available_regions",
    "custom",
    "load_region_file",
    "subset",
]


def available_regions() -> list[str]:
    """
    Return the official OceanBench region identifiers accepted by the CLI ``--region`` option.
    """

    return _get_pre_defined_region_names()


def custom(
    identifier: str,
    display_name: str,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
) -> RegionSpec:
    """
    Create a custom bounded region for programmatic evaluation.
    """

    return _custom_region(
        identifier=identifier,
        display_name=display_name,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
    )


def load_region_file(path: str | Path) -> RegionSpec:
    """
    Load a custom region definition from a JSON file.
    """

    return _load_region_file(path)


def subset(dataset: xarray.Dataset, region: RegionLike) -> xarray.Dataset:
    """
    Return a dataset restricted to the selected OceanBench region.
    """

    return subset_dataset_to_region(dataset, region)
