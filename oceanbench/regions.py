# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the pre-defined geographic regions supported by OceanBench.
"""

from os import environ

import xarray

from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.regions import GeographicRegion
from oceanbench.core.regions import get_pre_defined_region_names
from oceanbench.core.regions import normalize_region_name as _normalize_region_name
from oceanbench.core.regions import resolve_region as _resolve_region
from oceanbench.core.regions import subset_dataset_to_region


def available_regions() -> list[str]:
    """
    List the pre-defined geographic regions supported by OceanBench.

    Returns
    -------
    list[str]
        The available region identifiers.
    """

    return get_pre_defined_region_names()


def normalize_region_name(region_name: str | None) -> str:
    """
    Normalize a region identifier for OceanBench usage.

    Parameters
    ----------
    region_name : str, optional
        The region identifier. If ``None``, the global region is selected.

    Returns
    -------
    str
        The normalized region identifier.
    """

    return _normalize_region_name(region_name)


def resolve_region(
    region_name: str | None,
) -> GeographicRegion | None:
    """
    Resolve a region identifier to a pre-defined OceanBench geographic region.

    Parameters
    ----------
    region_name : str, optional
        The region identifier. If ``None`` or ``global``, the full global domain is selected.

    Returns
    -------
    GeographicRegion or None
        The matching pre-defined geographic region, or ``None`` for the full global domain.
    """

    return _resolve_region(region_name)


def selected_region_name_from_environment() -> str:
    """
    Read the configured OceanBench region from the process environment.

    Returns
    -------
    str
        The configured region identifier, normalized for OceanBench usage.
    """

    return _normalize_region_name(environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_REGION.value))


def subset(
    dataset: xarray.Dataset,
    region_name: str | None,
) -> xarray.Dataset:
    """
    Restrict a dataset to a pre-defined OceanBench region.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to subset.
    region_name : str, optional
        The region identifier. If ``None`` or ``global``, the dataset is returned unchanged.

    Returns
    -------
    xarray.Dataset
        The region subset of the dataset.
    """

    return subset_dataset_to_region(dataset, region_name)
