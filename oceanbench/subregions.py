# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the pre-defined geographic sub-regions supported by OceanBench.
"""

from os import environ

import xarray

from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.subregions import GeographicSubRegion
from oceanbench.core.subregions import get_pre_defined_sub_region_names
from oceanbench.core.subregions import resolve_sub_region as _resolve_sub_region
from oceanbench.core.subregions import subset_dataset_to_sub_region


def available_sub_regions() -> list[str]:
    """
    List the pre-defined geographic sub-regions supported by OceanBench.

    Returns
    -------
    list[str]
        The available sub-region identifiers.
    """

    return get_pre_defined_sub_region_names()


def resolve_sub_region(
    sub_region_name: str | None,
) -> GeographicSubRegion | None:
    """
    Resolve a sub-region identifier to a pre-defined OceanBench sub-region.

    Parameters
    ----------
    sub_region_name : str, optional
        The sub-region identifier. If ``None``, no sub-region is selected.

    Returns
    -------
    GeographicSubRegion or None
        The matching pre-defined sub-region.
    """

    return _resolve_sub_region(sub_region_name)


def selected_sub_region_name_from_environment() -> str | None:
    """
    Read the configured OceanBench sub-region from the process environment.

    Returns
    -------
    str or None
        The configured sub-region identifier, normalized for OceanBench usage.
    """

    selected_sub_region = _resolve_sub_region(environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_SUB_REGION.value))
    return None if selected_sub_region is None else selected_sub_region.identifier


def subset(
    dataset: xarray.Dataset,
    sub_region_name: str | None,
) -> xarray.Dataset:
    """
    Restrict a dataset to a pre-defined OceanBench sub-region.

    Parameters
    ----------
    dataset : xarray.Dataset
        The dataset to subset.
    sub_region_name : str, optional
        The sub-region identifier. If ``None``, the dataset is returned unchanged.

    Returns
    -------
    xarray.Dataset
        The sub-region subset of the dataset.
    """

    return subset_dataset_to_sub_region(dataset, sub_region_name)
