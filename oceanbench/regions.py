# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes helpers for regional OceanBench evaluations.
"""

from oceanbench.core.regions import (
    NORTH_ATLANTIC,
    RegionDefinition,
    filter_dataframe_from_challenger_region,
    filter_dataframe_to_region,
    filter_observation_dataset_from_challenger_region,
    filter_observation_dataset_to_region,
    filter_trajectory_dataframe_from_challenger_region,
    filter_trajectory_dataframe_by_initial_region,
    region_from_dataset,
    subset_dataset_from_challenger_region,
    subset_dataset_to_region,
)

__all__ = [
    "RegionDefinition",
    "NORTH_ATLANTIC",
    "region_from_dataset",
    "subset_dataset_to_region",
    "subset_dataset_from_challenger_region",
    "filter_observation_dataset_to_region",
    "filter_observation_dataset_from_challenger_region",
    "filter_dataframe_to_region",
    "filter_dataframe_from_challenger_region",
    "filter_trajectory_dataframe_by_initial_region",
    "filter_trajectory_dataframe_from_challenger_region",
]
