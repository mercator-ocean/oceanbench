# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import xarray

from pandas import DataFrame

from oceanbench.core import metrics


def rmsd_of_variables_compared_to_glorys(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    return metrics.rmsd_of_variables_compared_to_glorys(challenger_dataset=challenger_dataset)


def rmsd_of_mixed_layer_depth_compared_to_glorys(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    return metrics.rmsd_of_mixed_layer_depth_compared_to_glorys(challenger_dataset=challenger_dataset)


def rmsd_of_geostrophic_currents_compared_to_glorys(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    return metrics.rmsd_of_geostrophic_currents_compared_to_glorys(challenger_dataset=challenger_dataset)


def deviation_of_lagrangian_trajectories_compared_to_glorys(
    challenger_dataset: xarray.Dataset,
) -> DataFrame:
    return metrics.deviation_of_lagrangian_trajectories_compared_to_glorys(challenger_dataset=challenger_dataset)
