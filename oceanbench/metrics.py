from typing import List

import xarray

from pandas import DataFrame

from oceanbench.core import metrics


def rmsd_of_variables_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> DataFrame:
    return metrics.rmsd_of_variables_compared_to_glorys(challenger_datasets=challenger_datasets)


def rmsd_of_mixed_layer_depth_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> DataFrame:
    return metrics.rmsd_of_mixed_layer_depth_compared_to_glorys(challenger_datasets=challenger_datasets)


def rmsd_of_geostrophic_currents_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> DataFrame:
    return metrics.rmsd_of_geostrophic_currents_compared_to_glorys(challenger_datasets=challenger_datasets)


def deviation_of_lagrangian_trajectories_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> DataFrame:
    return metrics.deviation_of_lagrangian_trajectories_compared_to_glorys(challenger_datasets=challenger_datasets)
