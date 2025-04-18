from typing import List, Optional

import xarray

from . import plot
from pandas import DataFrame

from oceanbench.core.metrics import (
    analyze_energy_cascade_core,
)

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


def energy_cascade(
    challenger_datasets: List[xarray.Dataset],
    var: str = "uo",
    depth: float = 0,
    spatial_resolution: Optional[float] = 1 / 4,
    small_scale_cutoff_km: int = 100,
):
    _, gglonet_sc = analyze_energy_cascade_core(
        challenger_dataset=challenger_datasets[0],
        var=var,
        depth=depth,
        spatial_resolution=spatial_resolution,
        small_scale_cutoff_km=small_scale_cutoff_km,
    )
    plot.plot_energy_cascade(gglonet_sc)
