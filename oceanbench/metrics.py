from typing import List, Optional

import xarray

from . import plot
from pandas import DataFrame

from oceanbench.core.metrics.rmse_core import (
    analyze_energy_cascade_core,
    get_euclidean_distance_glorys_core,
    pointwise_evaluation_glorys_core,
)


def rmse_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> DataFrame:
    return pointwise_evaluation_glorys_core(challenger_datasets=challenger_datasets)


def euclidean_distance_to_glorys(
    challenger_datasets: List[xarray.Dataset],
    minimum_latitude: float = 466,
    maximum_latitude: float = 633,
    minimum_longitude: float = 400,
    maximum_longitude: float = 466,
):
    euclidean_distance = get_euclidean_distance_glorys_core(
        challenger_dataset=challenger_datasets[0],
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
    )
    plot.plot_euclidean_distance(euclidean_distance)


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
