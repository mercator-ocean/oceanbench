from typing import Any, List, Optional

import numpy
import xarray

from oceanbench.core.evaluate.rmse_core import (
    analyze_energy_cascade_core,
    get_euclidean_distance_glorys_core,
    pointwise_evaluation_glorys_core,
)


def rmse_to_glorys(
    candidate_datasets: List[xarray.Dataset],
) -> numpy.ndarray[Any, Any]:
    return pointwise_evaluation_glorys_core(
        candidate_datasets=candidate_datasets,
    )


def euclidean_distance_to_glorys(
    candidate_dataset: xarray.Dataset,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
):
    return get_euclidean_distance_glorys_core(
        candidate_dataset=candidate_dataset,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
    )


def energy_cascade(
    candidate_dataset: xarray.Dataset,
    var: str,
    depth: float,
    spatial_resolution: Optional[float] = None,
    small_scale_cutoff_km: Optional[float] = 100,
):
    return analyze_energy_cascade_core(
        candidate_dataset=candidate_dataset,
        var=var,
        depth=depth,
        spatial_resolution=spatial_resolution,
        small_scale_cutoff_km=small_scale_cutoff_km,
    )
