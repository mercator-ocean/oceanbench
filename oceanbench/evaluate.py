from pathlib import Path
from typing import Any, Optional

import numpy
import xarray

from oceanbench.core.evaluate.rmse_core import (
    analyze_energy_cascade_core,
    get_euclidean_distance_core,
    glonet_pointwise_evaluation_core,
)


def pointwise_evaluation(glonet_datasets_path: Path | str, glorys_datasets_path: Path | str) -> numpy.ndarray[Any]:
    if isinstance(glonet_datasets_path, str):
        glonet_datasets_path = Path(glonet_datasets_path)
    if isinstance(glorys_datasets_path, str):
        glorys_datasets_path = Path(glorys_datasets_path)
    return glonet_pointwise_evaluation_core(
        glonet_datasets_path=glonet_datasets_path,
        glorys_datasets_path=glorys_datasets_path,
    )


def get_euclidean_distance(
    first_dataset: xarray.Dataset,
    second_dataset: xarray.Dataset,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
):
    return get_euclidean_distance_core(
        first_dataset=first_dataset,
        second_dataset=second_dataset,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
    )


def analyze_energy_cascade(
    dataset: xarray.Dataset,
    var: str,
    depth: float,
    spatial_resolution: Optional[float] = None,
    small_scale_cutoff_km: Optional[float] = 100,
):
    return analyze_energy_cascade_core(
        glonet=dataset,
        var=var,
        depth=depth,
        spatial_resolution=spatial_resolution,
        small_scale_cutoff_km=small_scale_cutoff_km,
    )
