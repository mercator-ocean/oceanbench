from pathlib import Path
from typing import Any

import numpy
import xarray

from oceanbench.core.evaluate.rmse_core import (
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
