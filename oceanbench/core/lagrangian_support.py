# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable

import numpy
import pandas
import xarray

from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable

LAGRANGIAN_ROW_LABEL = "Lagrangian trajectory deviation (km) []{surface}"


def mean_weekly_lagrangian_deviations(weekly_deviations: list[numpy.ndarray]) -> numpy.ndarray:
    return pandas.concat(map(pandas.Series, weekly_deviations), axis=1).mean(axis=1).values


def surface_current_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    current_dataset = rename_dataset_with_standard_names(dataset)[
        [
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
        ]
    ]
    if Dimension.DEPTH.key() in current_dataset.dims:
        return current_dataset.isel({Dimension.DEPTH.key(): 0}, drop=True)
    return current_dataset


def all_weekly_lagrangian_deviations(
    challenger_week_datasets: list[xarray.Dataset],
    reference_week_datasets: list[xarray.Dataset],
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
    compute_weekly_deviation: Callable[[xarray.Dataset, xarray.Dataset, numpy.ndarray, numpy.ndarray], numpy.ndarray],
) -> list[numpy.ndarray]:
    return [
        compute_weekly_deviation(challenger_week_dataset, reference_week_dataset, latitudes, longitudes)
        for challenger_week_dataset, reference_week_dataset in zip(
            challenger_week_datasets,
            reference_week_datasets,
            strict=True,
        )
    ]
