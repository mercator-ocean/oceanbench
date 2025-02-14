from typing import Any

import numpy
import xarray

from oceanbench.core.plot.density_core import plot_density_core
from oceanbench.core.plot.geo_core import plot_geo_core
from oceanbench.core.plot.kinetic_energy_core import plot_kinetic_energy_core
from oceanbench.core.plot.mld_core import plot_mld_core
from oceanbench.core.plot.rmse_core import (
    plot_depth_rmse_average_on_time,
    plot_energy_cascade_core,
    plot_euclidean_distance_core,
    plot_temporal_rmse_for_average_depth,
    plot_temporal_rmse_for_depth,
)
from oceanbench.core.plot.vortocity_core import plot_vortocity_core


def plot_density(dataarray: xarray.DataArray):
    return plot_density_core(
        dataarray=dataarray,
    )


def plot_geo(dataset: xarray.Dataset):
    return plot_geo_core(
        dataset=dataset,
    )


def plot_mld(dataset: xarray.Dataset):
    return plot_mld_core(
        dataset=dataset,
    )


def plot_pointwise_evaluation(rmse_dataarray: numpy.ndarray[Any], depth: int):
    return plot_temporal_rmse_for_depth(
        rmse_dataarray=rmse_dataarray,
        depth=depth,
    )


def plot_pointwise_evaluation_for_average_depth(rmse_dataarray: numpy.ndarray[Any]):
    return plot_temporal_rmse_for_average_depth(
        rmse_dataarray=rmse_dataarray,
    )


def plot_pointwise_evaluation_depth_for_average_time(
    rmse_dataarray: numpy.ndarray[Any],
    dataset_depth_values: numpy.ndarray,
):
    return plot_depth_rmse_average_on_time(
        rmse_dataarray=rmse_dataarray,
        dataset_depth_values=dataset_depth_values,
    )


def plot_euclidean_distance(euclidean_distance):
    return plot_euclidean_distance_core(euclidean_distance)


def plot_energy_cascade(gglonet_sc):
    return plot_energy_cascade_core(gglonet_sc)


def plot_kinetic_energy(dataset: xarray.Dataset):
    return plot_kinetic_energy_core(dataset)


def plot_vortocity(dataset: xarray.Dataset):
    return plot_vortocity_core(dataset)
