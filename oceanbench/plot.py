from pathlib import Path

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


def plot_density(
    density_dataarray: xarray.DataArray,
    plot_output_path: Path,
    show_plot: bool,
):
    return plot_density_core(
        dataarray=density_dataarray,
        plot_output_path=plot_output_path,
        show_plot=show_plot,
    )


def plot_geo(geo_datasets_path: Path):
    return plot_geo_core(geo_datasets_path)


def plot_mld(mld_datasets_path: Path):
    return plot_mld_core(mld_datasets_path)


def plot_pointwise_evaluation(rmse_path: Path, depth: int, plot_output_path: Path, show_plot: bool):
    return plot_temporal_rmse_for_depth(rmse_path, depth, plot_output_path, show_plot)


def plot_pointwise_evaluation_for_average_depth(rmse_path: Path, plot_output_path: Path, show_plot: bool):
    return plot_temporal_rmse_for_average_depth(rmse_path, plot_output_path, show_plot)


def plot_pointwise_evaluation_depth_for_average_time(
    rmse_path: Path,
    glonet_datasets_path: Path,
    plot_output_path: Path,
    show_plot: bool,
):
    return plot_depth_rmse_average_on_time(rmse_path, glonet_datasets_path, plot_output_path, show_plot)


def plot_euclidean_distance(euclidean_distance):
    return plot_euclidean_distance_core(euclidean_distance)


def plot_energy_cascade(gglonet_sc):
    return plot_energy_cascade_core(gglonet_sc)


def plot_kinetic_energy(dataset: xarray.Dataset):
    return plot_kinetic_energy_core(dataset)
