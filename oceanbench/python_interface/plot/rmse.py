from pathlib import Path

from oceanbench.core.plot.rmse_core import (
    plot_depth_rmse_average_on_time,
    plot_temporal_rmse_for_average_depth,
    plot_temporal_rmse_for_depth,
)


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
