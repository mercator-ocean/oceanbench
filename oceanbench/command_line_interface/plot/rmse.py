from pathlib import Path
import click

from oceanbench.command_line_interface.common_options import show_plot_option
from oceanbench.core.plot.rmse_core import (
    plot_depth_rmse_average_on_time,
    plot_temporal_rmse_for_average_depth,
    plot_temporal_rmse_for_depth,
)


@click.command(help="Plot the temporal RMSE for a given depth")
@click.option(
    "-rmse",
    "--rmse-path",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--depth",
    type=click.INT,
    required=True,
)
@click.option(
    "--plot-output-path",
    type=click.Path(path_type=Path),
    default=Path("./plot.png"),
)
@show_plot_option
def plot_pointwise_evaluation(rmse_path: Path, depth: int, plot_output_path: Path, show_plot: bool):
    return plot_temporal_rmse_for_depth(rmse_path, depth, plot_output_path, show_plot)


@click.command(help="Plot the temporal RMSE for average depth")
@click.option(
    "-rmse",
    "--rmse-path",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "--plot-output-path",
    type=click.Path(path_type=Path),
    default=Path("./plot.png"),
)
@show_plot_option
def plot_pointwise_evaluation_for_average_depth(rmse_path: Path, plot_output_path: Path, show_plot: bool):
    return plot_temporal_rmse_for_average_depth(rmse_path, plot_output_path, show_plot)


@click.command(help="Plot the pointwise evalutaion depth for average time")
@click.option(
    "-rmse",
    "--rmse-path",
    type=click.Path(path_type=Path),
    required=True,
)
@click.option(
    "-gnpath",
    "--glonet-datasets-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--plot-output-path",
    type=click.Path(path_type=Path),
    default=Path("./plot.png"),
)
@show_plot_option
def plot_pointwise_evaluation_depth_for_average_time(
    rmse_path: Path, glonet_datasets_path: Path, plot_output_path: Path, show_plot: bool
):
    return plot_depth_rmse_average_on_time(rmse_path, glonet_datasets_path, plot_output_path, show_plot)
