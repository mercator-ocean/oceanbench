from pathlib import Path

import click

from oceanbench.plots import (
    plot_depth_rmse_average_on_time,
    plot_temporal_rmse_for_average_depth,
    plot_temporal_rmse_for_depth,
)
from oceanbench.pointwise_evaluation import glonet_pointwise_evaluation


@click.group()
def run():
    click.echo("Ocean Bench command line interface tool")


@click.command()
@click.option(
    "-gnpath",
    "--glonet-datasets-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "-grpath",
    "--glorys-datasets-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("-o", "--output-rmse", type=click.Path(path_type=Path), default=Path("."))
def pointwise_evaluation(glonet_datasets_path: Path, glorys_datasets_path: Path, output_rmse: Path):
    return glonet_pointwise_evaluation(glonet_datasets_path, glorys_datasets_path, output_rmse)


@click.command()
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
@click.option(
    "-show",
    "--show-plot",
    is_flag=True,
    default=False,
)
def plot_pointwise_evaluation(rmse_path: Path, depth: int, plot_output_path: Path, show_plot: bool):
    return plot_temporal_rmse_for_depth(rmse_path, depth, plot_output_path, show_plot)


@click.command()
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
@click.option(
    "-show",
    "--show-plot",
    is_flag=True,
    default=False,
)
def plot_pointwise_evaluation_for_average_depth(rmse_path: Path, plot_output_path: Path, show_plot: bool):
    return plot_temporal_rmse_for_average_depth(rmse_path, plot_output_path, show_plot)


@click.command()
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
@click.option(
    "-show",
    "--show-plot",
    is_flag=True,
    default=False,
)
def plot_pointwise_evaluation_depth_for_average_time(
    rmse_path: Path, glonet_datasets_path: Path, plot_output_path: Path, show_plot: bool
):
    return plot_depth_rmse_average_on_time(rmse_path, glonet_datasets_path, plot_output_path, show_plot)


run.add_command(pointwise_evaluation)
run.add_command(plot_pointwise_evaluation)
run.add_command(plot_pointwise_evaluation_for_average_depth)
run.add_command(plot_pointwise_evaluation_depth_for_average_time)

if __name__ == "__main__":
    run()
