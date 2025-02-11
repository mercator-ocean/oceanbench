from pathlib import Path
import click

from oceanbench.command_line_interface.common_options import show_plot_option
from oceanbench.core.plot.density_core import plot_density_core


@click.command(help="Plot the MLD")
@click.option(
    "-i",
    "--density-dataset-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--plot-output-path",
    type=click.Path(path_type=Path),
    default=Path("./plot.png"),
)
@show_plot_option
def plot_density(density_dataset_path: Path, plot_output_path: Path, show_plot: bool):
    return plot_density_core(
        dataset_path=density_dataset_path,
        plot_output_path=plot_output_path,
        show_plot=show_plot,
    )
