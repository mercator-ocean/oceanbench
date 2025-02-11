from pathlib import Path
import click

from oceanbench.core.plot.geo_core import plot_geo_core


@click.command(help="Plot the MLD")
@click.option(
    "-i",
    "--geo-datasets-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def plot_geo(geo_datasets_path: Path):
    return plot_geo_core(geo_datasets_path)
