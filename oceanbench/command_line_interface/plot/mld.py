from pathlib import Path
import click

from oceanbench.core.plot.mld_core import plot_mld_core


@click.command(help="Plot the MLD")
@click.option(
    "-i",
    "--mld-datasets-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
def plot_mld(mld_datasets_path: Path):
    return plot_mld_core(mld_datasets_path)
