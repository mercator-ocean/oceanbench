from pathlib import Path
import click

from oceanbench.core.process.calc_mld_core import calc_mld_core


@click.command(help="Calculate mld")
@click.option(
    "--glonet-dataset-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--lead",
    type=click.INT,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path),
    default=Path("./output.nc"),
)
def calc_mld(glonet_dataset_path: Path, lead: int, output_path: Path):
    return calc_mld_core(glonet_dataset_path, lead, output_path)
