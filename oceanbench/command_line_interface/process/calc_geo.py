from pathlib import Path
import click

from oceanbench.command_line_interface.common_options import lead_option, output_path_option
from oceanbench.core.process.calc_geo_core import calc_geo_core


@click.command(help="Calculate mld")
@click.option(
    "--glonet-dataset-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@lead_option
@click.option(
    "--variable",
    type=click.STRING,
)
@output_path_option
def calc_geo(
    dataset_path: Path,
    lead: int,
    variable: str,
    output_path: Path,
):
    return calc_geo_core(
        dataset_path=dataset_path,
        lead=lead,
        var=variable,
        output_path=output_path,
    )
