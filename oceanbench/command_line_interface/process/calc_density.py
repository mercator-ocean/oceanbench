from pathlib import Path
import click

from oceanbench.command_line_interface.common_options import (
    latitude_option,
    lead_option,
    longitude_option,
    output_path_option,
)
from oceanbench.core.process.calc_density_core import calc_density_core


@click.command(help="Calculate mld")
@click.option(
    "--glonet-dataset-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@lead_option
@longitude_option
@latitude_option
@output_path_option
def calc_density(
    dataset_path: Path,
    lead: int,
    latitude: float,
    longitude: float,
    output_path: Path,
):
    return calc_density_core(
        dataset_path=dataset_path,
        lead=lead,
        lat=latitude,
        lon=longitude,
        output_path=output_path,
    )
