from pathlib import Path
import click

from oceanbench.command_line_interface.common_options import (
    lead_option,
    maximum_latitude_option,
    maximum_longitude_option,
    minimum_latitude_option,
    minimum_longitude_option,
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
@minimum_latitude_option
@maximum_latitude_option
@minimum_longitude_option
@maximum_longitude_option
@output_path_option
def calc_density(
    glonet_dataset_path: Path,
    lead: int,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
    output_path: Path,
):
    return calc_density_core(
        dataset_path=glonet_dataset_path,
        lead=lead,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
        output_path=output_path,
    )
