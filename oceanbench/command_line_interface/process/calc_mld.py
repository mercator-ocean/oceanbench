from pathlib import Path
import click

from oceanbench.command_line_interface.common_options import lead_option, output_path_option
from oceanbench.core.process.calc_mld_core import calc_mld_core


@click.command(help="Calculate mld")
@click.option(
    "--glonet-dataset-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@lead_option
@output_path_option
def calc_mld(glonet_dataset_path: Path, lead: int, output_path: Path):
    return calc_mld_core(glonet_dataset_path, lead, output_path)
