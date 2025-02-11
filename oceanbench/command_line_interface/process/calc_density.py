from pathlib import Path
import click

from oceanbench.core.process.calc_density_core import calc_density_core


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
    "--latitude",
    type=click.FLOAT,
)
@click.option(
    "--longitude",
    type=click.FLOAT,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path),
    default=Path("./output.nc"),
)
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
