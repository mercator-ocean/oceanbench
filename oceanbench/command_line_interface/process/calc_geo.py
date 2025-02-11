from pathlib import Path
import click

from oceanbench.core.process.calc_geo_core import calc_geo_core


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
    "--variable",
    type=click.STRING,
)
@click.option(
    "-o",
    "--output-path",
    type=click.Path(path_type=Path),
    default=Path("./output.nc"),
)
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
