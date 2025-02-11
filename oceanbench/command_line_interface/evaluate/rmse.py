from pathlib import Path
import click

from oceanbench.core.evaluate.rmse_core import glonet_pointwise_evaluation_core


@click.command()
@click.option(
    "-gnpath",
    "--glonet-datasets-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "-grpath",
    "--glorys-datasets-path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option("-o", "--output-rmse", type=click.Path(path_type=Path), default=Path("."))
def pointwise_evaluation(glonet_datasets_path: Path, glorys_datasets_path: Path, output_rmse: Path):
    return glonet_pointwise_evaluation_core(glonet_datasets_path, glorys_datasets_path, output_rmse)
