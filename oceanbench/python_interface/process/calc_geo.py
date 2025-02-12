from pathlib import Path

from oceanbench.core.process.calc_geo_core import calc_geo_core


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
