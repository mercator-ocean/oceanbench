from pathlib import Path


from oceanbench.core.process.calc_density_core import calc_density_core


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
