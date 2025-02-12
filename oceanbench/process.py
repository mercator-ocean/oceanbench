from pathlib import Path

import xarray

from oceanbench.core.process.calc_mld_core import calc_mld_core
from oceanbench.core.process.calc_density_core import calc_density_core
from oceanbench.core.process.calc_geo_core import calc_geo_core


def calc_density(
    glonet_dataset_path: Path,
    lead: int,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
) -> xarray.Dataset:
    return calc_density_core(
        dataset_path=glonet_dataset_path,
        lead=lead,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
    )


def calc_geo(
    dataset_path: Path,
    lead: int,
    variable: str,
) -> xarray.Dataset:
    return calc_geo_core(
        dataset_path=dataset_path,
        lead=lead,
        var=variable,
    )


def calc_mld(glonet_dataset_path: Path, lead: int) -> xarray.Dataset:
    return calc_mld_core(
        glonet_dataset_path=glonet_dataset_path,
        lead=lead,
    )
