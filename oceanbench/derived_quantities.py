import xarray

from oceanbench.core.process.calc_mld_core import calc_mld_core
from oceanbench.core.process.calc_density_core import calc_density_core
from oceanbench.core.process.calc_geo_core import calc_geo_core
from oceanbench.core.process.lagrangian_analysis import get_particle_file_core
from oceanbench.core.process.utils import mass_conservation_core


def density(
    dataset: xarray.Dataset,
    lead: int,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
) -> xarray.Dataset:
    return calc_density_core(
        dataset=dataset,
        lead=lead,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
    )


def geostrophic_currents(
    dataset: xarray.Dataset,
    lead: int,
    variable: str,
) -> xarray.Dataset:
    return calc_geo_core(
        dataset=dataset,
        lead=lead,
        var=variable,
    )


def mld(dataset: xarray.Dataset, lead: int) -> xarray.Dataset:
    return calc_mld_core(
        dataset=dataset,
        lead=lead,
    )


def get_particle_file(
    dataset: xarray.Dataset,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
) -> xarray.Dataset:
    return get_particle_file_core(
        dataset=dataset,
        latzone=[minimum_latitude, maximum_latitude],
        lonzone=[minimum_longitude, maximum_longitude],
    )


def mass_conservation(
    dataset: xarray.Dataset, depth: float, deg_resolution: float = 0.25
) -> xarray.DataArray:
    return mass_conservation_core(
        dataset=dataset, depth=depth, deg_resolution=deg_resolution
    )
