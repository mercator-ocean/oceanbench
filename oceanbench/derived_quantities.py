from numpy import number
import xarray

from oceanbench.core.process.calc_mld_core import calc_mld_core
from oceanbench.core.process.calc_density_core import calc_density_core
from oceanbench.core.process.calc_geo_core import calc_geo_core
from oceanbench.core.process.utils import (
    compute_kinetic_energy_core,
    compute_vorticity_core,
    mass_conservation_core,
)


def density(
    candidate_dataset: xarray.Dataset,
    lead: int,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
) -> xarray.Dataset:
    return calc_density_core(
        dataset=candidate_dataset,
        lead=lead,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
    )


def geostrophic_currents(
    candidate_dataset: xarray.Dataset,
    lead: int,
    variable: str,
) -> xarray.Dataset:
    return calc_geo_core(
        dataset=candidate_dataset,
        lead=lead,
        var=variable,
    )


def mld(candidate_dataset: xarray.Dataset, lead: int) -> xarray.Dataset:
    return calc_mld_core(
        dataset=candidate_dataset,
        lead=lead,
    )


def mass_conservation(
    candidate_dataset: xarray.Dataset,
    depth: float,
    deg_resolution: float = 0.25,
) -> xarray.DataArray:
    return mass_conservation_core(
        dataset=candidate_dataset, depth=depth, deg_resolution=deg_resolution
    )


def kinetic_energy(candidate_dataset: xarray.Dataset) -> number:
    return compute_kinetic_energy_core(candidate_dataset)


def vorticity(candidate_dataset: xarray.Dataset) -> xarray.DataArray:
    return compute_vorticity_core(candidate_dataset)
