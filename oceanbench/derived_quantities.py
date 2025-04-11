from typing import List
from . import plot
import xarray

from oceanbench.core.process.calc_density_core import calc_density_core
from oceanbench.core.process.calc_geo_core import calc_geo_core
from oceanbench.core.process.utils import (
    mass_conservation_core,
)


def density(
    challenger_datasets: List[xarray.Dataset],
    lead: int = 1,
    minimum_latitude: float = -100,
    maximum_latitude: float = -40,
    minimum_longitude: float = -15,
    maximum_longitude: float = 50,
):
    dataarray = calc_density_core(
        dataset=challenger_datasets[0],
        lead=lead,
        minimum_latitude=minimum_latitude,
        maximum_latitude=maximum_latitude,
        minimum_longitude=minimum_longitude,
        maximum_longitude=maximum_longitude,
    )
    plot.plot_density(dataarray=dataarray)


def geostrophic_currents(
    challenger_datasets: List[xarray.Dataset],
    lead: int = 1,
    variable: str = "zos",
):
    dataset = calc_geo_core(
        dataset=challenger_datasets[0],
        lead=lead,
        var=variable,
    )
    plot.plot_geo(dataset=dataset)


def mass_conservation(
    challenger_datasets: List[xarray.Dataset],
    depth: float = 0,
    deg_resolution: float = 0.25,
):
    mean_div_time_series = mass_conservation_core(
        dataset=challenger_datasets[0],
        depth=depth,
        deg_resolution=deg_resolution,
    )
    print(mean_div_time_series.data)  # time-dependent scores


def kinetic_energy(challenger_datasets: List[xarray.Dataset]):
    plot.plot_kinetic_energy(challenger_datasets[0])


def vorticity(challenger_datasets: List[xarray.Dataset]):
    plot.plot_vorticity(challenger_datasets[0])
