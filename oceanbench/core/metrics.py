import pandas
import xarray

from typing import List

from oceanbench.core.derived_quantities import add_mixed_layer_depth
from oceanbench.core.derived_quantities import add_geostrophic_currents
from oceanbench.core.rmsd import Variable, rmsd
from oceanbench.core.references.glorys import glorys_datasets

from oceanbench.core.lagrangian_trajectory import (
    Zone,
    deviation_of_lagrangian_trajectories,
)


def rmsd_of_variables_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return rmsd(
        challenger_datasets=challenger_datasets,
        reference_datasets=glorys_datasets(challenger_datasets),
        variables=[
            Variable.HEIGHT,
            Variable.TEMPERATURE,
            Variable.SALINITY,
            Variable.NORTHWARD_VELOCITY,
            Variable.EASTWARD_VELOCITY,
        ],
    )


def rmsd_of_mixed_layer_depth_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return rmsd(
        challenger_datasets=add_mixed_layer_depth(challenger_datasets),
        reference_datasets=add_mixed_layer_depth(glorys_datasets(challenger_datasets)),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmsd_of_geostrophic_currents_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return rmsd(
        challenger_datasets=add_geostrophic_currents(challenger_datasets),
        reference_datasets=add_geostrophic_currents(glorys_datasets(challenger_datasets)),
        variables=[
            Variable.NORTHWARD_GEOSTROPHIC_VELOCITY,
            Variable.EASTWARD_GEOSTROPHIC_VELOCITY,
        ],
    )


def deviation_of_lagrangian_trajectories_compared_to_glorys(
    challenger_datasets: List[xarray.Dataset],
) -> pandas.DataFrame:
    return deviation_of_lagrangian_trajectories(
        challenger_datasets=challenger_datasets,
        reference_datasets=glorys_datasets(challenger_datasets),
        zone=Zone.SMALL_ATLANTIC_NEWYORK_TO_NOUADHIBOU,
    )
