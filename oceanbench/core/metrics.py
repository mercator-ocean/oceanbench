# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core.dataset_utils import harmonise_dataset
from oceanbench.core.derived_quantities import compute_mixed_layer_depth
from oceanbench.core.derived_quantities import compute_geostrophic_currents
from oceanbench.core.rmsd import Variable, rmsd
from oceanbench.core.references.glorys import glorys_dataset

from oceanbench.core.lagrangian_trajectory import (
    Zone,
    deviation_of_lagrangian_trajectories,
)


def rmsd_of_variables_compared_to_glorys(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=harmonise_dataset(challenger_dataset),
        reference_dataset=harmonise_dataset(glorys_dataset(challenger_dataset)),
        variables=[
            Variable.HEIGHT,
            Variable.TEMPERATURE,
            Variable.SALINITY,
            Variable.NORTHWARD_VELOCITY,
            Variable.EASTWARD_VELOCITY,
        ],
    )


def rmsd_of_mixed_layer_depth_compared_to_glorys(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=harmonise_dataset(compute_mixed_layer_depth(challenger_dataset)),
        reference_dataset=harmonise_dataset(compute_mixed_layer_depth(glorys_dataset(challenger_dataset))),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmsd_of_geostrophic_currents_compared_to_glorys(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=harmonise_dataset(compute_geostrophic_currents(challenger_dataset)),
        reference_dataset=harmonise_dataset(compute_geostrophic_currents(glorys_dataset(challenger_dataset))),
        variables=[
            Variable.NORTHWARD_GEOSTROPHIC_VELOCITY,
            Variable.EASTWARD_GEOSTROPHIC_VELOCITY,
        ],
    )


def deviation_of_lagrangian_trajectories_compared_to_glorys(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return deviation_of_lagrangian_trajectories(
        challenger_dataset=challenger_dataset,
        reference_dataset=glorys_dataset(challenger_dataset),
        zone=Zone.SMALL_ATLANTIC_NEWYORK_TO_NOUADHIBOU,
    )
