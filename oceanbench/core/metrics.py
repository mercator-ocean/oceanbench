# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_mixed_layer_depth
from oceanbench.core.derived_quantities import compute_geostrophic_currents
from oceanbench.core.rmsd import rmsd
from oceanbench.core.references.glorys import glorys_dataset

from oceanbench.core.lagrangian_trajectory import (
    Zone,
    deviation_of_lagrangian_trajectories,
)


def rmsd_of_variables_compared_to_glorys(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=glorys_dataset(challenger_dataset),
        variables=[
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
            Variable.SEA_WATER_SALINITY,
            Variable.NORTHWARD_SEA_WATER_VELOCITY,
            Variable.EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


def rmsd_of_mixed_layer_depth_compared_to_glorys(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=compute_mixed_layer_depth(challenger_dataset),
        reference_dataset=compute_mixed_layer_depth(glorys_dataset(challenger_dataset)),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmsd_of_geostrophic_currents_compared_to_glorys(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=compute_geostrophic_currents(challenger_dataset),
        reference_dataset=compute_geostrophic_currents(glorys_dataset(challenger_dataset)),
        variables=[
            Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
            Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
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
