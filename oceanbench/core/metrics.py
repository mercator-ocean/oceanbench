# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core.class4_drifters import deviation_of_lagrangian_trajectories_compared_to_class4_observations as class4_drifter_trajectory_deviation
from oceanbench.core import regions
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_mixed_layer_depth
from oceanbench.core.derived_quantities import compute_geostrophic_currents
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.rmsd import rmsd
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.classIV import rmsd_class4_validation
from oceanbench.core.references.observations import observations

from oceanbench.core.lagrangian_trajectory import (
    deviation_of_lagrangian_trajectories,
)

OBSERVATIONS_UNAVAILABLE_ERROR_PREFIX = "OBSERVATIONS_NOT_AVAILABLE:"


def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    try:
        observation_dataset = regions.filter_observation_dataset_from_challenger_region(
            observations(challenger_dataset),
            challenger_dataset=challenger_dataset,
        )
    except ValueError as error:
        error_message = str(error)
        if error_message.startswith(OBSERVATIONS_UNAVAILABLE_ERROR_PREFIX):
            return pandas.DataFrame(
                {"Message": [error_message.replace(OBSERVATIONS_UNAVAILABLE_ERROR_PREFIX, "", 1).strip()]}
            )
        raise
    result = rmsd_class4_validation(
        challenger_dataset=challenger_dataset,
        reference_dataset=observation_dataset,
        variables=[
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
            Variable.SEA_WATER_SALINITY,
            Variable.NORTHWARD_SEA_WATER_VELOCITY,
            Variable.EASTWARD_SEA_WATER_VELOCITY,
        ],
    )

    return result


def rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=regions.subset_dataset_from_challenger_region(
            glorys_reanalysis_dataset(challenger_dataset),
            challenger_dataset=challenger_dataset,
        ),
        variables=[
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
            Variable.SEA_WATER_SALINITY,
            Variable.NORTHWARD_SEA_WATER_VELOCITY,
            Variable.EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


def rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    glorys_dataset = regions.subset_dataset_from_challenger_region(
        glorys_reanalysis_dataset(challenger_dataset),
        challenger_dataset=challenger_dataset,
    )
    return rmsd(
        challenger_dataset=compute_mixed_layer_depth(challenger_dataset),
        reference_dataset=compute_mixed_layer_depth(glorys_dataset),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    glorys_dataset = regions.subset_dataset_from_challenger_region(
        glorys_reanalysis_dataset(challenger_dataset),
        challenger_dataset=challenger_dataset,
    )
    return rmsd(
        challenger_dataset=compute_geostrophic_currents(challenger_dataset),
        reference_dataset=compute_geostrophic_currents(glorys_dataset),
        variables=[
            Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
            Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


def deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return deviation_of_lagrangian_trajectories(
        challenger_dataset=challenger_dataset,
        reference_dataset=regions.subset_dataset_from_challenger_region(
            glorys_reanalysis_dataset(challenger_dataset),
            challenger_dataset=challenger_dataset,
        ),
    )


def rmsd_of_variables_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=regions.subset_dataset_from_challenger_region(
            glo12_analysis_dataset(challenger_dataset),
            challenger_dataset=challenger_dataset,
        ),
        variables=[
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
            Variable.SEA_WATER_SALINITY,
            Variable.NORTHWARD_SEA_WATER_VELOCITY,
            Variable.EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


def rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    glo12_dataset = regions.subset_dataset_from_challenger_region(
        glo12_analysis_dataset(challenger_dataset),
        challenger_dataset=challenger_dataset,
    )
    return rmsd(
        challenger_dataset=compute_mixed_layer_depth(challenger_dataset),
        reference_dataset=compute_mixed_layer_depth(glo12_dataset),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    glo12_dataset = regions.subset_dataset_from_challenger_region(
        glo12_analysis_dataset(challenger_dataset),
        challenger_dataset=challenger_dataset,
    )
    return rmsd(
        challenger_dataset=compute_geostrophic_currents(challenger_dataset),
        reference_dataset=compute_geostrophic_currents(glo12_dataset),
        variables=[
            Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
            Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


def deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return deviation_of_lagrangian_trajectories(
        challenger_dataset=challenger_dataset,
        reference_dataset=regions.subset_dataset_from_challenger_region(
            glo12_analysis_dataset(challenger_dataset),
            challenger_dataset=challenger_dataset,
        ),
    )


def deviation_of_lagrangian_trajectories_compared_to_class4_observations(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    try:
        observation_dataset = regions.filter_observation_dataset_from_challenger_region(
            observations(challenger_dataset),
            challenger_dataset=challenger_dataset,
        )
    except ValueError as error:
        error_message = str(error)
        if error_message.startswith(OBSERVATIONS_UNAVAILABLE_ERROR_PREFIX):
            return pandas.DataFrame(
                {"Message": [error_message.replace(OBSERVATIONS_UNAVAILABLE_ERROR_PREFIX, "", 1).strip()]}
            )
        raise

    return class4_drifter_trajectory_deviation(
        challenger_dataset=challenger_dataset,
        observation_dataset=observation_dataset,
    )
