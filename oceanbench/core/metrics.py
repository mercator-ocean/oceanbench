# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core.classIV import rmsd_class4_validation
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents, compute_mixed_layer_depth
from oceanbench.core.lagrangian_trajectory import deviation_of_lagrangian_trajectories
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import ObservationDataUnavailableError, observations
from oceanbench.core.regions import GLOBAL_REGION_NAME, subset_dataset_to_region
from oceanbench.core.rmsd import rmsd

GLOBAL_LAGRANGIAN_PARTICLE_COUNT = 10000
REGIONAL_LAGRANGIAN_PARTICLE_COUNT = 1000


def _lagrangian_particle_count(region: str) -> int:
    if region == GLOBAL_REGION_NAME:
        return GLOBAL_LAGRANGIAN_PARTICLE_COUNT
    return REGIONAL_LAGRANGIAN_PARTICLE_COUNT


def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
    region: str = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    try:
        observation_dataset = subset_dataset_to_region(observations(challenger_dataset), region)
    except ObservationDataUnavailableError as error:
        return pandas.DataFrame({"Message": [str(error)]})
    return rmsd_class4_validation(
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


def rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: str = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=subset_dataset_to_region(glorys_reanalysis_dataset(challenger_dataset), region),
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
    region: str = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return rmsd(
        challenger_dataset=compute_mixed_layer_depth(challenger_dataset),
        reference_dataset=compute_mixed_layer_depth(
            subset_dataset_to_region(glorys_reanalysis_dataset(challenger_dataset), region)
        ),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: str = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return rmsd(
        challenger_dataset=compute_geostrophic_currents(challenger_dataset),
        reference_dataset=compute_geostrophic_currents(
            subset_dataset_to_region(glorys_reanalysis_dataset(challenger_dataset), region)
        ),
        variables=[
            Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
            Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


def deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: str = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return deviation_of_lagrangian_trajectories(
        challenger_dataset=challenger_dataset,
        reference_dataset=subset_dataset_to_region(glorys_reanalysis_dataset(challenger_dataset), region),
        particle_count=_lagrangian_particle_count(region),
    )


def rmsd_of_variables_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: str = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=subset_dataset_to_region(glo12_analysis_dataset(challenger_dataset), region),
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
    region: str = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return rmsd(
        challenger_dataset=compute_mixed_layer_depth(challenger_dataset),
        reference_dataset=compute_mixed_layer_depth(
            subset_dataset_to_region(glo12_analysis_dataset(challenger_dataset), region)
        ),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: str = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return rmsd(
        challenger_dataset=compute_geostrophic_currents(challenger_dataset),
        reference_dataset=compute_geostrophic_currents(
            subset_dataset_to_region(glo12_analysis_dataset(challenger_dataset), region)
        ),
        variables=[
            Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
            Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


def deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: str = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return deviation_of_lagrangian_trajectories(
        challenger_dataset=challenger_dataset,
        reference_dataset=subset_dataset_to_region(glo12_analysis_dataset(challenger_dataset), region),
        particle_count=_lagrangian_particle_count(region),
    )
