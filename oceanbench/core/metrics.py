# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core.classIV import rmsd_class4_validation, rmsd_class4_validation_by_start
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents, compute_mixed_layer_depth
from oceanbench.core.lagrangian_trajectory import (
    deviation_of_lagrangian_trajectories,
    lagrangian_particle_count_for_region,
)
from oceanbench.core.ocean_mask import ocean_mask
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import (
    ObservationDataUnavailableError,
    has_increasing_start_dates,
    observations,
    observations_by_start,
)
from oceanbench.core.regions import GLOBAL_REGION_NAME, RegionLike, resolve_region, subset_dataset_to_region
from oceanbench.core.rmsd import rmsd

GLOBAL_LAGRANGIAN_PARTICLE_COUNT = 10000
MINIMUM_LAGRANGIAN_PARTICLE_COUNT = 2000
MESSAGE_COLUMN = "Message"


def _lagrangian_particle_count(
    global_challenger_dataset: xarray.Dataset,
    regional_challenger_dataset: xarray.Dataset,
) -> int:
    return lagrangian_particle_count_for_region(
        global_challenger_dataset,
        regional_challenger_dataset,
        global_particle_count=GLOBAL_LAGRANGIAN_PARTICLE_COUNT,
        minimum_particle_count=MINIMUM_LAGRANGIAN_PARTICLE_COUNT,
    )


CLASS4_VARIABLES = [
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    Variable.SEA_WATER_SALINITY,
    Variable.NORTHWARD_SEA_WATER_VELOCITY,
    Variable.EASTWARD_SEA_WATER_VELOCITY,
]


def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    # The observations are opened one start date at a time only without a region bounding box: the region subset
    # of the observations of one start date can differ from that of all of them (empty overlap, longitude range).
    by_start = resolve_region(region).bounds is None and has_increasing_start_dates(challenger_dataset)
    try:
        if by_start:
            start_observation_datasets = observations_by_start(challenger_dataset)
        else:
            observation_dataset = subset_dataset_to_region(observations(challenger_dataset), region)
    except ObservationDataUnavailableError as error:
        return pandas.DataFrame({MESSAGE_COLUMN: [str(error)]})
    if by_start:
        return rmsd_class4_validation_by_start(
            challenger_dataset=challenger_dataset,
            start_reference_datasets=start_observation_datasets,
            ocean_mask=ocean_mask(),
            variables=CLASS4_VARIABLES,
        )
    return rmsd_class4_validation(
        challenger_dataset=challenger_dataset,
        reference_dataset=observation_dataset,
        ocean_mask=ocean_mask(),
        variables=CLASS4_VARIABLES,
    )


def rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
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
        ocean_mask=ocean_mask(),
    )


def rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
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
        ocean_mask=ocean_mask(),
    )


def rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
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
        ocean_mask=ocean_mask(),
    )


def deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    regional_challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return deviation_of_lagrangian_trajectories(
        challenger_dataset=regional_challenger_dataset,
        reference_dataset=subset_dataset_to_region(glorys_reanalysis_dataset(regional_challenger_dataset), region),
        particle_count=_lagrangian_particle_count(challenger_dataset, regional_challenger_dataset),
    )


def rmsd_of_variables_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
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
        ocean_mask=ocean_mask(),
    )


def rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
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
        ocean_mask=ocean_mask(),
    )


def rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
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
        ocean_mask=ocean_mask(),
    )


def deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    regional_challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return deviation_of_lagrangian_trajectories(
        challenger_dataset=regional_challenger_dataset,
        reference_dataset=subset_dataset_to_region(glo12_analysis_dataset(regional_challenger_dataset), region),
        particle_count=_lagrangian_particle_count(challenger_dataset, regional_challenger_dataset),
    )
