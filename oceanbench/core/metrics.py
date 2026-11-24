# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core.classIV import rmsd_class4_validation
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents, compute_mixed_layer_depth
from oceanbench.core.lagrangian_trajectory import (
    deviation_of_lagrangian_trajectories,
    lagrangian_particle_count_for_region,
)
from oceanbench.core.ocean_mask import ocean_mask
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.climatology import glorys_monthly_climatology_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import ObservationDataUnavailableError, observations
from oceanbench.core.regions import GLOBAL_REGION_NAME, RegionLike, subset_dataset_to_region
from oceanbench.core.rmsd import rmsd
from oceanbench.core.smoothing_controlled_gridded_rmsd import smoothing_controlled_rmsd_gridded
from oceanbench.core.smoothing_controlled_rmsd import (
    GRID_SMOOTHING_METHOD,
    grid_extent,
    region_with_smoothing_margin,
    smoothing_controlled_rmsd_class4_validation,
)

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


def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    try:
        observation_dataset = subset_dataset_to_region(observations(challenger_dataset), region)
    except ObservationDataUnavailableError as error:
        return pandas.DataFrame({MESSAGE_COLUMN: [str(error)]})
    return rmsd_class4_validation(
        challenger_dataset=challenger_dataset,
        reference_dataset=observation_dataset,
        ocean_mask=ocean_mask(),
        variables=[
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
            Variable.SEA_WATER_SALINITY,
            Variable.NORTHWARD_SEA_WATER_VELOCITY,
            Variable.EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


def smoothing_controlled_rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
    climatology_dataset: xarray.Dataset | None = None,
    smoothing_method: str = GRID_SMOOTHING_METHOD,
) -> pandas.DataFrame:
    regional_challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    try:
        observation_dataset = subset_dataset_to_region(observations(regional_challenger_dataset), region)
    except ObservationDataUnavailableError as error:
        return pandas.DataFrame({MESSAGE_COLUMN: [str(error)]})
    # The challenger is smoothed with a margin around the region; point values stay those of the region cut.
    smoothing_challenger_dataset = subset_dataset_to_region(challenger_dataset, region_with_smoothing_margin(region))
    is_cut = dict(smoothing_challenger_dataset.sizes) != dict(regional_challenger_dataset.sizes)
    return smoothing_controlled_rmsd_class4_validation(
        challenger_dataset=smoothing_challenger_dataset,
        reference_dataset=observation_dataset,
        ocean_mask=ocean_mask(),
        variables=[
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
            Variable.SEA_WATER_SALINITY,
            Variable.NORTHWARD_SEA_WATER_VELOCITY,
            Variable.EASTWARD_SEA_WATER_VELOCITY,
        ],
        climatology_dataset=(
            climatology_dataset if climatology_dataset is not None else glorys_monthly_climatology_dataset()
        ),
        smoothing_method=smoothing_method,
        point_value_extent=(
            grid_extent(rename_dataset_with_standard_names(regional_challenger_dataset)) if is_cut else None
        ),
    )


def _smoothing_controlled_rmsd_compared_to_gridded_reference(
    challenger_dataset: xarray.Dataset,
    region: RegionLike,
    reference_dataset_of_challenger,
    climatology_dataset: xarray.Dataset | None,
) -> pandas.DataFrame:
    regional_challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    # The challenger is smoothed with a margin around the region; only the cells of the region cut are scored.
    smoothing_region = region_with_smoothing_margin(region)
    smoothing_challenger_dataset = subset_dataset_to_region(challenger_dataset, smoothing_region)
    is_cut = dict(smoothing_challenger_dataset.sizes) != dict(regional_challenger_dataset.sizes)
    return smoothing_controlled_rmsd_gridded(
        challenger_dataset=smoothing_challenger_dataset,
        reference_dataset=subset_dataset_to_region(
            reference_dataset_of_challenger(smoothing_challenger_dataset), smoothing_region
        ),
        variables=[
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
            Variable.SEA_WATER_SALINITY,
            Variable.NORTHWARD_SEA_WATER_VELOCITY,
            Variable.EASTWARD_SEA_WATER_VELOCITY,
        ],
        ocean_mask=ocean_mask(),
        climatology_dataset=(
            climatology_dataset if climatology_dataset is not None else glorys_monthly_climatology_dataset()
        ),
        scored_extent=(
            grid_extent(rename_dataset_with_standard_names(regional_challenger_dataset)) if is_cut else None
        ),
    )


def smoothing_controlled_rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
    climatology_dataset: xarray.Dataset | None = None,
) -> pandas.DataFrame:
    return _smoothing_controlled_rmsd_compared_to_gridded_reference(
        challenger_dataset, region, glorys_reanalysis_dataset, climatology_dataset
    )


def smoothing_controlled_rmsd_of_variables_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
    climatology_dataset: xarray.Dataset | None = None,
) -> pandas.DataFrame:
    return _smoothing_controlled_rmsd_compared_to_gridded_reference(
        challenger_dataset, region, glo12_analysis_dataset, climatology_dataset
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
