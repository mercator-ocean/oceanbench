# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable

import pandas
import xarray

from oceanbench.core.classIV import rmsd_class4_validation
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents, compute_mixed_layer_depth
from oceanbench.core.lagrangian_trajectory import (
    deviation_of_lagrangian_trajectories,
    lagrangian_particle_count_for_region,
)
from oceanbench.core.marine_heatwaves import marine_heatwave_diagnostics, reference_history_dataset
from oceanbench.core.ocean_mask import ocean_mask
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import ObservationDataUnavailableError, observations
from oceanbench.core.regions import GLOBAL_REGION_NAME, RegionLike, subset_dataset_to_region
from oceanbench.core.rmsd import rmsd

GLOBAL_LAGRANGIAN_PARTICLE_COUNT = 10000
MINIMUM_LAGRANGIAN_PARTICLE_COUNT = 2000


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
        return pandas.DataFrame({"Message": [str(error)]})
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


def marine_heatwave_diagnostics_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
    climatology_path: str | None = None,
) -> pandas.DataFrame:
    return _marine_heatwave_diagnostics_compared_to_reference(
        challenger_dataset, glorys_reanalysis_dataset, region, climatology_path
    )


def marine_heatwave_diagnostics_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
    climatology_path: str | None = None,
) -> pandas.DataFrame:
    return _marine_heatwave_diagnostics_compared_to_reference(
        challenger_dataset, glo12_analysis_dataset, region, climatology_path
    )


def _marine_heatwave_diagnostics_compared_to_reference(
    challenger_dataset: xarray.Dataset,
    reference_loader: Callable[[xarray.Dataset], xarray.Dataset],
    region: RegionLike,
    climatology_path: str | None,
) -> pandas.DataFrame:
    if climatology_path is None:
        raise ValueError(
            "No global marine heatwave climatology is published yet: "
            "pass climatology_path, a heatwave3 ts2clm3 climatology covering the region."
        )
    regional_challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return marine_heatwave_diagnostics(
        challenger_dataset=regional_challenger_dataset,
        reference_dataset=subset_dataset_to_region(reference_loader(regional_challenger_dataset), region),
        history_dataset=subset_dataset_to_region(
            reference_history_dataset(regional_challenger_dataset, reference_loader), region
        ),
        climatology_path=climatology_path,
    )
