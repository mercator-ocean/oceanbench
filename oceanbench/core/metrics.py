# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable

import pandas
import xarray

from oceanbench.core.classIV import rmsd_class4_validation
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents, compute_mixed_layer_depth
from oceanbench.core.marine_heatwave_climatology import (
    marine_heatwave_climatology_is_available,
    marine_heatwave_climatology_mean_and_percentile_90,
)
from oceanbench.core.marine_heatwave_history import (
    glo12_analysis_history_dataset,
    glorys_reanalysis_history_dataset,
)
from oceanbench.core.marine_heatwaves import marine_heatwave_diagnostics
from oceanbench.core.lagrangian_trajectory import (
    deviation_of_lagrangian_trajectories,
    lagrangian_particle_count_for_region,
)
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import ObservationDataUnavailableError, observations
from oceanbench.core.regions import GLOBAL_REGION_NAME, RegionLike, subset_dataset_to_region
from oceanbench.core.rmsd import rmsd

GLOBAL_LAGRANGIAN_PARTICLE_COUNT = 10000
MINIMUM_LAGRANGIAN_PARTICLE_COUNT = 2000
MARINE_HEATWAVE_HISTORY_DAYS = 7


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


def marine_heatwave_diagnostics_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    unavailable_result = _marine_heatwave_unavailable_result(challenger_dataset)
    if unavailable_result is not None:
        return unavailable_result

    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    reference_dataset = subset_dataset_to_region(glorys_reanalysis_dataset(challenger_dataset), region)
    challenger_history_dataset = _marine_heatwave_history_dataset(
        challenger_dataset,
        glo12_analysis_history_dataset,
        region,
    )
    reference_history_dataset = _marine_heatwave_history_dataset(
        challenger_dataset,
        glorys_reanalysis_history_dataset,
        region,
    )

    return _marine_heatwave_diagnostics_against_reference(
        challenger_dataset,
        reference_dataset,
        challenger_history_dataset,
        reference_history_dataset,
        region,
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


def marine_heatwave_diagnostics_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    unavailable_result = _marine_heatwave_unavailable_result(challenger_dataset)
    if unavailable_result is not None:
        return unavailable_result

    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    reference_dataset = subset_dataset_to_region(glo12_analysis_dataset(challenger_dataset), region)
    history_dataset = _marine_heatwave_history_dataset(
        challenger_dataset,
        glo12_analysis_history_dataset,
        region,
    )

    return _marine_heatwave_diagnostics_against_reference(
        challenger_dataset,
        reference_dataset,
        history_dataset,
        history_dataset,
        region,
    )


def _marine_heatwave_diagnostics_against_reference(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    challenger_history_dataset: xarray.Dataset | None,
    reference_history_dataset: xarray.Dataset | None,
    region: RegionLike,
) -> pandas.DataFrame:
    (
        climatology_mean,
        percentile_90,
    ) = marine_heatwave_climatology_mean_and_percentile_90(challenger_dataset)
    climatology_mean = _subset_dataarray_to_region(climatology_mean, region)
    percentile_90 = _subset_dataarray_to_region(percentile_90, region)

    return marine_heatwave_diagnostics(
        challenger_dataset=challenger_dataset,
        reference_dataset=reference_dataset,
        climatology_mean=climatology_mean,
        percentile_90=percentile_90,
        challenger_history_dataset=challenger_history_dataset,
        reference_history_dataset=reference_history_dataset,
    )


def _marine_heatwave_unavailable_result(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame | None:
    if marine_heatwave_climatology_is_available(challenger_dataset):
        return None

    return pandas.DataFrame(
        {
            "Message": [
                "Marine Heatwave diagnostics are unavailable because no "
                "compatible climatology is configured for this grid resolution."
            ]
        }
    )


def _marine_heatwave_history_dataset(
    challenger_dataset: xarray.Dataset,
    history_loader: Callable[[xarray.Dataset, int], xarray.Dataset],
    region: RegionLike,
) -> xarray.Dataset | None:
    first_day_dimension = Dimension.FIRST_DAY_DATETIME.key()
    if challenger_dataset.sizes[first_day_dimension] < 2:
        return None

    available_history = history_loader(
        challenger_dataset.isel({first_day_dimension: slice(1, None)}),
        MARINE_HEATWAVE_HISTORY_DAYS,
    )
    regional_history = subset_dataset_to_region(available_history, region)
    return regional_history.reindex({first_day_dimension: challenger_dataset[first_day_dimension]})


def _subset_dataarray_to_region(data: xarray.DataArray, region: RegionLike) -> xarray.DataArray:
    variable_name = data.name or "variable"
    return subset_dataset_to_region(data.to_dataset(name=variable_name), region)[variable_name]
