# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from collections.abc import Callable, Iterator

import pandas
import xarray

from oceanbench.core.classIV import class4_rmsd_and_matchups, class4_rmsd_and_matchups_by_start
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents, compute_mixed_layer_depth
from oceanbench.core.lagrangian_trajectory import (
    deviation_of_lagrangian_trajectories,
    lagrangian_particle_count_for_region,
)
from oceanbench.core import mld_reuse
from oceanbench.core.ocean_mask import ocean_mask
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import (
    ObservationDataUnavailableError,
    has_increasing_start_dates,
    observations,
    observations_by_start,
)
from oceanbench.core.regions import GLOBAL_REGION_NAME, RegionDoesNotOverlapError, RegionLike, subset_dataset_to_region
from oceanbench.core.rmsd import rmsd, rmsd_together

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


def _start_observations_in_region(
    start_observations_datasets: Iterator[xarray.Dataset],
    region: RegionLike,
) -> Iterator[xarray.Dataset]:
    overlap_error = None
    overlaps = False
    for start_observations in start_observations_datasets:
        try:
            start_observations_in_region = subset_dataset_to_region(start_observations, region)
        except RegionDoesNotOverlapError as error:
            overlap_error = error
            continue
        overlaps = True
        yield start_observations_in_region
    if not overlaps and overlap_error is not None:
        raise overlap_error


def _class4_compared_to_observations(
    challenger_dataset: xarray.Dataset,
    region: RegionLike,
    keep_matchups: bool,
) -> tuple[pandas.DataFrame, pandas.DataFrame | None]:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    by_start = has_increasing_start_dates(challenger_dataset)
    try:
        if by_start:
            start_observations_datasets = _start_observations_in_region(
                observations_by_start(challenger_dataset), region
            )
        else:
            observation_dataset = subset_dataset_to_region(observations(challenger_dataset), region)
    except ObservationDataUnavailableError as error:
        return pandas.DataFrame({MESSAGE_COLUMN: [str(error)]}), None
    if by_start:
        return class4_rmsd_and_matchups_by_start(
            challenger_dataset=challenger_dataset,
            start_reference_datasets=start_observations_datasets,
            ocean_mask=ocean_mask(),
            variables=CLASS4_VARIABLES,
            keep_matchups=keep_matchups,
        )
    return class4_rmsd_and_matchups(
        challenger_dataset=challenger_dataset,
        reference_dataset=observation_dataset,
        ocean_mask=ocean_mask(),
        variables=CLASS4_VARIABLES,
        keep_matchups=keep_matchups,
    )


def class4_rmsd_and_matchups_compared_to_observations(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
) -> tuple[pandas.DataFrame, pandas.DataFrame | None]:
    return _class4_compared_to_observations(challenger_dataset, region, keep_matchups=True)


def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
) -> pandas.DataFrame:
    return _class4_compared_to_observations(challenger_dataset, region, keep_matchups=False)[0]


def rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
    keep_start_tables: bool = False,
) -> pandas.DataFrame | tuple[pandas.DataFrame, list[pandas.DataFrame]]:
    return rmsd(
        **_variables_rmsd_arguments(challenger_dataset, region, glorys_reanalysis_dataset),
        keep_start_tables=keep_start_tables,
    )


def rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
    keep_start_tables: bool = False,
) -> pandas.DataFrame | tuple[pandas.DataFrame, list[pandas.DataFrame]]:
    return rmsd(
        **_mld_rmsd_arguments(challenger_dataset, region, glorys_reanalysis_dataset),
        keep_start_tables=keep_start_tables,
    )


def rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
    keep_start_tables: bool = False,
) -> pandas.DataFrame | tuple[pandas.DataFrame, list[pandas.DataFrame]]:
    return rmsd(
        **_geo_rmsd_arguments(challenger_dataset, region, glorys_reanalysis_dataset),
        keep_start_tables=keep_start_tables,
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
    keep_start_tables: bool = False,
) -> pandas.DataFrame | tuple[pandas.DataFrame, list[pandas.DataFrame]]:
    return rmsd(
        **_variables_rmsd_arguments(challenger_dataset, region, glo12_analysis_dataset),
        keep_start_tables=keep_start_tables,
    )


def rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
    keep_start_tables: bool = False,
) -> pandas.DataFrame | tuple[pandas.DataFrame, list[pandas.DataFrame]]:
    return rmsd(
        **_mld_rmsd_arguments(challenger_dataset, region, glo12_analysis_dataset), keep_start_tables=keep_start_tables
    )


def rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
    keep_start_tables: bool = False,
) -> pandas.DataFrame | tuple[pandas.DataFrame, list[pandas.DataFrame]]:
    return rmsd(
        **_geo_rmsd_arguments(challenger_dataset, region, glo12_analysis_dataset), keep_start_tables=keep_start_tables
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


_VARIABLES_COMPARED_TO_GRIDDED_REFERENCES = [
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    Variable.SEA_WATER_SALINITY,
    Variable.NORTHWARD_SEA_WATER_VELOCITY,
    Variable.EASTWARD_SEA_WATER_VELOCITY,
]


def _variables_rmsd_arguments(
    challenger_dataset: xarray.Dataset,
    region: RegionLike,
    reference_dataset_of: Callable[[xarray.Dataset], xarray.Dataset],
) -> dict:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return dict(
        challenger_dataset=challenger_dataset,
        reference_dataset=subset_dataset_to_region(reference_dataset_of(challenger_dataset), region),
        variables=_VARIABLES_COMPARED_TO_GRIDDED_REFERENCES,
        ocean_mask=ocean_mask(),
    )


def _mld_rmsd_arguments(
    challenger_dataset: xarray.Dataset,
    region: RegionLike,
    reference_dataset_of: Callable[[xarray.Dataset], xarray.Dataset],
) -> dict:
    # The mixed layer depth of the challenger is computed once per run and spilled to local disk, and that of
    # the reference is read from the reference store when it has it (mld_reuse).
    run_key = (id(challenger_dataset), repr(region))
    regional_challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return dict(
        challenger_dataset=mld_reuse.challenger_mixed_layer_depth(
            run_key, regional_challenger_dataset, compute_mixed_layer_depth
        ),
        reference_dataset=mld_reuse.reference_mixed_layer_depth(
            run_key + (reference_dataset_of.__name__,),
            subset_dataset_to_region(reference_dataset_of(regional_challenger_dataset), region),
            compute_mixed_layer_depth,
            keep=challenger_dataset,
        ),
        variables=[Variable.MIXED_LAYER_DEPTH],
        ocean_mask=ocean_mask(),
    )


def _geo_rmsd_arguments(
    challenger_dataset: xarray.Dataset,
    region: RegionLike,
    reference_dataset_of: Callable[[xarray.Dataset], xarray.Dataset],
) -> dict:
    challenger_dataset = subset_dataset_to_region(challenger_dataset, region)
    return dict(
        challenger_dataset=compute_geostrophic_currents(challenger_dataset),
        reference_dataset=compute_geostrophic_currents(
            subset_dataset_to_region(reference_dataset_of(challenger_dataset), region)
        ),
        variables=[
            Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
            Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
        ],
        ocean_mask=ocean_mask(),
    )


_GRIDDED_RMSD_ARGUMENTS = {
    "rmsd_of_variables_compared_to_glorys_reanalysis": (_variables_rmsd_arguments, glorys_reanalysis_dataset),
    "rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis": (_mld_rmsd_arguments, glorys_reanalysis_dataset),
    "rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis": (_geo_rmsd_arguments, glorys_reanalysis_dataset),
    "rmsd_of_variables_compared_to_glo12_analysis": (_variables_rmsd_arguments, glo12_analysis_dataset),
    "rmsd_of_mixed_layer_depth_compared_to_glo12_analysis": (_mld_rmsd_arguments, glo12_analysis_dataset),
    "rmsd_of_geostrophic_currents_compared_to_glo12_analysis": (_geo_rmsd_arguments, glo12_analysis_dataset),
}


def _gridded_rmsd_metrics_together(
    metric_function_names: list[str],
    challenger_dataset: xarray.Dataset,
    region: RegionLike = GLOBAL_REGION_NAME,
    keep_start_tables: bool = False,
) -> dict[str, pandas.DataFrame | tuple[pandas.DataFrame, list[pandas.DataFrame]]]:
    """
    The result of each of these gridded RMSD metric functions, computed in one pass so that the challenger,
    and what is derived from it such as its mixed layer depth, is read and computed once for all of them.
    Each result is the one of the metric function called alone.
    """
    results = rmsd_together(
        [
            {
                **_GRIDDED_RMSD_ARGUMENTS[name][0](challenger_dataset, region, _GRIDDED_RMSD_ARGUMENTS[name][1]),
                "keep_start_tables": keep_start_tables,
            }
            for name in metric_function_names
        ]
    )
    return dict(zip(metric_function_names, results))
