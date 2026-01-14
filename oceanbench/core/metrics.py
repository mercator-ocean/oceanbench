# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core.classIV import rmsd_class4
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_mixed_layer_depth
from oceanbench.core.derived_quantities import compute_geostrophic_currents
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.rmsd import rmsd
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import obs_insitu_dataset

from oceanbench.core.lagrangian_trajectory import (
    Zone,
    deviation_of_lagrangian_trajectories,
)


def rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=glorys_reanalysis_dataset(challenger_dataset),
        variables=[
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
            Variable.SEA_WATER_SALINITY,
            Variable.NORTHWARD_SEA_WATER_VELOCITY,
            Variable.EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd_class4(
        challenger_dataset=challenger_dataset,
        reference_dataset=obs_insitu_dataset(challenger_dataset),
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
    return rmsd(
        challenger_dataset=compute_mixed_layer_depth(challenger_dataset),
        reference_dataset=compute_mixed_layer_depth(glorys_reanalysis_dataset(challenger_dataset)),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=compute_geostrophic_currents(challenger_dataset),
        reference_dataset=compute_geostrophic_currents(glorys_reanalysis_dataset(challenger_dataset)),
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
        reference_dataset=glorys_reanalysis_dataset(challenger_dataset),
        zone=Zone.SMALL_ATLANTIC_NEWYORK_TO_NOUADHIBOU,
    )


def rmsd_of_variables_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=glo12_analysis_dataset(challenger_dataset),
        variables=[
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
            Variable.SEA_WATER_SALINITY,
            Variable.NORTHWARD_SEA_WATER_VELOCITY,
            Variable.EASTWARD_SEA_WATER_VELOCITY,
        ],
    )


"""from oceanbench.core.climate_forecast_standard_names import (
    VARIABLE_TO_OBSERVATION_MAPPING,
)"""

# from oceanbench.core.metrics import rmsd_class4, perform_matchup
# from oceanbench.core.metrics import VARIABLE_LABELS
"""
def rmsd_of_variables_compared_to_observations(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    #Compute RMSD of challenger dataset against observations using Class 4 methodology.
    obs_dict = obs_insitu_dataset(challenger_dataset)
    rmsd_results = {}

    for standard_var, (obs_source, obs_column) in VARIABLE_TO_OBSERVATION_MAPPING.items():
        var_key = standard_var.value

        if var_key not in challenger_dataset:
            continue

        obs_source_key = obs_source.value

        if obs_source_key not in obs_dict:
            continue

        obs_df = obs_dict[obs_source_key]

        if obs_column not in obs_df.columns:
            continue

        try:
            matchup_df = perform_matchup(challenger=challenger_dataset, obs_df=obs_df, var_name=var_key)

            if matchup_df.empty:
                continue

            rmsd_df = rmsd_class4(matchup_df=matchup_df, var_name=obs_column)

            variable_label = VARIABLE_LABELS.get(var_key, var_key)
            rmsd_results[variable_label] = rmsd_df["rmsd"].values

        except Exception:
            continue

    if not rmsd_results:
        return pandas.DataFrame()

    return pandas.DataFrame(rmsd_results, index=lead_day_labels(1, LEAD_DAYS_COUNT)).T
"""


def rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=compute_mixed_layer_depth(challenger_dataset),
        reference_dataset=compute_mixed_layer_depth(glo12_analysis_dataset(challenger_dataset)),
        variables=[
            Variable.MIXED_LAYER_DEPTH,
        ],
    )


def rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
    challenger_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return rmsd(
        challenger_dataset=compute_geostrophic_currents(challenger_dataset),
        reference_dataset=compute_geostrophic_currents(glo12_analysis_dataset(challenger_dataset)),
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
        reference_dataset=glo12_analysis_dataset(challenger_dataset),
        zone=Zone.SMALL_ATLANTIC_NEWYORK_TO_NOUADHIBOU,
    )
