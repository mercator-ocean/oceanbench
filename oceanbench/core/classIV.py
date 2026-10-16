# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

from oceanbench.core.classIV_support import (
    class4_observations_in_shared_population,
    class4_population_layers,
    compute_class4_rmsd_table,
    create_class4_observations_dataframe,
    format_class4_results,
    interpolate_class4_model_to_observations,
    prepare_class4_model_variable,
    reset_class4_observations_cache,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable

CLASS4_MATCHUP_COLUMNS = [
    "variable",
    "depth_bin",
    "lead_day",
    "first_day",
    Dimension.LATITUDE.key(),
    Dimension.LONGITUDE.key(),
    "observation_value",
    "model_value",
]


def _create_observations_dataframe(
    observations_dataset: xarray.Dataset,
    observation_variable_key: str,
    standard_variable_key: str,
    lead_days_count: int,
) -> pandas.DataFrame:
    return create_class4_observations_dataframe(
        observations_dataset,
        observation_variable_key,
        standard_variable_key,
        lead_days_count,
    )


def _interpolate_model_to_observations(
    model_data: xarray.DataArray,
    observations_dataframe: pandas.DataFrame,
) -> numpy.ndarray:
    return interpolate_class4_model_to_observations(model_data, observations_dataframe)


def _compute_rmsd_table(
    dataframe: pandas.DataFrame,
    variable_key: str,
) -> pandas.DataFrame:
    return compute_class4_rmsd_table(dataframe, variable_key)


def _convert_forecast_ssh_to_sla(
    model_variable: xarray.DataArray,
    variable_key: str,
) -> xarray.DataArray:
    return prepare_class4_model_variable(model_variable, variable_key)


def class4_matchups(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    ocean_mask: xarray.DataArray,
    variables: list[Variable],
) -> pandas.DataFrame:
    """
    One row per observation of the shared Class IV population, with the challenger value interpolated at it.

    The columns are ``variable``, ``depth_bin``, ``lead_day`` (starting at 0), ``first_day``, ``latitude``,
    ``longitude``, ``observation_value`` and ``model_value``. ``model_value`` is missing where the challenger
    has no value. The Class IV scores are computed from this table by ``class4_rmsd_table``.
    """
    reset_class4_observations_cache()
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    observations = reference_dataset
    population_layers = class4_population_layers(ocean_mask)

    all_matchups = []
    resolved_variables = [(variable.key(), variable.key(), variable.key()) for variable in variables]

    for standard_variable_key, observation_variable_key, challenger_variable_key in resolved_variables:
        observations_dataframe = _create_observations_dataframe(
            observations,
            observation_variable_key,
            standard_variable_key,
            lead_days_count,
        )
        if observations_dataframe.empty:
            continue

        observations_dataframe = class4_observations_in_shared_population(
            observations_dataframe.dropna(subset=["observation_value"]),
            population_layers,
        )
        if observations_dataframe.empty:
            continue

        model_variable = _convert_forecast_ssh_to_sla(
            challenger[challenger_variable_key],
            standard_variable_key,
        )
        observations_dataframe["model_value"] = _interpolate_model_to_observations(
            model_variable,
            observations_dataframe,
        )
        observations_dataframe["variable"] = standard_variable_key
        all_matchups.append(observations_dataframe[CLASS4_MATCHUP_COLUMNS])

    if not all_matchups:
        return pandas.DataFrame(columns=CLASS4_MATCHUP_COLUMNS)
    return pandas.concat(all_matchups, ignore_index=True)


def class4_rmsd_table(matchups: pandas.DataFrame, lead_days_count: int) -> pandas.DataFrame:
    all_results = []
    for variable_key, variable_matchups in matchups.groupby("variable", sort=False):
        variable_results = _compute_rmsd_table(variable_matchups, variable_key)
        if not variable_results.empty:
            all_results.append(variable_results)

    if not all_results:
        return pandas.DataFrame()
    final_dataframe = pandas.concat(all_results, ignore_index=True)
    return format_class4_results(final_dataframe, lead_days_count)


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    ocean_mask: xarray.DataArray,
    variables: list[Variable],
) -> pandas.DataFrame:
    return class4_rmsd_table(
        class4_matchups(challenger_dataset, reference_dataset, ocean_mask, variables),
        challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()],
    )
