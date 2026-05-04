# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core.classIV_support import (
    compute_class4_rmsd_table,
    create_class4_observations_dataframe,
    format_class4_results,
    interpolate_class4_model_to_observations,
    prepare_class4_model_variable,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable


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
) -> xarray.DataArray:
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


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
) -> pandas.DataFrame:
    comparison_dataframe = class4_validation_dataframe(
        challenger_dataset=challenger_dataset,
        reference_dataset=reference_dataset,
        variables=variables,
    )
    return rmsd_class4_validation_dataframe(
        comparison_dataframe,
        lead_days_count=_lead_days_count(challenger_dataset),
    )


def rmsd_class4_validation_dataframe(
    comparison_dataframe: pandas.DataFrame,
    lead_days_count: int,
) -> pandas.DataFrame:
    if comparison_dataframe.empty:
        return pandas.DataFrame()

    all_results = []
    for standard_variable_key, observations_dataframe in comparison_dataframe.groupby("variable", sort=False):
        variable_results = _compute_rmsd_table(observations_dataframe, standard_variable_key)
        if not variable_results.empty:
            all_results.append(variable_results)

    if not all_results:
        return pandas.DataFrame()
    final_dataframe = pandas.concat(all_results, ignore_index=True)
    return format_class4_results(final_dataframe, lead_days_count)


def _lead_days_count(challenger_dataset: xarray.Dataset) -> int:
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    return challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]


def class4_validation_dataframe(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
) -> pandas.DataFrame:
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    observations = reference_dataset

    all_dataframes = []
    resolved_variables = [(variable.key(), variable.key(), variable.key()) for variable in variables]

    for standard_variable_key, observation_variable_key, challenger_variable_key in resolved_variables:
        if challenger_variable_key not in challenger or observation_variable_key not in observations:
            continue
        observations_dataframe = _create_observations_dataframe(
            observations,
            observation_variable_key,
            standard_variable_key,
            lead_days_count,
        )
        if observations_dataframe.empty:
            continue

        observations_dataframe = observations_dataframe.dropna(subset=["observation_value"])
        model_variable = _convert_forecast_ssh_to_sla(
            challenger[challenger_variable_key],
            standard_variable_key,
        )
        observations_dataframe["model_value"] = _interpolate_model_to_observations(
            model_variable,
            observations_dataframe,
        )
        observations_dataframe["error"] = (
            observations_dataframe["model_value"] - observations_dataframe["observation_value"]
        )
        observations_dataframe["absolute_error"] = observations_dataframe["error"].abs()
        observations_dataframe["variable"] = standard_variable_key
        all_dataframes.append(observations_dataframe)

    if not all_dataframes:
        return pandas.DataFrame()
    return pandas.concat(all_dataframes, ignore_index=True)
