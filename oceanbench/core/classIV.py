# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
import xarray

from oceanbench.core.classIV_support import (
    compute_class4_rmsd_table,
    compute_class4_rmsd_table_per_start,
    create_class4_observations_dataframe,
    format_class4_results,
    interpolate_class4_model_to_observations,
    prepare_class4_model_variable,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_source import get_dataset_source
from oceanbench.core.dataset_utils import Dimension, Variable


def _challenger_name(challenger_dataset: xarray.Dataset) -> str | None:
    challenger_source = get_dataset_source(challenger_dataset)
    return challenger_source.name if challenger_source is not None else None


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
    challenger_name: str | None = None,
) -> xarray.DataArray:
    return prepare_class4_model_variable(model_variable, variable_key, challenger_name)


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
) -> pandas.DataFrame:
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    challenger_name = _challenger_name(challenger_dataset)
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    observations = reference_dataset

    all_results = []
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

        observations_dataframe = observations_dataframe.dropna(subset=["observation_value"])
        model_variable = _convert_forecast_ssh_to_sla(
            challenger[challenger_variable_key],
            standard_variable_key,
            challenger_name,
        )
        observations_dataframe["model_value"] = _interpolate_model_to_observations(
            model_variable,
            observations_dataframe,
        )

        variable_results = _compute_rmsd_table(observations_dataframe, standard_variable_key)
        if not variable_results.empty:
            all_results.append(variable_results)

    if not all_results:
        return pandas.DataFrame()
    final_dataframe = pandas.concat(all_results, ignore_index=True)
    return format_class4_results(final_dataframe, lead_days_count)


def _class4_matchups_per_variable(
    challenger: xarray.Dataset,
    observations: xarray.Dataset,
    variables: list[Variable],
    challenger_name: str | None = None,
) -> list[pandas.DataFrame]:
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    per_variable_tables = []
    for variable in variables:
        variable_key = variable.key()
        observations_dataframe = _create_observations_dataframe(
            observations,
            variable_key,
            variable_key,
            lead_days_count,
        )
        if observations_dataframe.empty:
            continue
        observations_dataframe = observations_dataframe.dropna(subset=["observation_value"])
        model_variable = _convert_forecast_ssh_to_sla(challenger[variable_key], variable_key, challenger_name)
        observations_dataframe["model_value"] = _interpolate_model_to_observations(
            model_variable,
            observations_dataframe,
        )
        per_start_table = compute_class4_rmsd_table_per_start(observations_dataframe, variable_key)
        if not per_start_table.empty:
            per_variable_tables.append(per_start_table)
    return per_variable_tables


def rmsd_class4_validation_per_start(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    variables: list[Variable],
) -> pandas.DataFrame:
    """Per-forecast-start Class-4 RMSD in long form (one row per start x variable x depth_bin x lead_day).

    Each row's ``rmsd`` is the RMSD over the observations of a single forecast start
    (``first_day``); ``count`` is that observation count. Columns:
    ``variable, first_day, depth_bin, lead_day, rmsd, count``. The published
    pooled-over-observations value is recovered exactly by
    :func:`oceanbench.core.classIV_support.recombine_class4_pooled_from_per_start`.
    ``lead_day`` here is 0-based (0 == the forecast's first day), matching
    :func:`rmsd_class4_validation`.
    """
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    challenger_name = _challenger_name(challenger_dataset)
    per_variable_tables = _class4_matchups_per_variable(challenger, reference_dataset, variables, challenger_name)
    if not per_variable_tables:
        return pandas.DataFrame()
    return pandas.concat(per_variable_tables, ignore_index=True)
