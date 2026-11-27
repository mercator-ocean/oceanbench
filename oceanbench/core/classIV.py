# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor

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
    REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT,
    prepare_class4_model_variable,
    sea_surface_height_shift,
    reset_class4_observations_cache,
)
from oceanbench.core.classIV_matchups import Class4Matchups
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
    mean_sea_surface_height_shift: float = REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT,
) -> xarray.DataArray:
    return prepare_class4_model_variable(model_variable, variable_key, mean_sea_surface_height_shift)


def _observations_in_shared_population(
    observations: xarray.Dataset,
    standard_variable_key: str,
    lead_days_count: int,
    population_layers,
) -> pandas.DataFrame | None:
    """The observations of the shared population, ``None`` without any."""
    observations_dataframe = _create_observations_dataframe(
        observations,
        standard_variable_key,
        standard_variable_key,
        lead_days_count,
    )
    if observations_dataframe.empty:
        return None
    observations_dataframe = class4_observations_in_shared_population(
        observations_dataframe.dropna(subset=["observation_value"]),
        population_layers,
    )
    if observations_dataframe.empty:
        return None
    return observations_dataframe


def _with_model_values(observations_dataframe: pandas.DataFrame | None, model_variable) -> pandas.DataFrame | None:
    if observations_dataframe is None:
        return None
    observations_dataframe["model_value"] = _interpolate_model_to_observations(
        model_variable(),
        observations_dataframe,
    )
    return observations_dataframe


def _matched_observations(
    observations: xarray.Dataset,
    standard_variable_key: str,
    lead_days_count: int,
    population_layers,
    model_variable,
) -> pandas.DataFrame | None:
    """The observations of the shared population with the challenger value at them, ``None`` without any."""
    return _with_model_values(
        _observations_in_shared_population(observations, standard_variable_key, lead_days_count, population_layers),
        model_variable,
    )


def _class4_observations_ahead() -> int:
    """Number of observation tables prepared ahead of the interpolation (OCEANBENCH_CLASS4_OBSERVATIONS_AHEAD, default 1)."""
    return max(0, int(os.environ.get("OCEANBENCH_CLASS4_OBSERVATIONS_AHEAD", "1")))


def _observations_by_start_and_variable(
    start_observations_datasets: Iterable[xarray.Dataset],
    variable_keys: list[str],
    lead_days_count: int,
    population_layers,
) -> Iterator[tuple[str, pandas.DataFrame | None]]:
    for start_observations in start_observations_datasets:
        # Without this, the prepared observations of every start date would stay in the cache.
        reset_class4_observations_cache()
        for variable_key in variable_keys:
            yield variable_key, _observations_in_shared_population(
                start_observations, variable_key, lead_days_count, population_layers
            )


def _read_ahead(items: Iterator, ahead_count: int) -> Iterator:
    """The items in order, the next ``ahead_count`` ones produced by one background thread meanwhile."""
    if ahead_count == 0:
        yield from items
        return
    end_of_items = object()
    with ThreadPoolExecutor(max_workers=1) as producer:
        pending = [producer.submit(next, items, end_of_items) for _ in range(ahead_count)]
        while True:
            item = pending.pop(0).result()
            if item is end_of_items:
                return
            pending.append(producer.submit(next, items, end_of_items))
            yield item


def _model_variables(challenger_dataset: xarray.Dataset, variable_keys: list[str]) -> dict:
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    prepared: dict[str, xarray.DataArray] = {}

    def model_variable(variable_key: str):
        if variable_key not in prepared:
            prepared[variable_key] = _convert_forecast_ssh_to_sla(
                challenger[variable_key],
                variable_key,
                sea_surface_height_shift(challenger_dataset),
            )
        return prepared[variable_key]

    return {variable_key: (lambda key=variable_key: model_variable(key)) for variable_key in variable_keys}


def _start_observation_slices(observations: xarray.Dataset) -> list[slice] | None:
    """The contiguous observation range of each start date, ``None`` when they are not sorted by start date."""
    first_days = observations[Dimension.FIRST_DAY_DATETIME.key()].values
    if len(first_days) == 0 or numpy.isnat(first_days).any() or (first_days[1:] < first_days[:-1]).any():
        return None
    edges = [0, *(numpy.flatnonzero(first_days[1:] != first_days[:-1]) + 1).tolist(), len(first_days)]
    return [slice(begin, end) for begin, end in zip(edges[:-1], edges[1:])]


def _rmsd_table_from_parts(matchups: Class4Matchups, variable_key: str) -> pandas.DataFrame:
    # A (depth bin, lead day) group lies in one lead day part, with its rows in the order of the table.
    tables = [
        _compute_rmsd_table(matchups.part(variable_key, lead_day), variable_key)
        for lead_day in matchups.lead_days(variable_key)
    ]
    return (
        pandas.concat(tables, ignore_index=True)
        .sort_values(["depth_bin", "lead_day"], kind="stable")
        .reset_index(drop=True)
    )


def _class4_by_start(
    start_observations_datasets: Iterable[xarray.Dataset],
    variable_keys: list[str],
    lead_days_count: int,
    population_layers,
    model_variables: dict,
) -> tuple[list[pandas.DataFrame], Class4Matchups]:
    matchups = Class4Matchups.on_disk(variable_keys)
    # The observation table of the next variable (or start date) is prepared while the challenger is
    # interpolated at the current one.
    prepared_observations = _observations_by_start_and_variable(
        start_observations_datasets, variable_keys, lead_days_count, population_layers
    )
    for variable_key, observations_dataframe in _read_ahead(prepared_observations, _class4_observations_ahead()):
        observations_dataframe = _with_model_values(observations_dataframe, model_variables[variable_key])
        if observations_dataframe is not None:
            observations_dataframe["variable"] = variable_key
            matchups.add(variable_key, observations_dataframe[CLASS4_MATCHUP_COLUMNS])
        del observations_dataframe
    reset_class4_observations_cache()
    tables = [_rmsd_table_from_parts(matchups, variable_key) for variable_key in matchups.variables()]
    return [table for table in tables if not table.empty], matchups


def _start_slices_observations(observations: xarray.Dataset, start_slices: list[slice]) -> Iterator[xarray.Dataset]:
    observation_dimension = observations[Dimension.TIME.key()].dims[0]
    return (observations.isel({observation_dimension: start_slice}) for start_slice in start_slices)


def _class4_by_variable(
    observations: xarray.Dataset,
    variable_keys: list[str],
    lead_days_count: int,
    population_layers,
    model_variables: dict,
    keep_matchups: bool,
) -> tuple[list[pandas.DataFrame], Class4Matchups | None]:
    matchups = Class4Matchups.on_disk(variable_keys) if keep_matchups else None
    tables = []
    for variable_key in variable_keys:
        observations_dataframe = _matched_observations(
            observations, variable_key, lead_days_count, population_layers, model_variables[variable_key]
        )
        if observations_dataframe is None:
            continue
        variable_results = _compute_rmsd_table(observations_dataframe, variable_key)
        if not variable_results.empty:
            tables.append(variable_results)
        if keep_matchups:
            observations_dataframe["variable"] = variable_key
            matchups.add(variable_key, observations_dataframe[CLASS4_MATCHUP_COLUMNS])
    return tables, matchups


def class4_rmsd_and_matchups(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    ocean_mask: xarray.DataArray,
    variables: list[Variable],
    keep_matchups: bool,
) -> tuple[pandas.DataFrame, Class4Matchups | None]:
    """
    The Class IV RMSD table and, when ``keep_matchups`` is set, the match-ups it is computed from.

    The table is reduced one variable at a time, so only one variable's observations are held unless the
    match-ups are kept. Kept match-ups are computed one start date at a time when the observations are sorted
    by start date, and written to a temporary directory as they are, so that the memory does not grow with the
    number of start dates; the table is then reduced from them, a lead day at a time.
    The match-ups have one row per observation of the shared Class IV population, with the
    challenger value interpolated at it. Their columns are ``variable``, ``depth_bin``, ``lead_day`` (starting
    at 0), ``first_day``, ``latitude``, ``longitude``, ``observation_value`` and ``model_value``.
    ``model_value`` is missing where the challenger has no value.
    """
    reset_class4_observations_cache()
    lead_days_count = rename_dataset_with_standard_names(challenger_dataset).sizes[Dimension.LEAD_DAY_INDEX.key()]
    population_layers = class4_population_layers(ocean_mask)
    variable_keys = [variable.key() for variable in variables]
    model_variables = _model_variables(challenger_dataset, variable_keys)
    start_slices = _start_observation_slices(reference_dataset) if keep_matchups else None
    if start_slices is not None:
        all_results, matchups = _class4_by_start(
            _start_slices_observations(reference_dataset, start_slices),
            variable_keys,
            lead_days_count,
            population_layers,
            model_variables,
        )
    else:
        all_results, matchups = _class4_by_variable(
            reference_dataset, variable_keys, lead_days_count, population_layers, model_variables, keep_matchups
        )
    if not all_results:
        return pandas.DataFrame(), matchups
    final_dataframe = pandas.concat(all_results, ignore_index=True)
    return format_class4_results(final_dataframe, lead_days_count), matchups


def class4_rmsd_and_matchups_by_start(
    challenger_dataset: xarray.Dataset,
    start_reference_datasets: Iterable[xarray.Dataset],
    ocean_mask: xarray.DataArray,
    variables: list[Variable],
    keep_matchups: bool,
) -> tuple[pandas.DataFrame, Class4Matchups | None]:
    """
    ``class4_rmsd_and_matchups`` with the observations given one start date at a time, in increasing start date
    order, each opened only when it is used, so that the observations of every start date are never held at once.
    The table and match-ups are those of ``class4_rmsd_and_matchups`` on the observations of all the start dates.
    """
    reset_class4_observations_cache()
    lead_days_count = rename_dataset_with_standard_names(challenger_dataset).sizes[Dimension.LEAD_DAY_INDEX.key()]
    variable_keys = [variable.key() for variable in variables]
    all_results, matchups = _class4_by_start(
        start_reference_datasets,
        variable_keys,
        lead_days_count,
        class4_population_layers(ocean_mask),
        _model_variables(challenger_dataset, variable_keys),
    )
    kept_matchups = matchups if keep_matchups else None
    if not all_results:
        return pandas.DataFrame(), kept_matchups
    return format_class4_results(pandas.concat(all_results, ignore_index=True), lead_days_count), kept_matchups


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    ocean_mask: xarray.DataArray,
    variables: list[Variable],
) -> pandas.DataFrame:
    table, _ = class4_rmsd_and_matchups(
        challenger_dataset, reference_dataset, ocean_mask, variables, keep_matchups=False
    )
    return table


def rmsd_class4_validation_by_start(
    challenger_dataset: xarray.Dataset,
    start_reference_datasets: Iterable[xarray.Dataset],
    ocean_mask: xarray.DataArray,
    variables: list[Variable],
) -> pandas.DataFrame:
    table, _ = class4_rmsd_and_matchups_by_start(
        challenger_dataset, start_reference_datasets, ocean_mask, variables, keep_matchups=False
    )
    return table
