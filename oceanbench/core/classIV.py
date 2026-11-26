# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import os
import tempfile
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

CLASS4_RMSD_COLUMNS = ["depth_bin", "lead_day", "observation_value", "model_value"]


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


def _observations_in_shared_population(
    observations: xarray.Dataset,
    variable_key: str,
    lead_days_count: int,
    population_layers,
) -> pandas.DataFrame | None:
    """The observations of the shared population, ``None`` without any."""
    observations_dataframe = _create_observations_dataframe(
        observations,
        variable_key,
        variable_key,
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


def _with_model_values(
    challenger: xarray.Dataset,
    observations_dataframe: pandas.DataFrame | None,
    variable_key: str,
) -> pandas.DataFrame | None:
    if observations_dataframe is None:
        return None
    model_variable = _convert_forecast_ssh_to_sla(
        challenger[variable_key],
        variable_key,
    )
    observations_dataframe["model_value"] = _interpolate_model_to_observations(
        model_variable,
        observations_dataframe,
    )
    return observations_dataframe


def _matched_observations(
    challenger: xarray.Dataset,
    observations: xarray.Dataset,
    variable_key: str,
    lead_days_count: int,
    population_layers,
) -> pandas.DataFrame | None:
    """The observations of the shared population with the challenger value at them, ``None`` without any."""
    return _with_model_values(
        challenger,
        _observations_in_shared_population(observations, variable_key, lead_days_count, population_layers),
        variable_key,
    )


def _class4_observations_ahead() -> int:
    """Number of observation tables prepared ahead of the interpolation (OCEANBENCH_CLASS4_OBSERVATIONS_AHEAD, default 1)."""
    return max(0, int(os.environ.get("OCEANBENCH_CLASS4_OBSERVATIONS_AHEAD", "1")))


def _observations_by_start_and_variable(
    start_reference_datasets: Iterable[xarray.Dataset],
    variable_keys: list[str],
    lead_days_count: int,
    population_layers,
) -> Iterator[tuple[str, pandas.DataFrame | None]]:
    for start_observations in start_reference_datasets:
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


def rmsd_class4_validation(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    ocean_mask: xarray.DataArray,
    variables: list[Variable],
) -> pandas.DataFrame:
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    population_layers = class4_population_layers(ocean_mask)

    all_results = []
    for variable in variables:
        observations_dataframe = _matched_observations(
            challenger, reference_dataset, variable.key(), lead_days_count, population_layers
        )
        if observations_dataframe is None:
            continue
        variable_results = _compute_rmsd_table(observations_dataframe, variable.key())
        if not variable_results.empty:
            all_results.append(variable_results)

    if not all_results:
        return pandas.DataFrame()
    final_dataframe = pandas.concat(all_results, ignore_index=True)
    return format_class4_results(final_dataframe, lead_days_count)


def rmsd_class4_validation_by_start(
    challenger_dataset: xarray.Dataset,
    start_reference_datasets: Iterable[xarray.Dataset],
    ocean_mask: xarray.DataArray,
    variables: list[Variable],
) -> pandas.DataFrame:
    """
    ``rmsd_class4_validation`` with the observations given one start date at a time, in increasing start date
    order, so that the observations and match-ups of every start date are never held at once.

    The match-ups of each start date are written to a temporary directory, one file per variable and lead day.
    The table of a variable is then reduced one lead day at a time, from the files of that lead day read back
    in start date order: a (depth bin, lead day) group gets its rows in the order of the table of all the start
    dates, so the table is the same.
    """
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    population_layers = class4_population_layers(ocean_mask)
    variable_keys = [variable.key() for variable in variables]

    with tempfile.TemporaryDirectory(prefix="oceanbench-class4-matchups-") as directory:
        part_paths: dict[str, dict[int, list[Path]]] = {variable_key: {} for variable_key in variable_keys}
        part_count = 0
        # The observation table of the next variable (or start date) is prepared while the challenger is
        # interpolated at the current one.
        prepared_observations = _observations_by_start_and_variable(
            start_reference_datasets, variable_keys, lead_days_count, population_layers
        )
        for variable_key, observations_dataframe in _read_ahead(prepared_observations, _class4_observations_ahead()):
            observations_dataframe = _with_model_values(challenger, observations_dataframe, variable_key)
            if observations_dataframe is None:
                continue
            matchups = observations_dataframe[CLASS4_RMSD_COLUMNS]
            for lead_day, part in matchups.groupby("lead_day", sort=False):
                path = Path(directory) / f"{part_count}.pickle"
                part_count += 1
                part.reset_index(drop=True).to_pickle(path)
                part_paths[variable_key].setdefault(int(lead_day), []).append(path)
            del observations_dataframe, matchups
        reset_class4_observations_cache()

        all_results = []
        for variable_key in variable_keys:
            if not part_paths[variable_key]:
                continue
            tables = [
                _compute_rmsd_table(
                    pandas.concat([pandas.read_pickle(path) for path in paths], ignore_index=True),
                    variable_key,
                )
                for _, paths in sorted(part_paths[variable_key].items())
            ]
            variable_results = (
                pandas.concat(tables, ignore_index=True)
                .sort_values(["depth_bin", "lead_day"], kind="stable")
                .reset_index(drop=True)
            )
            if not variable_results.empty:
                all_results.append(variable_results)

    if not all_results:
        return pandas.DataFrame()
    final_dataframe = pandas.concat(all_results, ignore_index=True)
    return format_class4_results(final_dataframe, lead_days_count)
