# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Class-4 observation match-up artifact (contracts.md §4, ``class4-matchups`` parquet).

One row per observation point: the observation value, the model value interpolated
to that point, its latitude / longitude / depth / time, the CF standard-name
variable, the 1-based ``lead_day``, the forecast ``start_date``, and — for sea level —
the SLA shift already folded into ``model_value``.

The model-at-observation values are the very ones the Class-4 metric consumes. This
writer does not touch the numerical core: it composes the same public core
functions (``create_class4_observations_dataframe``,
``prepare_class4_model_variable``, ``interpolate_class4_model_to_observations``) that
``oceanbench.core.classIV`` uses, so the RMSD recomputed from the match-up parquet by
:func:`recompute_class4_rmsd` equals the Class-4 metric output exactly (see
``tests/runner/test_matchups.py``).
"""

from collections.abc import Iterator
import concurrent.futures
import math
import multiprocessing
import os

import numpy
import pandas
import xarray

from oceanbench.core.classIV_support import (
    create_class4_observations_dataframe,
    interpolate_class4_model_to_observations,
    mean_sea_surface_height_shift,
    prepare_class4_model_variable,
    reset_class4_observations_cache,
)
from oceanbench.core.climate_forecast_standard_names import rename_dataset_with_standard_names
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.runner.records import RunContext

# The per-observation grid interpolation dominates the match-up cost and is effectively
# single-threaded (dask threaded scheduler, BLAS pinned low), so a whole-year run is
# parallelised across forecast start dates: each start loads one forecast field and
# interpolates at its own observations, fully independent of the others. Workers are forked
# so the lazy challenger/observation datasets are inherited through copy-on-write memory
# rather than pickled across the process boundary (pickling remote-backed xarray graphs is
# what makes naive process parallelism awkward here); only the small per-start observation
# index array is sent as a task argument. The datasets are published to this module-level
# handle before the fork so every worker reads the same objects.
_DEFAULT_MATCHUP_WORKER_CAP = 32
_parallel_matchup_state: dict | None = None

MATCHUP_COLUMNS = [
    "challenger",
    "challenger_version",
    "year",
    "region",
    "variable",
    "depth",
    "depth_bin",
    "lead_day",
    "start_date",
    "time",
    "latitude",
    "longitude",
    "observation_value",
    "model_value",
    "sla_shift",
]


def _observation_frame_with_model(
    challenger: xarray.Dataset,
    observations: xarray.Dataset,
    variable_key: str,
    lead_days_count: int,
    challenger_slug: str | None = None,
) -> pandas.DataFrame:
    observation_frame = create_class4_observations_dataframe(
        observations,
        variable_key,
        variable_key,
        lead_days_count,
    )
    if observation_frame.empty:
        return observation_frame
    observation_frame = observation_frame.dropna(subset=["observation_value"])
    model_variable = prepare_class4_model_variable(challenger[variable_key], variable_key, challenger_slug)
    observation_frame = observation_frame.assign(
        model_value=interpolate_class4_model_to_observations(model_variable, observation_frame)
    )
    return observation_frame


def _shaped_matchups(
    observation_frame: pandas.DataFrame,
    variable_key: str,
    context: RunContext,
    challenger_slug: str | None = None,
) -> pandas.DataFrame:
    is_sea_surface_height = variable_key == Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
    return pandas.DataFrame(
        {
            "challenger": context.challenger,
            "challenger_version": context.challenger_version,
            "year": context.year,
            "region": context.region,
            "variable": variable_key,
            "depth": observation_frame[Dimension.DEPTH.key()].to_numpy(),
            "depth_bin": observation_frame["depth_bin"].to_numpy(),
            "lead_day": observation_frame["lead_day"].to_numpy().astype("int64") + 1,
            "start_date": observation_frame["first_day"].to_numpy(),
            "time": observation_frame[Dimension.TIME.key()].to_numpy(),
            "latitude": observation_frame[Dimension.LATITUDE.key()].to_numpy(),
            "longitude": observation_frame[Dimension.LONGITUDE.key()].to_numpy(),
            "observation_value": observation_frame["observation_value"].to_numpy(),
            "model_value": observation_frame["model_value"].to_numpy(),
            "sla_shift": mean_sea_surface_height_shift(challenger_slug) if is_sea_surface_height else numpy.nan,
        },
        columns=MATCHUP_COLUMNS,
    )


def class4_matchups(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
    variables: list[Variable],
    *,
    context: RunContext,
) -> pandas.DataFrame:
    """Build the Class-4 match-up dataframe (one row per observation point) for ``variables``.

    Column ``model_value`` is the model interpolated to the observation exactly as the
    Class-4 metric computes it; for sea-surface-height rows it is already SLA-shifted and
    ``sla_shift`` records the applied constant.
    """
    challenger = rename_dataset_with_standard_names(challenger_dataset)
    challenger_slug = context.challenger
    lead_days_count = challenger.sizes[Dimension.LEAD_DAY_INDEX.key()]
    per_variable = [
        _shaped_matchups(observation_frame, variable.key(), context, challenger_slug)
        for variable in variables
        for observation_frame in [
            _observation_frame_with_model(
                challenger, observation_dataset, variable.key(), lead_days_count, challenger_slug
            )
        ]
        if not observation_frame.empty
    ]
    if not per_variable:
        return pandas.DataFrame(columns=MATCHUP_COLUMNS)
    return pandas.concat(per_variable, ignore_index=True)


def default_matchup_worker_count() -> int:
    """Resolve the match-up worker count from the environment (default ``min(32, cpu_count)``)."""
    override = os.environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_MATCHUP_WORKERS.value)
    if override:
        return max(1, int(override))
    return max(1, min(_DEFAULT_MATCHUP_WORKER_CAP, os.cpu_count() or 1))


def _limit_worker_thread_pools() -> None:
    # One match-up per process already saturates a core; keep the numeric libraries single
    # threaded so N workers use N cores rather than oversubscribing.
    for variable_name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[variable_name] = "1"


def _matchups_for_single_start(task: tuple[int, numpy.ndarray]) -> pandas.DataFrame:
    start_index, observation_indices = task
    _limit_worker_thread_pools()
    # The prepared-observations cache is keyed on object id; each start's observation slice is a
    # fresh short-lived object, so reset the cache to forbid a recycled id returning another
    # start's coordinates (the cache still serves this start's variables, which share the slice).
    reset_class4_observations_cache()
    state = _parallel_matchup_state
    forecast_slice = state["forecast"].isel({state["first_day_key"]: [start_index]})
    observation_slice = state["observations"].isel({state["observation_dimension"]: observation_indices})
    return class4_matchups(forecast_slice, observation_slice, state["variables"], context=state["context"])


def _fork_executor(worker_count: int) -> concurrent.futures.ProcessPoolExecutor | None:
    if worker_count <= 1 or "fork" not in multiprocessing.get_all_start_methods():
        return None
    return concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count, mp_context=multiprocessing.get_context("fork")
    )


def iter_class4_matchups_by_start(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
    variables: list[Variable],
    *,
    context: RunContext,
    max_workers: int | None = None,
) -> Iterator[pandas.DataFrame]:
    """Yield the Class-4 match-up dataframe of each forecast start in ascending start-date order.

    Each yielded frame is the match-up of one forecast start (all ``variables``, all lead days),
    computed by the same untouched core as :func:`class4_matchups`. Starts are independent and are
    computed in parallel across forked worker processes when ``max_workers > 1`` and forking is
    available, otherwise serially; either way the frames are produced in ascending start-date order
    so a consumer can stream them to a globally-sorted parquet without buffering the whole year.
    """
    global _parallel_matchup_state
    worker_count = default_matchup_worker_count() if max_workers is None else max(1, max_workers)
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    start_values = numpy.asarray(challenger_dataset[first_day_key].values)
    ascending_order = numpy.argsort(start_values, kind="stable")
    observation_first_day = observation_dataset[first_day_key]
    observation_dimension = observation_first_day.dims[0]
    observation_first_day_values = numpy.asarray(observation_first_day.values)
    tasks = [
        (
            int(start_index),
            numpy.flatnonzero(observation_first_day_values == start_values[start_index]).astype("int64"),
        )
        for start_index in ascending_order
    ]
    _parallel_matchup_state = {
        "forecast": challenger_dataset,
        "observations": observation_dataset,
        "first_day_key": first_day_key,
        "observation_dimension": observation_dimension,
        "variables": variables,
        "context": context,
    }
    try:
        executor = _fork_executor(worker_count if len(tasks) > 1 else 1)
        if executor is None:
            for task in tasks:
                yield _matchups_for_single_start(task)
            return
        try:
            yield from executor.map(_matchups_for_single_start, tasks)
        finally:
            executor.shutdown()
    finally:
        _parallel_matchup_state = None


def recompute_class4_rmsd(matchups: pandas.DataFrame) -> pandas.DataFrame:
    """Recompute the Class-4 RMSD table from a match-up dataframe.

    Same reduction as ``oceanbench.core.classIV_support.compute_class4_rmsd_table``:
    ``sqrt(mean(squared_difference))`` over the observations of each
    ``(variable, depth_bin, lead_day)`` cell, dropping rows with a missing model or
    observation value. ``lead_day`` is the match-up's 1-based value.
    """
    valid = matchups.dropna(subset=["model_value", "observation_value"])
    grouped = (
        valid.assign(squared_difference=(valid["model_value"] - valid["observation_value"]) ** 2)
        .groupby(["variable", "depth_bin", "lead_day"], as_index=False)
        .agg(
            rmsd=("squared_difference", lambda values: math.sqrt(values.mean())),
            count=("squared_difference", "size"),
        )
    )
    grouped["count"] = grouped["count"].astype(int)
    return grouped[["variable", "depth_bin", "lead_day", "rmsd", "count"]]


def write_class4_matchups(matchups: pandas.DataFrame, output_path: str) -> tuple[str, int]:
    """Write the match-up dataframe to parquet, returning ``(path, bytes)``."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    matchups.to_parquet(output_path, index=False)
    return output_path, os.path.getsize(output_path)
