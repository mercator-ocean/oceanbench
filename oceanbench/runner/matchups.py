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
from dataclasses import dataclass
import concurrent.futures
import logging
import math
import multiprocessing
import os
import shutil
import tempfile

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
from oceanbench.core.multistore import (
    MultiStoreConcatRecipe,
    get_multistore_recipe,
    open_multistore_dataset,
)
from oceanbench.runner.records import RunContext

# The per-observation grid interpolation dominates the match-up cost and is effectively
# single-threaded (dask threaded scheduler, BLAS pinned to one thread), so a whole-year run is
# parallelised across forecast start dates: each start loads one forecast field and interpolates
# at its own observations, fully independent of the others.
#
# Workers are SPAWNED, never forked. Forking after the parent has already touched the remote-backed
# datasets copies a multi-threaded process (an aiohttp event-loop thread, SSL sockets, dask worker
# threads) whose locks and sockets are meaningless in the child, which deadlocks or corrupts reads.
# A spawned worker instead starts from a clean interpreter and opens its OWN copy of the challenger
# and observation datasets from a picklable spec — preferably the ORIGINAL store URL plus the exact
# coordinate labels of the applied subsetting, so at native resolution nothing is copied and each
# worker lazily reads only its own starts' slices; a small in-memory dataset falls back to a
# size-guarded temporary zarr copy. The opened datasets are cached in a module-level global so every
# task that worker runs reuses them; only the per-start observation-index array travels as a task
# argument.
#
# BLAS/OpenMP thread-pool pinning is made effective by exporting the single-thread environment
# variables in the PARENT before the workers are spawned: a spawned child reads them at interpreter
# start-up, before numpy/BLAS initialise their pools, so the pools are actually sized to one thread
# (setting them inside the child after numpy is imported would be a no-op). The pool initializer sets
# them again as a defensive measure for libraries that read the variables lazily at first use.
_DEFAULT_MATCHUP_WORKER_CAP = 32
_THREAD_POOL_ENVIRONMENT_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
_LOGGER = logging.getLogger(__name__)

# Set in the main process on the serial path and in each spawned worker by the pool initializer; the
# per-start worker function reads it either way, so serial and parallel share one code path.
_matchup_worker_state: dict | None = None

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


# A dataset that is not reconstructable from its original store is never copied to a temporary
# zarr above this size: at native resolution the copy would stream terabytes single-threaded
# through the parent (or exhaust disk) before any worker starts, erasing the very speedup the
# parallelism exists for. Such a dataset degrades to the serial path instead.
_MATERIALISE_MAXIMUM_BYTES = 50 * 2**30


class _DatasetSpecUnavailableError(Exception):
    """The dataset can neither be reconstructed from a store nor safely materialised."""


@dataclass(frozen=True)
class _StoreBackedDatasetSpec:
    """Picklable recipe to re-open a store-backed dataset: the ORIGINAL store plus its subsetting.

    ``source`` is the store URL/path xarray recorded at open time
    (``dataset.encoding["source"]``, preserved across ``isel``/``sel``). The subsetting applied
    since the open (region boxes, start limits) is recovered as the exact per-dimension
    coordinate labels of the subset dataset; a worker re-opens the original store lazily and
    ``.sel``s those labels, so it reads only the slices its own starts touch and no bulk copy
    ever leaves the parent.
    """

    source: str
    variable_names: tuple[str, ...]
    dimension_selectors: dict[str, numpy.ndarray]
    attributes: dict


@dataclass(frozen=True)
class _MultiStoreDatasetSpec:
    """Picklable recipe to re-open a MULTI-store concat dataset (challenger weeks / observation days).

    ``recipe`` rebuilds the full lazy concat over the ORIGINAL member stores; ``selection`` label-
    selects the subset the parent dataset had applied (region boxes, start limits, the observation
    forecast-window match) against the reconstructed concat; ``assigned_coordinates`` re-attaches the
    coordinates the parent injected that are not reconstructable from the stores (the observation
    ``first_day_datetime``). Every reconstructable coordinate is validated array-exactly in the
    parent before the spec is trusted, so a divergent reconstruction degrades to serial.
    """

    recipe: MultiStoreConcatRecipe
    variable_names: tuple[str, ...]
    selection: dict[str, numpy.ndarray]
    assigned_coordinates: dict[str, tuple[tuple[str, ...], numpy.ndarray]]
    attributes: dict


@dataclass(frozen=True)
class _MaterialisedDatasetSpec:
    """Picklable fallback for genuinely in-memory datasets: the path of a temporary zarr copy."""

    zarr_path: str


_DatasetSpec = _StoreBackedDatasetSpec | _MultiStoreDatasetSpec | _MaterialisedDatasetSpec


def _open_source_store(source: str) -> xarray.Dataset:
    try:
        return xarray.open_zarr(source)
    except Exception:  # noqa: BLE001 - the source may be a non-zarr store xarray can still open
        return xarray.open_dataset(source)


def _apply_store_backed_spec(opened: xarray.Dataset, spec: _StoreBackedDatasetSpec) -> xarray.Dataset:
    reconstructed = opened[list(spec.variable_names)].sel(dict(spec.dimension_selectors))
    reconstructed.attrs = dict(spec.attributes)
    return reconstructed


def _store_backed_spec(dataset: xarray.Dataset) -> _StoreBackedDatasetSpec | None:
    """Build (and validate) a reconstructable-store spec, or ``None`` when not reconstructable.

    Reconstruction is only trusted when re-opening the original store and label-selecting it in
    the parent reproduces the dataset's dimension sizes and coordinate values exactly. That
    covers every pure-indexing subset (``subset_dataset_to_region`` boxes, ``isel`` start
    limits) while any transformed dataset either lost its ``source`` encoding or fails the
    coordinate check. The validation touches coordinates only (lazy open), never data payload.
    """
    source = dataset.encoding.get("source")
    if not source or not isinstance(source, str):
        return None
    if any(dimension not in dataset.coords for dimension in dataset.dims):
        return None
    if any(coordinate.ndim > 1 for coordinate in dataset.coords.values()):
        return None
    spec = _StoreBackedDatasetSpec(
        source=source,
        variable_names=tuple(dataset.data_vars),
        dimension_selectors={dimension: numpy.asarray(dataset[dimension].values) for dimension in dataset.dims},
        attributes=dict(dataset.attrs),
    )
    try:
        candidate = _apply_store_backed_spec(_open_source_store(source), spec)
        if dict(candidate.sizes) != dict(dataset.sizes):
            return None
        for name, coordinate in dataset.coords.items():
            if name not in candidate.coords or not numpy.array_equal(
                numpy.asarray(candidate[name].values), numpy.asarray(coordinate.values)
            ):
                return None
    except Exception:  # noqa: BLE001 - any reconstruction failure means the spec is unusable
        return None
    return spec


def _apply_multistore_spec(reconstructed_base: xarray.Dataset, spec: _MultiStoreDatasetSpec) -> xarray.Dataset:
    selected = reconstructed_base[list(spec.variable_names)].sel(dict(spec.selection))
    if spec.assigned_coordinates:
        selected = selected.assign_coords(
            {
                name: (dimensions, values)
                for name, (dimensions, values) in spec.assigned_coordinates.items()
            }
        )
    selected.attrs = dict(spec.attributes)
    return selected


def _multistore_spec(dataset: xarray.Dataset) -> tuple[_MultiStoreDatasetSpec | None, str]:
    """Build (and validate) a multi-store reconstruction spec, or ``(None, reason)`` when unavailable."""
    recipe = get_multistore_recipe(dataset)
    if recipe is None:
        return None, "no multi-store recipe attached to the dataset"
    try:
        base = open_multistore_dataset(recipe)
    except Exception as error:  # noqa: BLE001 - any reconstruction failure means the spec is unusable
        return None, f"reconstruction from member stores failed ({error})"

    selection: dict[str, numpy.ndarray] = {}
    for dimension in dataset.dims:
        if dimension not in dataset.coords:
            return None, f"concat dimension {dimension!r} has no coordinate to select on"
        if dimension not in base.coords:
            return None, f"dimension {dimension!r} is absent from the reconstruction"
        selection[dimension] = numpy.asarray(dataset[dimension].values)

    try:
        candidate = _apply_multistore_spec(
            base,
            _MultiStoreDatasetSpec(
                recipe=recipe,
                variable_names=tuple(dataset.data_vars),
                selection=selection,
                assigned_coordinates={},
                attributes=dict(dataset.attrs),
            ),
        )
    except Exception as error:  # noqa: BLE001
        return None, f"selecting the reconstruction failed ({error})"

    if dict(candidate.sizes) != dict(dataset.sizes):
        return None, "reconstructed sizes differ from the dataset"

    assigned_coordinates: dict[str, tuple[tuple[str, ...], numpy.ndarray]] = {}
    for name, coordinate in dataset.coords.items():
        if name in candidate.coords:
            if not numpy.array_equal(
                numpy.asarray(candidate[name].values), numpy.asarray(coordinate.values)
            ):
                return None, f"reconstructed coordinate {name!r} differs from the dataset"
        else:
            assigned_coordinates[name] = (
                tuple(coordinate.dims),
                numpy.asarray(coordinate.values),
            )

    spec = _MultiStoreDatasetSpec(
        recipe=recipe,
        variable_names=tuple(dataset.data_vars),
        selection=selection,
        assigned_coordinates=assigned_coordinates,
        attributes=dict(dataset.attrs),
    )
    return spec, f"multi-store, {len(recipe.member_stores)} members"


def _materialise_dataset_spec(dataset: xarray.Dataset, directory: str, name: str) -> _MaterialisedDatasetSpec:
    zarr_path = os.path.join(directory, name)
    prepared = dataset.copy()
    for variable in prepared.variables.values():
        variable.encoding = {}
    # Ragged dask chunks (remote observations concat) make ``to_zarr`` raise "Zarr requires uniform
    # chunk sizes"; rechunk to a single uniform chunk per dimension so the temporary copy can never
    # crash spec-building on that.
    if prepared.chunks:
        prepared = prepared.chunk({dimension: -1 for dimension in prepared.dims})
    prepared.to_zarr(zarr_path, mode="w", consolidated=True)
    return _MaterialisedDatasetSpec(zarr_path)


def _dataset_spec(dataset: xarray.Dataset, directory: str, name: str) -> tuple[_DatasetSpec, str]:
    multistore, multistore_reason = _multistore_spec(dataset)
    if multistore is not None:
        return multistore, multistore_reason
    store_backed = _store_backed_spec(dataset)
    if store_backed is not None:
        return store_backed, "single store-backed reconstruction"
    if int(dataset.nbytes) > _MATERIALISE_MAXIMUM_BYTES:
        raise _DatasetSpecUnavailableError(
            f"dataset {name!r} is not reconstructable from a store ({multistore_reason}) and its "
            f"~{dataset.nbytes / 2**30:.0f} GiB exceed the "
            f"{_MATERIALISE_MAXIMUM_BYTES / 2**30:.0f} GiB temporary-copy guard"
        )
    try:
        return _materialise_dataset_spec(dataset, directory, name), "materialised to a temporary zarr copy"
    except Exception as error:  # noqa: BLE001 - a materialise failure must degrade to serial, never raise out
        raise _DatasetSpecUnavailableError(
            f"dataset {name!r} could not be materialised to a temporary copy ({error})"
        ) from error


def _open_dataset_from_spec(spec: _DatasetSpec) -> xarray.Dataset:
    if isinstance(spec, _StoreBackedDatasetSpec):
        return _apply_store_backed_spec(_open_source_store(spec.source), spec)
    if isinstance(spec, _MultiStoreDatasetSpec):
        return _apply_multistore_spec(open_multistore_dataset(spec.recipe), spec)
    return xarray.open_zarr(spec.zarr_path, consolidated=True)


def _matchups_for_single_start(task: tuple[int, numpy.ndarray]) -> pandas.DataFrame:
    start_index, observation_indices = task
    # The prepared-observations cache is keyed on object id; each start's observation slice is a
    # fresh short-lived object, so reset the cache to forbid a recycled id returning another
    # start's coordinates (the cache still serves this start's variables, which share the slice).
    reset_class4_observations_cache()
    state = _matchup_worker_state
    forecast_slice = state["forecast"].isel({state["first_day_key"]: [start_index]})
    observation_slice = state["observations"].isel({state["observation_dimension"]: observation_indices})
    return class4_matchups(forecast_slice, observation_slice, state["variables"], context=state["context"])


def _worker_initializer(
    challenger_spec: _DatasetSpec,
    observation_spec: _DatasetSpec,
    variables: list[Variable],
    context: RunContext,
    first_day_key: str,
    observation_dimension: str,
) -> None:
    for variable_name in _THREAD_POOL_ENVIRONMENT_VARIABLES:
        os.environ[variable_name] = "1"
    global _matchup_worker_state
    _matchup_worker_state = {
        "forecast": _open_dataset_from_spec(challenger_spec),
        "observations": _open_dataset_from_spec(observation_spec),
        "first_day_key": first_day_key,
        "observation_dimension": observation_dimension,
        "variables": variables,
        "context": context,
    }


def _dataset_spec_logged(
    label: str, dataset: xarray.Dataset, directory: str, name: str
) -> _DatasetSpec:
    try:
        spec, reason = _dataset_spec(dataset, directory, name)
    except _DatasetSpecUnavailableError as error:
        _LOGGER.info("matchups: %s serial because %s", label, error)
        raise
    _LOGGER.info("matchups: %s parallel (%s)", label, reason)
    return spec


def _build_dataset_specs(
    challenger_dataset: xarray.Dataset,
    observation_dataset: xarray.Dataset,
    directory: str,
) -> tuple[_DatasetSpec, _DatasetSpec]:
    challenger_spec = _dataset_spec_logged("challenger", challenger_dataset, directory, "challenger.zarr")
    observation_spec = _dataset_spec_logged("observations", observation_dataset, directory, "observations.zarr")
    return challenger_spec, observation_spec


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
    computed in parallel across SPAWNED worker processes when ``max_workers > 1`` and a picklable
    dataset spec can be built, otherwise serially; either way the frames are produced in ascending
    start-date order so a consumer can stream them to a globally-sorted parquet without buffering the
    whole year. A spawned worker opens its own copy of the datasets from the spec, so no open dataset,
    socket or thread-pool lock is ever inherited across the process boundary.
    """
    global _matchup_worker_state
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

    wants_parallel = worker_count > 1 and len(tasks) > 1 and "spawn" in multiprocessing.get_all_start_methods()
    temporary_directory = None
    dataset_specs = None
    if wants_parallel:
        try:
            temporary_directory = tempfile.mkdtemp(prefix="oceanbench-matchups-")
            dataset_specs = _build_dataset_specs(challenger_dataset, observation_dataset, temporary_directory)
        except Exception as error:  # noqa: BLE001 - any failure to serialise degrades to serial
            _LOGGER.warning(
                "class-4 match-up parallelism unavailable (%s); computing serially instead", error
            )
            if temporary_directory is not None:
                shutil.rmtree(temporary_directory, ignore_errors=True)
            temporary_directory = None
            dataset_specs = None

    if dataset_specs is None:
        if not wants_parallel:
            if worker_count <= 1:
                serial_reason = "worker count is 1"
            elif len(tasks) <= 1:
                serial_reason = "only one forecast start"
            else:
                serial_reason = "the spawn start method is unavailable"
        else:
            serial_reason = "no picklable dataset spec could be built"
        _LOGGER.info("class-4 matchups: SERIAL (%s)", serial_reason)
        _matchup_worker_state = {
            "forecast": challenger_dataset,
            "observations": observation_dataset,
            "first_day_key": first_day_key,
            "observation_dimension": observation_dimension,
            "variables": variables,
            "context": context,
        }
        try:
            for task in tasks:
                yield _matchups_for_single_start(task)
        finally:
            _matchup_worker_state = None
        return

    _LOGGER.info("class-4 matchups: PARALLEL %d workers", worker_count)

    initializer_arguments = (
        dataset_specs[0],
        dataset_specs[1],
        variables,
        context,
        first_day_key,
        observation_dimension,
    )
    previous_environment = {name: os.environ.get(name) for name in _THREAD_POOL_ENVIRONMENT_VARIABLES}
    for variable_name in _THREAD_POOL_ENVIRONMENT_VARIABLES:
        os.environ[variable_name] = "1"
    try:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=worker_count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_worker_initializer,
            initargs=initializer_arguments,
        )
        try:
            yield from executor.map(_matchups_for_single_start, tasks)
        finally:
            executor.shutdown()
    finally:
        for variable_name, previous_value in previous_environment.items():
            if previous_value is None:
                os.environ.pop(variable_name, None)
            else:
                os.environ[variable_name] = previous_value
        shutil.rmtree(temporary_directory, ignore_errors=True)


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
