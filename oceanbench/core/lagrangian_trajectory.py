# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import get_context
import os
import shutil
import time
import uuid
import numpy
import pandas
from parcels import StatusCode
from parcels import (
    FieldSet,
    ParticleSet,
    JITParticle,
    AdvectionRK4,
    Variable as ParcelsVariable,
)
import xarray


from oceanbench.core.climate_forecast_standard_names import (
    rename_dataset_with_standard_names,
)
from oceanbench.core import challenger_datasets as challenger_dataset_sources
from oceanbench.core.dataset_source import DatasetSource, get_dataset_source
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.instrumentation import instrumented_operation, log_event
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.references import glo12 as glo12_reference_datasets
from oceanbench.core.references import glorys as glorys_reference_datasets
from oceanbench.core.remote_http import require_remote_dataset_dimensions
import logging

VARIABLE = Variable
logger = logging.getLogger("parcels.tools.loggers")
logger.setLevel(level=logging.WARNING)
import warnings

warnings.filterwarnings(
    "ignore",
    message=".*'where' used without 'out'.*",
)


@dataclass
class ZoneCoordinates:
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float


class FreezeParticle(JITParticle):
    frozen = ParcelsVariable("frozen", dtype=numpy.int32, initial=0)
    lat0 = ParcelsVariable("lat0", dtype=numpy.float32, to_write=False)
    lon0 = ParcelsVariable("lon0", dtype=numpy.float32, to_write=False)
    pid = ParcelsVariable("pid", dtype=numpy.int32)


LEAD_DAY_START = 2
LAGRANGIAN_MAX_WORKERS_ENVIRONMENT_VARIABLE = "OCEANBENCH_LAGRANGIAN_MAX_WORKERS"
DEFAULT_LAGRANGIAN_MAX_WORKERS = 1
SUPPORTED_CHALLENGER_SOURCES = frozenset({"glo12", "glo36v1", "glonet", "xihe", "wenhai"})
SUPPORTED_REFERENCE_SOURCES = frozenset(
    {
        ("glorys", "quarter_degree"),
        ("glorys", "twelfth_degree"),
        ("glorys", "one_degree"),
        ("glo12", "quarter_degree"),
        ("glo12", "twelfth_degree"),
        ("glo12", "one_degree"),
    }
)


@dataclass(frozen=True)
class LagrangianWeekTask:
    week_index: int
    weeks_count: int
    first_day_datetime: str
    lead_days_count: int
    surface_depth: float | None
    latitudes: numpy.ndarray
    longitudes: numpy.ndarray
    challenger_source: DatasetSource
    reference_source: DatasetSource


def _delete_error_particle(particle, _fieldset, _time):
    if particle.state == StatusCode.ErrorOutOfBounds:
        particle.delete()


def deviation_of_lagrangian_trajectories(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    return _deviation_of_lagrangian_trajectories(
        _harmonise_dataset(challenger_dataset),
        _harmonise_dataset(reference_dataset),
    )


def _harmonise_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    return rename_dataset_with_standard_names(dataset)


def _deviation_of_lagrangian_trajectories(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
) -> pandas.DataFrame:
    lead_day_stop = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()] - 1
    latitudes, longitudes = _get_random_ocean_points_from_file(
        challenger_dataset,
        variable_name=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        n=10000,
        seed=123,
    )
    deviations = (
        pandas.concat(
            map(
                pandas.Series,
                _all_deviation_of_lagrangian_trajectories(challenger_dataset, reference_dataset, latitudes, longitudes),
            ),
            axis=1,
        )
        .mean(axis=1)
        .values
    )
    score_dataframe = pandas.DataFrame(
        {"Surface Lagrangian trajectory deviation (km)": deviations[LEAD_DAY_START - 1 : lead_day_stop]}
    )
    score_dataframe.index = lead_day_labels(LEAD_DAY_START, lead_day_stop)
    return score_dataframe.T


def _rebuild_time(
    first_day_datetime: numpy.datetime64,
    dataset: xarray.Dataset,
) -> xarray.Dataset:
    first_day = datetime.fromisoformat(str(first_day_datetime))
    return (
        dataset.sel({Dimension.FIRST_DAY_DATETIME.key(): first_day_datetime})
        .rename({Dimension.LEAD_DAY_INDEX.key(): Dimension.TIME.key()})
        .assign(
            {
                Dimension.TIME.key(): [
                    first_day + timedelta(days=int(i)) for i in dataset[Dimension.LEAD_DAY_INDEX.key()].values
                ]
            }
        )
    )


def _split_dataset(dataset: xarray.Dataset) -> list[xarray.Dataset]:
    return list(
        map(
            partial(_rebuild_time, dataset=dataset),
            dataset[Dimension.FIRST_DAY_DATETIME.key()].values,
        )
    )


def _all_deviation_of_lagrangian_trajectories(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
):
    weeks_count = challenger_dataset.sizes[Dimension.FIRST_DAY_DATETIME.key()]
    requested_max_workers = _lagrangian_max_workers(weeks_count)
    week_tasks = None
    max_workers = requested_max_workers
    if requested_max_workers > 1:
        week_tasks = _build_parallel_lagrangian_week_tasks(
            challenger_dataset,
            reference_dataset,
            latitudes,
            longitudes,
        )
        if week_tasks is None:
            log_event(
                "lagrangian_parallel_unavailable",
                requested_max_workers=requested_max_workers,
                reason="dataset_source_unknown_or_unsupported",
            )
            max_workers = 1
    deviations = []
    log_event("lagrangian_parallel_configuration", weeks_count=weeks_count, max_workers=max_workers)
    log_event("lagrangian_weeks_started", weeks_count=weeks_count, max_workers=max_workers)
    if max_workers == 1:
        challenger_datasets = _split_dataset(challenger_dataset)
        reference_datasets = _split_dataset(reference_dataset)
        for week_index, (challenger_week_dataset, reference_week_dataset) in enumerate(
            zip(challenger_datasets, reference_datasets, strict=True), start=1
        ):
            first_day_datetime = str(challenger_week_dataset[Dimension.TIME.key()].values[0])
            with instrumented_operation(
                "lagrangian_week",
                week_index=week_index,
                weeks_count=weeks_count,
                first_day_datetime=first_day_datetime,
            ):
                deviations.append(
                    _one_deviation_of_lagrangian_trajectories(
                        challenger_week_dataset,
                        reference_week_dataset,
                        latitudes,
                        longitudes,
                    )
                )
    else:
        deviations = _parallel_deviation_of_lagrangian_trajectories(week_tasks, max_workers)
    log_event("lagrangian_weeks_completed", weeks_count=weeks_count, max_workers=max_workers)
    return deviations


def _lagrangian_max_workers(weeks_count: int) -> int:
    raw_max_workers = os.environ.get(
        LAGRANGIAN_MAX_WORKERS_ENVIRONMENT_VARIABLE,
        str(DEFAULT_LAGRANGIAN_MAX_WORKERS),
    )
    requested_max_workers = int(raw_max_workers)
    if requested_max_workers < 1:
        raise ValueError(
            f"{LAGRANGIAN_MAX_WORKERS_ENVIRONMENT_VARIABLE} must be a positive integer, "
            f"got {requested_max_workers!r}."
        )
    if requested_max_workers > 1 and not _supports_parallel_lagrangian_workers():
        log_event(
            "lagrangian_parallel_unavailable",
            requested_max_workers=requested_max_workers,
            reason="fork_start_method_unavailable",
        )
        return 1
    return min(requested_max_workers, weeks_count)


def _build_parallel_lagrangian_week_tasks(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
) -> list[LagrangianWeekTask] | None:
    challenger_source = get_dataset_source(challenger_dataset)
    reference_source = get_dataset_source(reference_dataset)
    if (
        challenger_source is None
        or reference_source is None
        or challenger_source.kind != "challenger"
        or reference_source.kind != "reference"
        or challenger_source.name not in SUPPORTED_CHALLENGER_SOURCES
        or (reference_source.name, reference_source.resolution) not in SUPPORTED_REFERENCE_SOURCES
    ):
        return None

    weeks_count = challenger_dataset.sizes[Dimension.FIRST_DAY_DATETIME.key()]
    lead_days_count = challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    surface_depth = _surface_depth(challenger_dataset)
    return [
        LagrangianWeekTask(
            week_index=week_index,
            weeks_count=weeks_count,
            first_day_datetime=str(first_day_datetime),
            lead_days_count=lead_days_count,
            surface_depth=surface_depth,
            latitudes=latitudes,
            longitudes=longitudes,
            challenger_source=challenger_source,
            reference_source=reference_source,
        )
        for week_index, first_day_datetime in enumerate(
            challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values,
            start=1,
        )
    ]


def _surface_depth(dataset: xarray.Dataset) -> float | None:
    if Dimension.DEPTH.key() not in dataset.coords:
        return None
    return float(dataset[Dimension.DEPTH.key()].values[0])


def _parallel_deviation_of_lagrangian_trajectories(
    week_tasks: list[LagrangianWeekTask],
    max_workers: int,
) -> list[numpy.ndarray]:
    deviations_by_week_index: dict[int, numpy.ndarray] = {}
    start_times: dict[int, float] = {}

    mp_context = get_context("fork")
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as executor:
        future_to_task: dict = {}
        next_task_index = 0
        for _ in range(min(max_workers, len(week_tasks))):
            next_task_index = _submit_parallel_lagrangian_week(
                executor,
                future_to_task,
                start_times,
                week_tasks,
                next_task_index,
            )

        while future_to_task:
            future = next(as_completed(tuple(future_to_task)))
            week_task = future_to_task.pop(future)
            duration_seconds = time.monotonic() - start_times.pop(week_task.week_index)
            try:
                week_index, deviation = future.result()
            except Exception as error:
                log_event(
                    "lagrangian_week_failed",
                    week_index=week_task.week_index,
                    weeks_count=week_task.weeks_count,
                    first_day_datetime=week_task.first_day_datetime,
                    duration_seconds=duration_seconds,
                    error_type=error.__class__.__name__,
                    error_module=error.__class__.__module__,
                    error_message=str(error),
                )
                raise

            deviations_by_week_index[week_index] = deviation
            log_event(
                "lagrangian_week_completed",
                week_index=week_index,
                weeks_count=week_task.weeks_count,
                first_day_datetime=week_task.first_day_datetime,
                duration_seconds=duration_seconds,
            )
            if next_task_index < len(week_tasks):
                next_task_index = _submit_parallel_lagrangian_week(
                    executor,
                    future_to_task,
                    start_times,
                    week_tasks,
                    next_task_index,
                )

    return [deviations_by_week_index[week_index] for week_index in range(1, len(week_tasks) + 1)]


def _submit_parallel_lagrangian_week(
    executor: ProcessPoolExecutor,
    future_to_task: dict,
    start_times: dict[int, float],
    week_tasks: list[LagrangianWeekTask],
    task_index: int,
) -> int:
    week_task = week_tasks[task_index]
    start_times[week_task.week_index] = time.monotonic()
    log_event(
        "lagrangian_week_submitted",
        week_index=week_task.week_index,
        weeks_count=week_task.weeks_count,
        first_day_datetime=week_task.first_day_datetime,
    )
    log_event(
        "lagrangian_week_started",
        week_index=week_task.week_index,
        weeks_count=week_task.weeks_count,
        first_day_datetime=week_task.first_day_datetime,
    )
    future_to_task[executor.submit(_compute_parallel_lagrangian_week, week_task)] = week_task
    return task_index + 1


def _supports_parallel_lagrangian_workers() -> bool:
    try:
        get_context("fork")
    except ValueError:
        return False
    return True


def _compute_parallel_lagrangian_week(week_task: LagrangianWeekTask) -> tuple[int, numpy.ndarray]:
    challenger_week_dataset = _open_parallel_challenger_week(week_task)
    reference_week_dataset = _open_parallel_reference_week(week_task)
    try:
        deviation = _one_deviation_of_lagrangian_trajectories(
            challenger_week_dataset,
            reference_week_dataset,
            week_task.latitudes,
            week_task.longitudes,
        )
    finally:
        challenger_week_dataset.close()
        reference_week_dataset.close()
    return week_task.week_index, deviation


def _open_parallel_challenger_week(week_task: LagrangianWeekTask) -> xarray.Dataset:
    first_day_datetime = datetime.fromisoformat(week_task.first_day_datetime)
    if week_task.challenger_source.name == "glo36v1":
        dataset_path = (
            f"https://minio.dive.edito.eu/project-moi-glo36-oceanbench/public/"
            f"{first_day_datetime.strftime('%Y%m%d')}.zarr"
        )
        dataset = xarray.open_dataset(dataset_path, engine="zarr").rename({"lat": "latitude", "lon": "longitude"})
        return _prepare_week_dataset(
            dataset,
            first_day_datetime,
            week_task.lead_days_count,
            "glo36v1 lagrangian week open",
        )

    zarr_path_callback_by_name = {
        "glo12": challenger_dataset_sources._glo12_dataset_path,
        "glonet": challenger_dataset_sources._glonet_dataset_path,
        "xihe": challenger_dataset_sources._xihe_dataset_path,
        "wenhai": challenger_dataset_sources._wenhai_dataset_path,
    }
    dataset_path = zarr_path_callback_by_name[week_task.challenger_source.name](first_day_datetime)
    dataset = xarray.open_dataset(dataset_path, engine="zarr")
    return _prepare_week_dataset(
        dataset,
        first_day_datetime,
        week_task.lead_days_count,
        "challenger lagrangian week open",
    )


def _open_parallel_reference_week(week_task: LagrangianWeekTask) -> xarray.Dataset:
    first_day_datetime = datetime.fromisoformat(week_task.first_day_datetime)
    if week_task.reference_source.name == "glorys":
        return _open_parallel_glorys_reference_week(first_day_datetime, week_task)
    return _open_parallel_glo12_reference_week(first_day_datetime, week_task)


def _open_parallel_glorys_reference_week(
    first_day_datetime: datetime,
    week_task: LagrangianWeekTask,
) -> xarray.Dataset:
    if week_task.reference_source.resolution == "quarter_degree":
        dataset = xarray.open_dataset(
            glorys_reference_datasets._glorys_1_4_path(first_day_datetime),
            engine="zarr",
        )
    elif week_task.reference_source.resolution == "one_degree":
        dataset = xarray.open_dataset(
            glorys_reference_datasets._glorys_1_degree_path(first_day_datetime),
            engine="zarr",
        )
    else:
        surface_depth = 0.0 if week_task.surface_depth is None else week_task.surface_depth
        dataset = glorys_reference_datasets._glorys_1_12_path(
            first_day_datetime,
            days_count=week_task.lead_days_count,
            target_depths=numpy.array([surface_depth]),
        )
    return _prepare_week_dataset(
        dataset,
        first_day_datetime,
        week_task.lead_days_count,
        "glorys lagrangian week open",
    )


def _open_parallel_glo12_reference_week(
    first_day_datetime: datetime,
    week_task: LagrangianWeekTask,
) -> xarray.Dataset:
    if week_task.reference_source.resolution == "quarter_degree":
        dataset = xarray.open_dataset(
            glo12_reference_datasets._glo12_1_4_path(first_day_datetime),
            engine="zarr",
        )
    elif week_task.reference_source.resolution == "one_degree":
        dataset = xarray.open_dataset(
            glo12_reference_datasets._glo12_1_degree_path(first_day_datetime),
            engine="zarr",
        )
    else:
        surface_depth = 0.0 if week_task.surface_depth is None else week_task.surface_depth
        dataset = glo12_reference_datasets._glo12_1_12_path(
            first_day_datetime,
            days_count=week_task.lead_days_count,
            target_depths=numpy.array([surface_depth]),
        )
    return _prepare_week_dataset(
        dataset,
        first_day_datetime,
        week_task.lead_days_count,
        "glo12 lagrangian week open",
    )


def _prepare_week_dataset(
    dataset: xarray.Dataset,
    first_day_datetime: datetime,
    lead_days_count: int,
    operation_name: str,
) -> xarray.Dataset:
    week_dataset = require_remote_dataset_dimensions(dataset, [Dimension.TIME.key()], operation_name)
    week_dataset = week_dataset.isel({Dimension.TIME.key(): slice(0, lead_days_count)})
    week_lead_days_count = week_dataset.sizes[Dimension.TIME.key()]
    return _assign_week_time(
        first_day_datetime,
        week_dataset.rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()}).assign_coords(
            {Dimension.LEAD_DAY_INDEX.key(): range(week_lead_days_count)}
        ),
    )


def _assign_week_time(
    first_day_datetime: datetime,
    dataset: xarray.Dataset,
) -> xarray.Dataset:
    lead_day_values = dataset[Dimension.LEAD_DAY_INDEX.key()].values
    return dataset.rename({Dimension.LEAD_DAY_INDEX.key(): Dimension.TIME.key()}).assign(
        {
            Dimension.TIME.key(): [
                first_day_datetime + timedelta(days=int(lead_day_index)) for lead_day_index in lead_day_values
            ]
        }
    )


def _one_deviation_of_lagrangian_trajectories(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
):
    challenger_trajectories = _get_particle_dataset(
        dataset=_surface_current_dataset(challenger_dataset),
        latitudes=latitudes,
        longitudes=longitudes,
    )

    reference_trajectories = _get_particle_dataset(
        dataset=_surface_current_dataset(reference_dataset),
        latitudes=latitudes,
        longitudes=longitudes,
    )

    euclideandistance = euclidean_distance(challenger_trajectories, reference_trajectories)
    return euclideandistance


def _surface_current_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    current_dataset = _harmonise_dataset(dataset)[
        [
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
        ]
    ]
    if Dimension.DEPTH.key() in current_dataset.dims:
        return current_dataset.isel({Dimension.DEPTH.key(): 0}, drop=True)
    return current_dataset


def _set_domain_bounds(field_set: FieldSet, dataset: xarray.Dataset):
    field_set.add_constant("lon_min", float(dataset.longitude.values.min()))
    field_set.add_constant("lon_max", float(dataset.longitude.values.max()))
    field_set.add_constant("lat_min", float(dataset.latitude.values.min()))
    field_set.add_constant("lat_max", float(dataset.latitude.values.max()))
    return field_set


def _run_simulation(particle_set: ParticleSet, kernels, runtime_days: int):
    unique_id = uuid.uuid4()
    output_path = f"/tmp/tmp_particles_{unique_id}.zarr"
    output_file = particle_set.ParticleFile(name=output_path, outputdt=timedelta(hours=24))
    particle_set.execute(
        kernels,
        runtime=timedelta(days=runtime_days),
        dt=timedelta(minutes=60),
        output_file=output_file,
        verbose_progress=False,
    )
    return output_path


def _reorder_particles_by_pid(
    particle_latitudes: numpy.ndarray,
    particle_longitudes: numpy.ndarray,
    particle_ids: numpy.ndarray,
):
    sort_idx = numpy.argsort(particle_ids[:, 0])
    particle_latitudes = particle_latitudes[sort_idx, :]
    particle_longitudes = particle_longitudes[sort_idx, :]
    return particle_latitudes, particle_longitudes


def _read_output_file(file_path: str):
    dataset = xarray.open_zarr(file_path)
    particle_latitudes = dataset.lat.values  # shape: (time, n_particles)
    particle_longitudes = dataset.lon.values
    particle_ids = dataset.pid.values  # shape: (time, n_particles)
    dataset.close()
    shutil.rmtree(file_path)
    return particle_latitudes, particle_longitudes, particle_ids


def _get_all_particles_positions(
    dataset: xarray.Dataset,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
) -> xarray.Dataset:
    assert latitudes.shape == longitudes.shape, "latitudes and longitudes must be the same shape"
    variables = {
        "U": VARIABLE.EASTWARD_SEA_WATER_VELOCITY.key(),
        "V": VARIABLE.NORTHWARD_SEA_WATER_VELOCITY.key(),
    }
    dimensions = {"lat": "latitude", "lon": "longitude", "time": "time"}
    field_set = FieldSet.from_xarray_dataset(dataset, variables, dimensions)
    field_set = _set_domain_bounds(field_set, dataset)

    particle_set = ParticleSet.from_list(
        fieldset=field_set,
        pclass=FreezeParticle,
        lon=longitudes,
        lat=latitudes,
        time=dataset.time[0],
        pid=numpy.arange(len(latitudes)),
    )

    kernels = [
        AdvectionRK4,
        _delete_error_particle,
    ]

    runtime_days = len(dataset.time) - 1
    output_path = _run_simulation(particle_set, kernels, runtime_days)

    particle_latitudes, particle_longitudes, particle_ids = _read_output_file(output_path)

    particle_latitudes, particle_longitudes = _reorder_particles_by_pid(
        particle_latitudes, particle_longitudes, particle_ids
    )

    n_particles = latitudes.shape[0]
    n_times = particle_latitudes.shape[1]

    return xarray.Dataset(
        {
            "lat": (["particle", "time"], particle_latitudes),
            "lon": (["particle", "time"], particle_longitudes),
        },
        coords={
            "time": dataset.time[:n_times],
            "particle": numpy.arange(n_particles),
            "lat0": ("particle", latitudes),
            "lon0": ("particle", longitudes),
        },
    )


def _get_particle_dataset(
    dataset: xarray.Dataset,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
) -> xarray.Dataset:
    particle_initial_latitudes = latitudes
    particle_initial_longitudes = longitudes

    return _get_all_particles_positions(
        dataset,
        particle_initial_latitudes,
        particle_initial_longitudes,
    )


def _get_random_ocean_points_from_file(
    dataset: xarray.Dataset, variable_name: str, n: int, seed: int
) -> tuple[numpy.ndarray, numpy.ndarray]:

    variable_values = dataset[variable_name].isel(lead_day_index=0)
    mask = ~numpy.isnan(variable_values)[0].squeeze()

    latitude_grid, longitude_grid = numpy.meshgrid(
        dataset[Dimension.LATITUDE.key()],
        dataset[Dimension.LONGITUDE.key()],
        indexing="ij",
    )
    latitude_values = latitude_grid[mask.values]
    longitude_values = longitude_grid[mask.values]

    if len(latitude_values) < n:
        raise ValueError(f"Requested {n} points, but only {len(latitude_values)} ocean points available.")

    numpy.random.seed(seed)
    idx = numpy.random.choice(len(latitude_values), n, replace=False)

    return latitude_values[idx], longitude_values[idx]


def euclidean_distance(model_set: xarray.Dataset, reference_set: xarray.Dataset) -> numpy.ndarray:

    model_set["time"] = model_set["time"].dt.floor("D")
    reference_set["time"] = reference_set["time"].dt.floor("D")
    latitude_reference_set_rad = numpy.deg2rad(reference_set["lat"])

    dlatitude = (model_set["lat"] - reference_set["lat"]) * 111  # meters
    dlongitude = (model_set["lon"] - reference_set["lon"]) * 111 * numpy.cos(latitude_reference_set_rad)

    distance = numpy.sqrt(dlatitude**2 + dlongitude**2)  # shape: (particle, time)
    distance = distance.mean(axis=0)  # shape: (time,)
    return distance.values
