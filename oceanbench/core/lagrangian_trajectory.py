# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
import shutil
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
from oceanbench.core.challenger_stage import should_stage_challenger_locally
from oceanbench.core.dataset_source import DatasetSource, get_dataset_source
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels
from oceanbench.core.local_stage import local_stage_build_guard, local_stage_directory
from oceanbench.core.references.reference_stage import should_stage_reference_locally
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
_LAGRANGIAN_ROW_LABEL = "Lagrangian trajectory deviation (km) []{surface}"


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
        number_of_points=10000,
        seed=123,
    )
    weekly_deviations = _all_deviation_of_lagrangian_trajectories(
        challenger_dataset,
        reference_dataset,
        latitudes,
        longitudes,
    )
    deviations = _mean_weekly_lagrangian_deviations(weekly_deviations)
    score_dataframe = pandas.DataFrame({_LAGRANGIAN_ROW_LABEL: deviations[LEAD_DAY_START - 1 : lead_day_stop]})
    score_dataframe.index = lead_day_labels(LEAD_DAY_START, lead_day_stop)
    return score_dataframe.T


def _mean_weekly_lagrangian_deviations(weekly_deviations: list[numpy.ndarray]) -> numpy.ndarray:
    return pandas.concat(map(pandas.Series, weekly_deviations), axis=1).mean(axis=1).values


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
    lagrangian_stage_sources = _lagrangian_stage_sources(challenger_dataset, reference_dataset)
    return [
        _weekly_deviation_of_lagrangian_trajectories(
            challenger_week_dataset,
            reference_week_dataset,
            latitudes,
            longitudes,
            lagrangian_stage_sources,
        )
        for challenger_week_dataset, reference_week_dataset in zip(
            _split_dataset(challenger_dataset),
            _split_dataset(reference_dataset),
            strict=True,
        )
    ]


def _weekly_deviation_of_lagrangian_trajectories(
    challenger_week_dataset: xarray.Dataset,
    reference_week_dataset: xarray.Dataset,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
    lagrangian_stage_sources: tuple[DatasetSource, DatasetSource] | None,
) -> numpy.ndarray:
    if lagrangian_stage_sources is None:
        return _one_deviation_of_lagrangian_trajectories(
            challenger_week_dataset,
            reference_week_dataset,
            latitudes,
            longitudes,
        )
    return _staged_weekly_deviation_of_lagrangian_trajectories(
        challenger_week_dataset,
        reference_week_dataset,
        latitudes,
        longitudes,
        lagrangian_stage_sources,
    )


def _first_day_datetime_of_week_dataset(dataset: xarray.Dataset) -> str:
    return str(dataset[Dimension.TIME.key()].values[0])


def _staged_weekly_deviation_of_lagrangian_trajectories(
    challenger_week_dataset: xarray.Dataset,
    reference_week_dataset: xarray.Dataset,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
    lagrangian_stage_sources: tuple[DatasetSource, DatasetSource],
) -> numpy.ndarray:
    first_day_datetime = _first_day_datetime_of_week_dataset(challenger_week_dataset)
    staged_challenger_week_dataset = _open_or_stage_lagrangian_dataset(
        dataset=challenger_week_dataset,
        dataset_source=lagrangian_stage_sources[0],
        first_day_datetime=first_day_datetime,
    )
    staged_reference_week_dataset = _open_or_stage_lagrangian_dataset(
        dataset=reference_week_dataset,
        dataset_source=lagrangian_stage_sources[1],
        first_day_datetime=first_day_datetime,
    )
    try:
        return _one_deviation_of_lagrangian_trajectories(
            staged_challenger_week_dataset,
            staged_reference_week_dataset,
            latitudes,
            longitudes,
        )
    finally:
        staged_challenger_week_dataset.close()
        staged_reference_week_dataset.close()


def _lagrangian_stage_sources(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
) -> tuple[DatasetSource, DatasetSource] | None:
    if not should_stage_challenger_locally() and not should_stage_reference_locally():
        return None
    challenger_source = get_dataset_source(challenger_dataset)
    reference_source = get_dataset_source(reference_dataset)
    if challenger_source is None or reference_source is None:
        return None
    return challenger_source, reference_source


def _lagrangian_stage_directory(dataset_source: DatasetSource, lead_days_count: int) -> Path:
    resolution_suffix = "" if dataset_source.resolution is None else f"-{dataset_source.resolution}"
    return local_stage_directory() / (
        f"lagrangian-{dataset_source.kind}-{dataset_source.name}{resolution_suffix}-{lead_days_count}d"
    )


def _lagrangian_stage_path(
    dataset_source: DatasetSource,
    lead_days_count: int,
    first_day_datetime: str,
) -> Path:
    first_day = datetime.fromisoformat(first_day_datetime).strftime("%Y%m%d")
    return _lagrangian_stage_directory(dataset_source, lead_days_count) / f"{first_day}.zarr"


def _write_staged_lagrangian_dataset(dataset: xarray.Dataset, stage_path: Path) -> None:
    staged_dataset = _surface_current_dataset(dataset).load()
    for variable_name in staged_dataset.variables:
        staged_dataset[variable_name].encoding.pop("chunks", None)
    temporary_stage_path = stage_path.with_name(f"{stage_path.name}.tmp")
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(temporary_stage_path, ignore_errors=True)
    staged_dataset.to_zarr(temporary_stage_path, mode="w")
    shutil.rmtree(stage_path, ignore_errors=True)
    temporary_stage_path.rename(stage_path)


def _open_staged_lagrangian_dataset(stage_path: Path) -> xarray.Dataset:
    return xarray.open_dataset(stage_path, engine="zarr")


def _open_or_stage_lagrangian_dataset(
    dataset: xarray.Dataset,
    dataset_source: DatasetSource,
    first_day_datetime: str,
) -> xarray.Dataset:
    lead_days_count = dataset.sizes[Dimension.TIME.key()]
    stage_path = _lagrangian_stage_path(dataset_source, lead_days_count, first_day_datetime)
    with local_stage_build_guard(stage_path) as should_build_stage:
        if should_build_stage:
            _write_staged_lagrangian_dataset(dataset, stage_path)
    return _open_staged_lagrangian_dataset(stage_path)


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
    sort_indices = numpy.argsort(particle_ids[:, 0])
    particle_latitudes = particle_latitudes[sort_indices, :]
    particle_longitudes = particle_longitudes[sort_indices, :]
    return particle_latitudes, particle_longitudes


def _read_output_file(file_path: str):
    dataset = xarray.open_zarr(file_path)
    particle_latitudes = dataset.lat.values
    particle_longitudes = dataset.lon.values
    particle_ids = dataset.pid.values
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
    return _get_all_particles_positions(dataset, latitudes, longitudes)


def _get_random_ocean_points_from_file(
    dataset: xarray.Dataset,
    variable_name: str,
    number_of_points: int,
    seed: int,
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

    if len(latitude_values) < number_of_points:
        raise ValueError(
            f"Requested {number_of_points} points, but only {len(latitude_values)} ocean points available."
        )

    numpy.random.seed(seed)
    selected_indices = numpy.random.choice(len(latitude_values), number_of_points, replace=False)

    return latitude_values[selected_indices], longitude_values[selected_indices]


def euclidean_distance(model_set: xarray.Dataset, reference_set: xarray.Dataset) -> numpy.ndarray:
    model_set["time"] = model_set["time"].dt.floor("D")
    reference_set["time"] = reference_set["time"].dt.floor("D")
    latitude_reference_set_rad = numpy.deg2rad(reference_set["lat"])

    latitude_difference = (model_set["lat"] - reference_set["lat"]) * 111
    longitude_difference = (model_set["lon"] - reference_set["lon"]) * 111 * numpy.cos(latitude_reference_set_rad)

    distance = numpy.sqrt(latitude_difference**2 + longitude_difference**2)
    return distance.mean(axis=0).values
