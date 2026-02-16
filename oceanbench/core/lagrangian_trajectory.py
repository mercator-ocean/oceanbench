# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
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
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lead_day_utils import lead_day_labels
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
    deviations = numpy.array(
        _all_deviation_of_lagrangian_trajectories(challenger_dataset, reference_dataset, latitudes, longitudes)
    ).mean(axis=0)
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
    return list(
        map(
            partial(
                _one_deviation_of_lagrangian_trajectories,
                latitudes=latitudes,
                longitudes=longitudes,
            ),
            _split_dataset(challenger_dataset),
            _split_dataset(reference_dataset),
        )
    )


def _one_deviation_of_lagrangian_trajectories(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
):
    challenger_trajectories = _get_particle_dataset(
        dataset=challenger_dataset.isel({Dimension.DEPTH.key(): 0}),
        latitudes=latitudes,
        longitudes=longitudes,
    )

    reference_trajectories = _get_particle_dataset(
        dataset=reference_dataset.isel({Dimension.DEPTH.key(): 0}),
        latitudes=latitudes,
        longitudes=longitudes,
    )

    euclideandistance = euclidean_distance(challenger_trajectories, reference_trajectories)
    return euclideandistance


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

    def delete_error_particle(particle, _fieldset, _time):
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()

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
        delete_error_particle,
    ]  # Keep your original kernel setup

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
