# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from functools import partial
from typing import Any
import numpy
import pandas
from parcels import (
    AdvectionRK4,
    FieldSet,
    JITParticle,
    ParticleSet,
    StatusCode,
)

from parcels.kernel import shutil
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


@dataclass
class ZoneCoordinates:
    minimum_latitude: float
    maximum_latitude: float
    minimum_longitude: float
    maximum_longitude: float


class Zone(Enum):
    SMALL_ATLANTIC_NEWYORK_TO_NOUADHIBOU = ZoneCoordinates(
        minimum_latitude=20.303418,
        maximum_latitude=40.580585,
        minimum_longitude=-71.542969,
        maximum_longitude=-17.753906,
    )


LEAD_DAY_START = 2
LEAD_DAY_STOP = 9


def deviation_of_lagrangian_trajectories(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    latitudes: xarray.Dataset,
    longitudes: xarray.Dataset,
) -> pandas.DataFrame:
    return _deviation_of_lagrangian_trajectories(
        _harmonise_dataset(challenger_dataset), _harmonise_dataset(reference_dataset), latitudes, longitudes
    )


def _harmonise_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    return rename_dataset_with_standard_names(dataset)


def _deviation_of_lagrangian_trajectories(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    latitudes: xarray.Dataset,
    longitudes: xarray.Dataset,
) -> pandas.DataFrame:
    deviations = numpy.array(
        _all_deviation_of_lagrangian_trajectories(challenger_dataset, reference_dataset, latitudes, longitudes)
    ).mean(axis=0)
    score_dataframe = pandas.DataFrame(
        {"Surface Lagrangian trajectory deviation (km)": deviations[LEAD_DAY_START - 1 : LEAD_DAY_STOP]}
    )
    score_dataframe.index = lead_day_labels(LEAD_DAY_START, LEAD_DAY_STOP)
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
    latitudes: xarray.Dataset,
    longitudes: xarray.Dataset,
):
    return list(
        map(
            partial(_one_deviation_of_lagrangian_trajectories, latitudes=latitudes, longitudes=longitudes),
            _split_dataset(challenger_dataset),
            _split_dataset(reference_dataset),
        )
    )


def _one_deviation_of_lagrangian_trajectories(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    latitudes: xarray.Dataset,
    longitudes: xarray.Dataset,
):
    challenger_trajectories = _get_particle_dataset(
        dataset=challenger_dataset.isel({Dimension.DEPTH.key(): 0}), latitudes=latitudes, longitudes=longitudes
    )

    reference_trajectories = _get_particle_dataset(
        dataset=reference_dataset.isel({Dimension.DEPTH.key(): 0}), latitudes=latitudes, longitudes=longitudes
    )

    euclidean_distance = Euclidean_distance(challenger_trajectories, reference_trajectories)
    return euclidean_distance


def _zone_dimensions(dataset: xarray.Dataset, zone: Zone) -> tuple[Any, Any]:
    latitudes = dataset.sel(
        {Dimension.LATITUDE.key(): slice(zone.value.minimum_latitude, zone.value.maximum_latitude)}
    )[Dimension.LATITUDE.key()].data
    longitudes = dataset.sel(
        {Dimension.LONGITUDE.key(): slice(zone.value.minimum_longitude, zone.value.maximum_longitude)}
    )[Dimension.LONGITUDE.key()].data
    return latitudes, longitudes


def _particle_initial_positions(latitudes, longitudes):
    longitude_mesh, latitude_mesh = numpy.meshgrid(longitudes, latitudes)
    particle_latitudes = latitude_mesh.flatten()

    particle_longitudes = longitude_mesh.flatten()
    return particle_latitudes, particle_longitudes


def _build_field_set(dataset) -> FieldSet:
    variable_mapping = {
        "U": Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
        "V": Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
    }
    dimension_mapping = {
        "lat": Dimension.LATITUDE.key(),
        "lon": Dimension.LONGITUDE.key(),
        "time": Dimension.TIME.key(),
    }
    return FieldSet.from_xarray_dataset(
        dataset,
        variables=variable_mapping,
        dimensions=dimension_mapping,
    )


def _get_all_particles_positions(dataset: xarray.Dataset, latitudes: numpy.ndarray, longitudes: numpy.ndarray) -> xarray.Dataset:
    from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, Variable
    from datetime import timedelta
    import numpy as numpy
    import xarray

    assert latitudes.shape == longitudes.shape, "latitudes and longitudes must be the same shape"
    variables = {"U": VARIABLE.EASTWARD_SEA_WATER_VELOCITY.key(), "V": VARIABLE.NORTHWARD_SEA_WATER_VELOCITY.key()}
    dimensions = {"lat": "latitude", "lon": "longitude", "time": "time"}

    fieldset = FieldSet.from_xarray_dataset(dataset, variables, dimensions)

    # Define a custom particle class with a `frozen` flag
    class FreezeParticle(JITParticle):
        frozen = Variable("frozen", dtype=numpy.int32, initial=0)
        lat0 = Variable("lat0", dtype=numpy.float32, to_write=False)
        lon0 = Variable("lon0", dtype=numpy.float32, to_write=False)
        pid = Variable("pid", dtype=numpy.int32)  # <-- added only this to track order

    # Set domain bounds
    fieldset.add_constant("lon_min", float(dataset.longitude.values.min()))
    fieldset.add_constant("lon_max", float(dataset.longitude.values.max()))
    fieldset.add_constant("lat_min", float(dataset.latitude.values.min()))
    fieldset.add_constant("lat_max", float(dataset.latitude.values.max()))

    # Custom kernel: freeze particles that exit the domain
    def FreezeIfOutOfBounds(particle, fieldset, time):
        if particle.frozen == 0:
            if (
                particle.lon < fieldset.lon_min
                or particle.lon > fieldset.lon_max
                or particle.lat < fieldset.lat_min
                or particle.lat > fieldset.lat_max
            ):
                particle.frozen = 1
                particle.lat0 = particle.lat  # store frozen position
                particle.lon0 = particle.lon
        else:
            particle.lat = particle.lat0
            particle.lon = particle.lon0

    def DeleteErrorParticle(particle, fieldset, time):
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()

    # Initialize particle set with `pid`
    pset = ParticleSet.from_list(
        fieldset=fieldset,
        pclass=FreezeParticle,
        lon=longitudes,
        lat=latitudes,
        time=dataset.time[0],
        pid=numpy.arange(len(latitudes)),
    )

    # Keep your original kernel setup
    kernels = [AdvectionRK4, DeleteErrorParticle]

    # Run simulation
    output_file = pset.ParticleFile(name="tmp_particles.zarr", outputdt=timedelta(hours=24))
    pset.execute(
        kernels,
        runtime=timedelta(days=9),
        dt=timedelta(minutes=60),
        output_file=output_file,
        verbose_progress=False,
    )

    # Read output
    ds = xarray.open_zarr("tmp_particles.zarr")
    plats = ds.lat.values  # shape: (time, n_particles)
    plons = ds.lon.values
    pids = ds.pid.values  # shape: (time, n_particles)

    # Reorder based on pid at time 0
    sort_idx = numpy.argsort(pids[:, 0])
    plats = plats[sort_idx, :]
    plons = plons[sort_idx, :]

    n_particles = latitudes.shape[0]
    n_times = plats.shape[1]

    return xarray.Dataset(
        {
            "lat": (["particle", "time"], plats),
            "lon": (["particle", "time"], plons),
        },
        coords={
            "time": dataset.time[:n_times],
            "particle": numpy.arange(n_particles),
            "lat0": ("particle", latitudes),
            "lon0": ("particle", longitudes),
        },
    )



def __get_all_particles_positions(
    dataset: xarray.Dataset,
    field_set: FieldSet,
    particle_initial_latitudes,
    particle_initial_longitudes,
) -> tuple[Any, Any]:
    first_day = dataset[Dimension.TIME.key()][0]
    particle_set = ParticleSet.from_list(
        fieldset=field_set,  # the fields on which the particles are advected
        pclass=JITParticle,  # the type of particles (JITParticle or ScipyParticle)
        lon=particle_initial_longitudes,  # a vector of release longitudes
        lat=particle_initial_latitudes,
        time=first_day,
    )
    particle_zarr_folder = "tmp_particles.zarr"

    output_file = particle_set.ParticleFile(name=particle_zarr_folder, outputdt=timedelta(hours=24))
    particle_set.execute(
        AdvectionRK4,
        runtime=timedelta(days=LEAD_DAY_STOP),
        dt=timedelta(minutes=60),
        output_file=output_file,
        verbose_progress=False,
    )
    particle_dataset = xarray.open_zarr(particle_zarr_folder)
    all_particle_latitudes = particle_dataset.lat.values
    all_particle_longitudes = particle_dataset.lon.values
    shutil.rmtree(particle_zarr_folder)
    return all_particle_latitudes, all_particle_longitudes


def _get_particle_dataset(
    dataset: xarray.Dataset, latitudes: xarray.Dataset, longitudes: xarray.Dataset
) -> xarray.Dataset:
    particle_initial_latitudes = latitudes
    particle_initial_longitudes = longitudes

    return _get_all_particles_positions(
        dataset,
        particle_initial_latitudes,
        particle_initial_longitudes,
    )
    


def get_random_ocean_points_from_file(dataset: xarray.Dataset, variable_name: str = "zos", n: int = 100, seed: int = 42):

    var = dataset[variable_name].isel(lead_day_index=0)
    mask = ~numpy.isnan(variable_name)[0].squeeze()
    lat = dataset.lat
    lon = dataset.lon

    lat_grid, lon_grid = numpy.meshgrid(lat, lon, indexing="ij")
    lat_vals = lat_grid[mask.values]
    lon_vals = lon_grid[mask.values]

    if len(lat_vals) < n:
        raise ValueError(f"Requested {n} points, but only {len(lat_vals)} ocean points available.")

    numpy.random.seed(seed)
    idx = numpy.random.choice(len(lat_vals), n, replace=False)

    return lat_vals[idx], lon_vals[idx]


def Euclidean_distance(
    model_set: xarray.Dataset, reference_set: xarray.Dataset, pad: int = 10
) -> numpy.ndarray:

    model_set["time"] = model_set["time"].dt.floor("D")
    reference_set["time"] = reference_set["time"].dt.floor("D")
    lat_reference_set_rad = numpy.deg2rad(reference_set["lat"])
    
    dlat = (model_set["lat"] - reference_set["lat"]) * 111  # meters
    dlon = (model_set["lon"] - reference_set["lon"]) * 111 * numpy.cos(lat_reference_set_rad)
    
    distance = numpy.sqrt(dlat**2 + dlon**2)  # shape: (particle, time)
    distance = distance.mean(axis=0)  # shape: (time,)
    return distance.values
