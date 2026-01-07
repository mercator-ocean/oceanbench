# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import partial
import numpy
import pandas
from parcels import (
    AdvectionRK4,
    FieldSet,
    JITParticle,
    ParticleSet,
    StatusCode,
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


def _get_all_particles_positions(
    dataset: xarray.Dataset, latitudes: numpy.ndarray, longitudes: numpy.ndarray
) -> xarray.Dataset:
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

    def DeleteErrorParticle(particle):
        if particle.state == StatusCode.ErrorOutOfBounds:
            particle.delete()

    # Initialize particle set with `pid`
    particle_set = ParticleSet.from_list(
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
    output_file = particle_set.ParticleFile(name="tmp_particles.zarr", outputdt=timedelta(hours=24))
    particle_set.execute(
        kernels,
        runtime=timedelta(days=9),
        dt=timedelta(minutes=60),
        output_file=output_file,
        verbose_progress=False,
    )

    # Read output
    dataset = xarray.open_zarr("tmp_particles.zarr")
    particle_lats = dataset.lat.values  # shape: (time, n_particles)
    particle_lons = dataset.lon.values
    particle_ids = dataset.pid.values  # shape: (time, n_particles)

    # Reorder based on pid at time 0
    sort_idx = numpy.argsort(particle_ids[:, 0])
    particle_lats = particle_lats[sort_idx, :]
    particle_lons = particle_lons[sort_idx, :]

    n_particles = latitudes.shape[0]
    n_times = particle_lats.shape[1]

    return xarray.Dataset(
        {
            "lat": (["particle", "time"], particle_lats),
            "lon": (["particle", "time"], particle_lons),
        },
        coords={
            "time": dataset.time[:n_times],
            "particle": numpy.arange(n_particles),
            "lat0": ("particle", latitudes),
            "lon0": ("particle", longitudes),
        },
    )


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


def get_random_ocean_points_from_file(
    dataset: xarray.Dataset, variable_name: str = "zos", n: int = 100, seed: int = 42
):

    var = dataset[variable_name].isel(lead_day_index=0)
    mask = ~numpy.isnan(var)[0].squeeze()
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


def Euclidean_distance(model_set: xarray.Dataset, reference_set: xarray.Dataset, pad: int = 10) -> numpy.ndarray:

    model_set["time"] = model_set["time"].dt.floor("D")
    reference_set["time"] = reference_set["time"].dt.floor("D")
    lat_reference_set_rad = numpy.deg2rad(reference_set["lat"])

    dlat = (model_set["lat"] - reference_set["lat"]) * 111  # meters
    dlon = (model_set["lon"] - reference_set["lon"]) * 111 * numpy.cos(lat_reference_set_rad)

    distance = numpy.sqrt(dlat**2 + dlon**2)  # shape: (particle, time)
    distance = distance.mean(axis=0)  # shape: (time,)
    return distance.values
