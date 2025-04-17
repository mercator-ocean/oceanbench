from dataclasses import dataclass
from datetime import timedelta
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
)
from parcels.kernel import shutil
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable, get_dimension
from oceanbench.core.lead_day_utils import lead_day_labels

import logging

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
    challenger_datasets: list[xarray.Dataset],
    reference_datasets: list[xarray.Dataset],
    zone: Zone,
) -> pandas.DataFrame:
    scores = numpy.array(
        list(
            map(
                partial(
                    _deviation_of_lagrangian_trajectories,
                    zone=zone,
                ),
                challenger_datasets,
                reference_datasets,
            )
        )
    )
    score_dataframe = pandas.DataFrame(
        {"Surface Lagrangian trajectory deviation (km)": scores.mean(axis=0)[LEAD_DAY_START - 1 : LEAD_DAY_STOP]}
    )
    score_dataframe.index = lead_day_labels(LEAD_DAY_START, LEAD_DAY_STOP)
    return score_dataframe.T


def _deviation_of_lagrangian_trajectories(
    challenger_dataset: xarray.Dataset,
    reference_dataset: xarray.Dataset,
    zone: Zone,
):
    challenger_trajectories = _get_particle_dataset(
        dataset=challenger_dataset.isel({Dimension.DEPTH.dimension_name_from_dataset(challenger_dataset): 0}),
        zone=zone,
    )

    reference_trajectories = _get_particle_dataset(
        dataset=reference_dataset.isel({Dimension.DEPTH.dimension_name_from_dataset(reference_dataset): 0}),
        zone=zone,
    )

    euclidean_distance = numpy.sqrt(
        ((challenger_trajectories.x.data - reference_trajectories.x.data) * 111.32) ** 2
        + (
            111.32
            * numpy.cos(
                numpy.radians(challenger_trajectories.lat.data).reshape(1, challenger_trajectories.lat.shape[0], 1)
            )
            * (challenger_trajectories.y.data - reference_trajectories.y.data)
        )
        ** 2
    )
    return numpy.nanmean(euclidean_distance, axis=(1, 2))


def _zone_dimensions(dataset: xarray.Dataset, zone: Zone) -> tuple[Any, Any]:
    latitude_name = Dimension.LATITUDE.dimension_name_from_dataset(dataset)
    longitude_name = Dimension.LONGITUDE.dimension_name_from_dataset(dataset)
    latitudes = dataset.sel({latitude_name: slice(zone.value.minimum_latitude, zone.value.maximum_latitude)})[
        latitude_name
    ].data
    longitudes = dataset.sel({longitude_name: slice(zone.value.minimum_longitude, zone.value.maximum_longitude)})[
        longitude_name
    ].data
    return latitudes, longitudes


def _particle_initial_positions(latitudes, longitudes):
    longitude_mesh, latitude_mesh = numpy.meshgrid(longitudes, latitudes)
    particle_latitudes = latitude_mesh.flatten()

    particle_longitudes = longitude_mesh.flatten()
    return particle_latitudes, particle_longitudes


def _build_field_set(dataset) -> FieldSet:
    variable_mapping = {
        "U": Variable.EASTWARD_VELOCITY.variable_name_from_dataset(dataset),
        "V": Variable.NORTHWARD_VELOCITY.variable_name_from_dataset(dataset),
    }
    dimension_mapping = {
        "lat": Dimension.LATITUDE.dimension_name_from_dataset(dataset),
        "lon": Dimension.LONGITUDE.dimension_name_from_dataset(dataset),
        "time": Dimension.TIME.dimension_name_from_dataset(dataset),
    }
    return FieldSet.from_xarray_dataset(
        dataset,
        variables=variable_mapping,
        dimensions=dimension_mapping,
    )


def _get_all_particles_positions(
    dataset: xarray.Dataset,
    field_set: FieldSet,
    particle_initial_latitudes,
    particle_initial_longitudes,
) -> tuple[Any, Any]:
    first_day = get_dimension(dataset, Dimension.TIME)[0]
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


def _get_particle_dataset(dataset: xarray.Dataset, zone: Zone) -> xarray.Dataset:
    field_set = _build_field_set(dataset)
    latitudes, longitudes = _zone_dimensions(dataset, zone)
    particle_initial_latitudes, particle_initial_longitudes = _particle_initial_positions(latitudes, longitudes)

    all_particle_latitudes, all_particle_longitudes = _get_all_particles_positions(
        dataset,
        field_set,
        particle_initial_latitudes,
        particle_initial_longitudes,
    )
    x = all_particle_latitudes.reshape(latitudes.shape[0], longitudes.shape[0], LEAD_DAY_STOP).transpose(2, 0, 1)
    y = all_particle_longitudes.reshape(latitudes.shape[0], longitudes.shape[0], LEAD_DAY_STOP).transpose(2, 0, 1)

    return xarray.Dataset(
        {
            "x": (["time", "lat", "lon"], x),
            "y": (["time", "lat", "lon"], y),
        },
        coords={
            "time": get_dimension(dataset, Dimension.TIME)[0:LEAD_DAY_STOP],
            "lat": latitudes,
            "lon": longitudes,
        },
    )
