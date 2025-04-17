import numpy
import xarray

from oceanbench.core.dataset_utils import (
    Dimension,
    Variable,
    get_dimension,
    get_variable,
)


def add_geostrophic_currents(dataset: xarray.Dataset) -> xarray.Dataset:
    sea_surface_height = get_variable(dataset, Variable.HEIGHT)
    latitude = get_dimension(dataset, Dimension.LATITUDE).values
    longitude = get_dimension(dataset, Dimension.LONGITUDE).values

    latitude_radian = numpy.deg2rad(latitude)

    # coriolis
    omega = 7.2921e-5
    f = 2 * omega * numpy.sin(latitude_radian)
    R = 6371000

    # Compute grid spacing
    dx = numpy.gradient(longitude) * (numpy.pi / 180) * R * numpy.cos(latitude_radian[:, numpy.newaxis])
    dy = numpy.gradient(latitude)[:, numpy.newaxis] * (numpy.pi / 180) * R

    dssh_dx = numpy.gradient(sea_surface_height, axis=-1) / dx
    dssh_dy = numpy.gradient(sea_surface_height, axis=-2) / dy

    g = 9.81  # gravity
    eastward_geostrophic_velocity = -g / f[:, numpy.newaxis] * dssh_dy
    northward_geostrophic_velocity = g / f[:, numpy.newaxis] * dssh_dx

    dimensions = (
        Dimension.TIME.dimension_name_from_dataset(dataset),
        Dimension.LATITUDE.dimension_name_from_dataset(dataset),
        Dimension.LONGITUDE.dimension_name_from_dataset(dataset),
    )

    dataset_with_geostrophic_current = dataset.assign(
        {
            Variable.EASTWARD_GEOSTROPHIC_VELOCITY.value: (
                dimensions,
                eastward_geostrophic_velocity,
            ),
            Variable.NORTHWARD_GEOSTROPHIC_VELOCITY.value: (
                dimensions,
                northward_geostrophic_velocity,
            ),
        }
    )
    return _exclude_equator(dataset_with_geostrophic_current)


def _exclude_equator(dataset: xarray.Dataset) -> xarray.Dataset:
    latitude = get_dimension(dataset, Dimension.LATITUDE)
    not_on_equator = (latitude < -0.5) | (latitude > 0.5)
    return dataset.assign(
        {
            Variable.EASTWARD_GEOSTROPHIC_VELOCITY.value: dataset[Variable.EASTWARD_GEOSTROPHIC_VELOCITY.value].where(
                not_on_equator, drop=False
            ),
            Variable.NORTHWARD_GEOSTROPHIC_VELOCITY.value: dataset[Variable.NORTHWARD_GEOSTROPHIC_VELOCITY.value].where(
                not_on_equator, drop=False
            ),
        }
    )
