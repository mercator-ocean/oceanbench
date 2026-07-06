# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lagrangian_trajectory import lagrangian_particle_count_for_region


def _challenger_dataset(latitude_count: int, longitude_count: int, ocean_point_count: int) -> xarray.Dataset:
    values = numpy.full((1, 1, latitude_count, longitude_count), numpy.nan)
    values.reshape(-1)[:ocean_point_count] = 1.0
    return xarray.Dataset(
        {
            "zos": (
                ["first_day_datetime", "lead_day_index", "lat", "lon"],
                values,
                {"standard_name": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()},
            )
        },
        coords={
            "first_day_datetime": [numpy.datetime64("2024-01-03")],
            "lead_day_index": [0],
            "lat": xarray.DataArray(
                numpy.linspace(-10, 10, latitude_count),
                dims=["lat"],
                attrs={"standard_name": Dimension.LATITUDE.key()},
            ),
            "lon": xarray.DataArray(
                numpy.linspace(-20, 20, longitude_count),
                dims=["lon"],
                attrs={"standard_name": Dimension.LONGITUDE.key()},
            ),
        },
    )


def test_lagrangian_particle_count_preserves_global_density_with_floor() -> None:
    global_dataset = _challenger_dataset(latitude_count=200, longitude_count=200, ocean_point_count=40000)
    regional_dataset = _challenger_dataset(latitude_count=45, longitude_count=50, ocean_point_count=2250)

    particle_count = lagrangian_particle_count_for_region(global_dataset, regional_dataset)

    assert particle_count == 2000


def test_lagrangian_particle_count_keeps_current_global_count() -> None:
    global_dataset = _challenger_dataset(latitude_count=200, longitude_count=200, ocean_point_count=40000)

    particle_count = lagrangian_particle_count_for_region(global_dataset, global_dataset)

    assert particle_count == 10000


def test_lagrangian_particle_count_uses_all_available_points_when_region_is_tiny() -> None:
    global_dataset = _challenger_dataset(latitude_count=200, longitude_count=200, ocean_point_count=40000)
    tiny_regional_dataset = _challenger_dataset(latitude_count=20, longitude_count=50, ocean_point_count=1000)

    particle_count = lagrangian_particle_count_for_region(global_dataset, tiny_regional_dataset)

    assert particle_count == 1000
