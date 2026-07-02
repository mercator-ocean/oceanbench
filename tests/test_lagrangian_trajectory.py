# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lagrangian_trajectory import (
    _get_random_ocean_points_from_file,
    lagrangian_particle_count_for_region,
)


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


def test_lagrangian_ocean_point_sampling_uses_area_probabilities_over_valid_points(monkeypatch) -> None:
    variable_key = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
    dataset = xarray.Dataset(
        {
            variable_key: (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                numpy.array([[[[1.0, numpy.nan], [1.0, 1.0]]]]),
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): [numpy.datetime64("2024-01-03")],
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.LATITUDE.key(): [0.0, 60.0],
            Dimension.LONGITUDE.key(): [10.0, 20.0],
        },
    )
    captured = {}

    def choose_indices(population_size, size, replace, p):
        captured["population_size"] = population_size
        captured["size"] = size
        captured["replace"] = replace
        captured["probabilities"] = p
        return numpy.array([0, 2])

    monkeypatch.setattr(numpy.random, "choice", choose_indices)

    latitudes, longitudes = _get_random_ocean_points_from_file(dataset, variable_key, n=2, seed=123)

    assert latitudes.tolist() == [0.0, 60.0]
    assert longitudes.tolist() == [10.0, 20.0]
    assert captured["population_size"] == 3
    assert captured["size"] == 2
    assert captured["replace"] is False
    assert numpy.allclose(captured["probabilities"], [0.5, 0.25, 0.25])
