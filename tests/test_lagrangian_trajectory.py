# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.lagrangian_trajectory import (
    _get_all_particles_positions,
    _get_random_ocean_points_from_file,
    euclidean_distance,
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


def _eastward_current_dataset(longitudes: numpy.ndarray, days: int) -> xarray.Dataset:
    latitudes = numpy.linspace(-10.0, 10.0, 21)
    # 1852 m per arc minute: this current moves a particle on the equator one degree a day
    eastward_velocity = 1852 * 60 / 86400
    shape = (days + 1, latitudes.size, longitudes.size)
    return xarray.Dataset(
        {
            Variable.EASTWARD_SEA_WATER_VELOCITY.key(): (
                [Dimension.TIME.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
                numpy.full(shape, eastward_velocity, dtype=numpy.float32),
            ),
            Variable.NORTHWARD_SEA_WATER_VELOCITY.key(): (
                [Dimension.TIME.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
                numpy.zeros(shape, dtype=numpy.float32),
            ),
        },
        coords={
            Dimension.TIME.key(): numpy.datetime64("2024-01-03") + numpy.arange(days + 1).astype("timedelta64[D]"),
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def test_lagrangian_particle_crosses_the_dateline_on_a_global_grid() -> None:
    dataset = _eastward_current_dataset(numpy.arange(-179.5, 180.0, 1.0), days=4)

    positions = _get_all_particles_positions(dataset, numpy.array([0.0, 0.0]), numpy.array([178.8, 0.0]))

    final_longitudes = positions["lon"].isel(time=-1).values
    assert positions.sizes["time"] == 4
    assert (final_longitudes + 180) % 360 - 180 == pytest.approx([-178.2, 3.0], abs=1e-3)


def test_lagrangian_particle_leaving_a_regional_grid_is_deleted() -> None:
    dataset = _eastward_current_dataset(numpy.arange(-20.0, 20.5, 1.0), days=4)

    positions = _get_all_particles_positions(dataset, numpy.array([0.0, 0.0]), numpy.array([18.8, 0.0]))

    final_longitudes = positions["lon"].isel(time=-1).values
    assert positions.sizes["time"] == 4
    assert numpy.isnan(final_longitudes[0])
    assert final_longitudes[1] == pytest.approx(3.0, abs=1e-3)


def test_euclidean_distance_takes_the_short_way_across_the_dateline() -> None:
    def positions(longitude: float) -> xarray.Dataset:
        return xarray.Dataset(
            {"lat": (["particle", "time"], [[0.0]]), "lon": (["particle", "time"], [[longitude]])},
            coords={"time": [numpy.datetime64("2024-01-03")]},
        )

    distance = euclidean_distance(positions(179.9), positions(-179.9))

    assert distance[0] == pytest.approx(0.2 * 111)
