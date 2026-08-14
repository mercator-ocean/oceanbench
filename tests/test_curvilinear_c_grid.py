# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest

from oceanbench.core.curvilinear_c_grid import (
    GRID_TYPE_MERIDIONAL_VELOCITY,
    GRID_TYPE_TRACER,
    GRID_TYPE_ZONAL_VELOCITY,
    c_grid_positions,
    grid_type_of_variable,
)


def _tracer_grid() -> tuple[numpy.ndarray, numpy.ndarray]:
    """A small turning grid: the rows tilt north as they run east, as a fold makes them."""
    row = numpy.arange(4, dtype="float64")
    column = numpy.arange(3, dtype="float64")
    longitude = 10.0 + row[numpy.newaxis, :] * 0.25 + column[:, numpy.newaxis] * 0.05
    latitude = 40.0 + column[:, numpy.newaxis] * 0.25 + row[numpy.newaxis, :] * 0.10
    return latitude, longitude


def test_the_tracer_point_is_the_grid_it_is_given():
    latitude, longitude = _tracer_grid()

    tracer_latitude, tracer_longitude = c_grid_positions(latitude, longitude, GRID_TYPE_TRACER)

    numpy.testing.assert_array_equal(tracer_latitude, latitude)
    numpy.testing.assert_array_equal(tracer_longitude, longitude)


def test_the_zonal_velocity_point_is_the_midpoint_of_the_next_cell_along_the_row():
    latitude, longitude = _tracer_grid()

    zonal_latitude, zonal_longitude = c_grid_positions(latitude, longitude, GRID_TYPE_ZONAL_VELOCITY)

    numpy.testing.assert_allclose(zonal_latitude[:, :-1], 0.5 * (latitude[:, :-1] + latitude[:, 1:]))
    numpy.testing.assert_allclose(zonal_longitude[:, :-1], 0.5 * (longitude[:, :-1] + longitude[:, 1:]))


def test_the_meridional_velocity_point_is_the_midpoint_of_the_next_cell_along_the_column():
    latitude, longitude = _tracer_grid()

    meridional_latitude, meridional_longitude = c_grid_positions(latitude, longitude, GRID_TYPE_MERIDIONAL_VELOCITY)

    numpy.testing.assert_allclose(meridional_latitude[:-1], 0.5 * (latitude[:-1] + latitude[1:]))
    numpy.testing.assert_allclose(meridional_longitude[:-1], 0.5 * (longitude[:-1] + longitude[1:]))


def test_the_velocity_points_are_not_the_tracer_points():
    latitude, longitude = _tracer_grid()

    zonal_latitude, zonal_longitude = c_grid_positions(latitude, longitude, GRID_TYPE_ZONAL_VELOCITY)
    meridional_latitude, _meridional_longitude = c_grid_positions(latitude, longitude, GRID_TYPE_MERIDIONAL_VELOCITY)

    assert numpy.abs(zonal_longitude - longitude).min() > 0.0
    assert numpy.abs(zonal_latitude - latitude).min() > 0.0
    assert numpy.abs(meridional_latitude - latitude).min() > 0.0


def test_the_last_column_and_the_last_row_repeat_the_local_step():
    latitude, longitude = _tracer_grid()

    zonal_latitude, zonal_longitude = c_grid_positions(latitude, longitude, GRID_TYPE_ZONAL_VELOCITY)
    meridional_latitude, _meridional_longitude = c_grid_positions(latitude, longitude, GRID_TYPE_MERIDIONAL_VELOCITY)

    numpy.testing.assert_allclose(
        zonal_longitude[:, -1], longitude[:, -1] + 0.5 * (longitude[:, -1] - longitude[:, -2])
    )
    numpy.testing.assert_allclose(zonal_latitude[:, -1], latitude[:, -1] + 0.5 * (latitude[:, -1] - latitude[:, -2]))
    numpy.testing.assert_allclose(meridional_latitude[-1], latitude[-1] + 0.5 * (latitude[-1] - latitude[-2]))


def test_a_row_crossing_the_date_line_keeps_its_midpoint_between_its_cells():
    latitude = numpy.zeros((1, 3))
    longitude = numpy.array([[179.75, -180.0, -179.75]])

    _zonal_latitude, zonal_longitude = c_grid_positions(latitude, longitude, GRID_TYPE_ZONAL_VELOCITY)

    numpy.testing.assert_allclose(zonal_longitude[0, :2], [179.875, -179.875])


def test_an_unknown_grid_point_is_refused():
    latitude, longitude = _tracer_grid()

    with pytest.raises(ValueError, match="unknown C-grid point"):
        c_grid_positions(latitude, longitude, "W")


def test_a_one_dimensional_grid_is_refused():
    with pytest.raises(ValueError, match="two-dimensional"):
        c_grid_positions(numpy.arange(3.0), numpy.arange(3.0), GRID_TYPE_ZONAL_VELOCITY)


def test_only_the_velocity_components_are_staggered():
    assert grid_type_of_variable("uo") == GRID_TYPE_ZONAL_VELOCITY
    assert grid_type_of_variable("eastward_sea_water_velocity") == GRID_TYPE_ZONAL_VELOCITY
    assert grid_type_of_variable("vo") == GRID_TYPE_MERIDIONAL_VELOCITY
    assert grid_type_of_variable("northward_sea_water_velocity") == GRID_TYPE_MERIDIONAL_VELOCITY
    for variable_name in ("thetao", "so", "zos", "ssh_ib", "sea_water_potential_temperature"):
        assert grid_type_of_variable(variable_name) == GRID_TYPE_TRACER
