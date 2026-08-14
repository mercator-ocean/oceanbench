# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest

from oceanbench.core.curvilinear_c_grid import (
    GRID_TYPE_MERIDIONAL_VELOCITY,
    GRID_TYPE_TRACER,
    GRID_TYPE_ZONAL_VELOCITY,
    c_grid_ocean_mask,
    c_grid_positions,
    grid_type_of_variable,
    i_axis_angle_to_east,
    rotated_to_east_north,
)


def _rotated_grid(rotation_degrees: float, latitude_centre: float = 0.0) -> tuple[numpy.ndarray, numpy.ndarray]:
    """A small grid whose i-axis runs ``rotation_degrees`` anticlockwise from true east."""
    rotation = numpy.radians(rotation_degrees)
    row = numpy.arange(5, dtype="float64")
    column = numpy.arange(4, dtype="float64")
    step = 0.1
    latitude = latitude_centre + step * (
        row[numpy.newaxis, :] * numpy.sin(rotation) + column[:, numpy.newaxis] * numpy.cos(rotation)
    )
    eastward = step * (row[numpy.newaxis, :] * numpy.cos(rotation) - column[:, numpy.newaxis] * numpy.sin(rotation))
    longitude = eastward / numpy.cos(numpy.radians(latitude))
    return latitude, longitude


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


def _periodic_grid() -> tuple[numpy.ndarray, numpy.ndarray]:
    """A grid whose rows run all the way round in longitude and turn as they go."""
    column = numpy.arange(72, dtype="float64")
    longitude = numpy.broadcast_to(-180.0 + 5.0 * column, (3, 72)).copy()
    latitude = 30.0 + numpy.arange(3.0)[:, numpy.newaxis] + 5.0 * numpy.sin(numpy.radians(longitude))
    return latitude, longitude


def test_the_seam_column_of_a_grid_that_closes_steps_onto_the_first_column():
    latitude, longitude = _periodic_grid()

    angle = i_axis_angle_to_east(latitude, longitude)

    latitude_step = latitude[:, 0] - latitude[:, -1]
    longitude_step = (longitude[:, 0] - longitude[:, -1] + 180.0) % 360.0 - 180.0
    eastward = longitude_step * numpy.cos(numpy.radians(latitude[:, -1] + 0.5 * latitude_step))
    numpy.testing.assert_allclose(angle[:, -1], numpy.arctan2(latitude_step, eastward))


def test_the_seam_column_of_a_grid_that_closes_is_no_different_from_an_interior_one():
    latitude, longitude = _periodic_grid()

    angle = i_axis_angle_to_east(latitude, longitude)
    rolled = i_axis_angle_to_east(numpy.roll(latitude, 7, axis=1), numpy.roll(longitude, 7, axis=1))

    numpy.testing.assert_allclose(rolled, numpy.roll(angle, 7, axis=1))


def test_the_last_column_and_the_last_row_repeat_the_local_step_when_the_grid_does_not_close():
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


def test_a_grid_running_east_has_no_angle_to_east():
    latitude, longitude = _rotated_grid(0.0)

    numpy.testing.assert_allclose(i_axis_angle_to_east(latitude, longitude), 0.0, atol=1e-9)


def test_the_angle_to_east_is_the_angle_the_grid_was_turned_by():
    for rotation_degrees in (30.0, -45.0, 150.0):
        latitude, longitude = _rotated_grid(rotation_degrees)

        angle = i_axis_angle_to_east(latitude, longitude)

        numpy.testing.assert_allclose(numpy.degrees(angle), rotation_degrees, atol=1e-3)


def test_a_current_along_the_i_axis_rotates_into_east_and_north():
    latitude, longitude = _rotated_grid(30.0)
    angle = i_axis_angle_to_east(latitude, longitude)
    along_i_axis = numpy.ones(latitude.shape)
    along_j_axis = numpy.zeros(latitude.shape)

    eastward, northward = rotated_to_east_north(along_i_axis, along_j_axis, angle)

    numpy.testing.assert_allclose(eastward, numpy.cos(numpy.radians(30.0)), atol=1e-5)
    numpy.testing.assert_allclose(northward, numpy.sin(numpy.radians(30.0)), atol=1e-5)


def test_a_grid_whose_i_axis_points_west_flips_the_sign_of_an_eastward_current():
    latitude, longitude = _rotated_grid(180.0)
    angle = i_axis_angle_to_east(latitude, longitude)

    eastward, northward = rotated_to_east_north(numpy.ones(latitude.shape), numpy.zeros(latitude.shape), angle)

    numpy.testing.assert_allclose(eastward, -1.0, atol=1e-5)
    numpy.testing.assert_allclose(northward, 0.0, atol=1e-5)


def test_the_rotation_keeps_the_speed_it_was_given():
    latitude, longitude = _rotated_grid(37.0, latitude_centre=70.0)
    angle = i_axis_angle_to_east(latitude, longitude)
    generator = numpy.random.default_rng(7)
    along_i_axis = generator.normal(size=latitude.shape)
    along_j_axis = generator.normal(size=latitude.shape)

    eastward, northward = rotated_to_east_north(along_i_axis, along_j_axis, angle)

    numpy.testing.assert_allclose(
        eastward**2 + northward**2,
        along_i_axis**2 + along_j_axis**2,
        rtol=1e-12,
    )


def test_a_velocity_face_is_ocean_only_between_two_ocean_cells():
    tracer_mask = numpy.array([[True, True, False], [True, True, True], [False, True, True]])

    zonal_mask = c_grid_ocean_mask(tracer_mask, GRID_TYPE_ZONAL_VELOCITY)
    meridional_mask = c_grid_ocean_mask(tracer_mask, GRID_TYPE_MERIDIONAL_VELOCITY)

    numpy.testing.assert_array_equal(
        zonal_mask,
        numpy.array([[True, False, False], [True, True, True], [False, True, True]]),
    )
    numpy.testing.assert_array_equal(
        meridional_mask,
        numpy.array([[True, True, False], [False, True, True], [False, True, True]]),
    )
    numpy.testing.assert_array_equal(c_grid_ocean_mask(tracer_mask, GRID_TYPE_TRACER), tracer_mask)


def test_only_the_velocity_components_are_staggered():
    assert grid_type_of_variable("uo") == GRID_TYPE_ZONAL_VELOCITY
    assert grid_type_of_variable("eastward_sea_water_velocity") == GRID_TYPE_ZONAL_VELOCITY
    assert grid_type_of_variable("vo") == GRID_TYPE_MERIDIONAL_VELOCITY
    assert grid_type_of_variable("northward_sea_water_velocity") == GRID_TYPE_MERIDIONAL_VELOCITY
    for variable_name in ("thetao", "so", "zos", "ssh_ib", "sea_water_potential_temperature"):
        assert grid_type_of_variable(variable_name) == GRID_TYPE_TRACER
