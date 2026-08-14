# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import pytest
import xarray

from oceanbench.core.curvilinear_class4 import (
    NativeGrid,
    interpolate_class4_native_ensemble_to_observations,
    interpolate_class4_native_model_to_observations,
    native_grid_of_dataset,
    velocity_component_names,
)
from oceanbench.core.curvilinear_staging import CurvilinearChallenger
from oceanbench.core.dataset_source import with_dataset_source
from oceanbench.core.dataset_utils import Dimension, Variable

TEMPERATURE_KEY = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
EASTWARD_KEY = Variable.EASTWARD_SEA_WATER_VELOCITY.key()
NORTHWARD_KEY = Variable.NORTHWARD_SEA_WATER_VELOCITY.key()

FIRST_DAY = numpy.datetime64("2024-01-04")
SOURCE_DIMENSIONS = ("y", "x")


def _tracer_grid() -> tuple[numpy.ndarray, numpy.ndarray]:
    """A grid of whole degrees that turns north as it runs east."""
    row = numpy.arange(4, dtype="float64")
    column = numpy.arange(4, dtype="float64")
    latitude = 40.0 + row[:, numpy.newaxis] + 0.0 * column[numpy.newaxis, :]
    longitude = 10.0 + column[numpy.newaxis, :] + 0.0 * row[:, numpy.newaxis]
    return latitude, longitude


def _native_grid(ocean_mask: numpy.ndarray | None = None) -> NativeGrid:
    latitude, longitude = _tracer_grid()
    return NativeGrid(
        latitude=latitude,
        longitude=longitude,
        ocean_mask=numpy.ones(latitude.shape, dtype=bool) if ocean_mask is None else ocean_mask,
        source_dimensions=SOURCE_DIMENSIONS,
    )


def _observations(latitudes, longitudes, depths=None) -> pandas.DataFrame:
    count = len(latitudes)
    return pandas.DataFrame(
        {
            "observation_value": numpy.zeros(count),
            Dimension.TIME.key(): [FIRST_DAY] * count,
            Dimension.LATITUDE.key(): numpy.asarray(latitudes, dtype="float64"),
            Dimension.LONGITUDE.key(): numpy.asarray(longitudes, dtype="float64"),
            "first_day": [FIRST_DAY] * count,
            Dimension.DEPTH.key(): numpy.zeros(count) if depths is None else numpy.asarray(depths, dtype="float64"),
            "lead_day": numpy.ones(count, dtype="int64"),
            "depth_bin": ["surface"] * count,
        }
    )


def _model_dataset(values: numpy.ndarray, name: str = TEMPERATURE_KEY, depths: numpy.ndarray | None = None):
    """One challenger variable over ``(first_day, lead_day[, depth], y, x)``."""
    dimensions = [Dimension.FIRST_DAY_DATETIME.key(), Dimension.LEAD_DAY_INDEX.key()]
    coordinates = {Dimension.FIRST_DAY_DATETIME.key(): [FIRST_DAY], Dimension.LEAD_DAY_INDEX.key(): [1]}
    if depths is not None:
        dimensions.append("deptht")
        coordinates["deptht"] = depths
    dimensions += list(SOURCE_DIMENSIONS)
    return xarray.Dataset({name: (dimensions, values)}, coords=coordinates)


def test_an_observation_takes_the_value_of_its_nearest_native_cell():
    values = numpy.arange(16.0).reshape(1, 1, 4, 4)
    observations = _observations([40.1, 42.9], [12.2, 10.1])

    model_values = interpolate_class4_native_model_to_observations(
        _model_dataset(values),
        TEMPERATURE_KEY,
        observations,
        _native_grid(),
    )

    numpy.testing.assert_array_equal(model_values, [2.0, 12.0])


def test_an_observation_beyond_the_neighbour_cutoff_is_dropped():
    observations = _observations([40.0, 50.0], [10.0, 10.0])

    model_values = interpolate_class4_native_model_to_observations(
        _model_dataset(numpy.ones((1, 1, 4, 4))),
        TEMPERATURE_KEY,
        observations,
        _native_grid(),
    )

    assert model_values[0] == 1.0
    assert numpy.isnan(model_values[1])


def test_an_observation_over_a_land_cell_is_dropped():
    ocean_mask = numpy.ones((4, 4), dtype=bool)
    ocean_mask[0, 2] = False
    observations = _observations([40.05, 40.05], [12.05, 11.05])

    model_values = interpolate_class4_native_model_to_observations(
        _model_dataset(numpy.arange(16.0).reshape(1, 1, 4, 4)),
        TEMPERATURE_KEY,
        observations,
        _native_grid(ocean_mask),
    )

    assert numpy.isnan(model_values[0])
    assert model_values[1] == 1.0


def test_the_member_loop_answers_for_every_member():
    values = numpy.arange(16.0).reshape(1, 1, 4, 4)
    members = numpy.stack([values, values + 100.0, values + 200.0])
    dataset = xarray.Dataset(
        {TEMPERATURE_KEY: (["member", *_model_dataset(values)[TEMPERATURE_KEY].dims], members)},
        coords={
            "member": [0, 1, 2],
            Dimension.FIRST_DAY_DATETIME.key(): [FIRST_DAY],
            Dimension.LEAD_DAY_INDEX.key(): [1],
        },
    )
    observations = _observations([40.1], [12.2])

    member_values = interpolate_class4_native_ensemble_to_observations(
        dataset,
        TEMPERATURE_KEY,
        observations,
        _native_grid(),
        ensemble_dimension="member",
    )

    assert member_values.shape == (1, 3)
    numpy.testing.assert_array_equal(member_values[0], [2.0, 102.0, 202.0])


def test_a_dataset_without_the_member_dimension_is_refused():
    with pytest.raises(ValueError, match="no member dimension"):
        interpolate_class4_native_ensemble_to_observations(
            _model_dataset(numpy.ones((1, 1, 4, 4))),
            TEMPERATURE_KEY,
            _observations([40.1], [12.2]),
            _native_grid(),
            ensemble_dimension="member",
        )


def test_the_observation_descends_to_its_own_depth():
    values = numpy.zeros((1, 1, 2, 4, 4))
    values[0, 0, 0] = 10.0
    values[0, 0, 1] = 20.0
    observations = _observations([40.1], [10.1], depths=[5.0])

    model_values = interpolate_class4_native_model_to_observations(
        _model_dataset(values, depths=numpy.array([0.0, 10.0])),
        TEMPERATURE_KEY,
        observations,
        _native_grid(),
    )

    numpy.testing.assert_allclose(model_values, [15.0])


def test_a_staggered_vertical_axis_is_read_as_the_common_one():
    values = numpy.zeros((1, 1, 2, 4, 4))
    values[0, 0, 0] = 1.0
    values[0, 0, 1] = 3.0
    dataset = _model_dataset(values, name="uo", depths=numpy.array([0.0, 10.0])).rename({"deptht": "depthu"})
    dataset = dataset.assign({"vo": xarray.zeros_like(dataset["uo"])})
    observations = _observations([40.05], [10.05], depths=[5.0])

    model_values = interpolate_class4_native_model_to_observations(dataset, EASTWARD_KEY, observations, _native_grid())

    numpy.testing.assert_allclose(model_values, [2.0], atol=1e-6)


def test_the_current_components_are_turned_onto_east_and_north():
    zonal = numpy.ones((1, 1, 4, 4))
    dataset = _model_dataset(zonal, name="uo")
    dataset = dataset.assign({"vo": xarray.zeros_like(dataset["uo"])})
    latitude, longitude = _tracer_grid()
    turning_grid = NativeGrid(
        latitude=latitude + 0.5 * numpy.arange(4.0)[numpy.newaxis, :],
        longitude=longitude,
        ocean_mask=numpy.ones(latitude.shape, dtype=bool),
        source_dimensions=SOURCE_DIMENSIONS,
    )
    observations = _observations([40.1], [10.1])

    eastward = interpolate_class4_native_model_to_observations(dataset, EASTWARD_KEY, observations, turning_grid)
    northward = interpolate_class4_native_model_to_observations(dataset, NORTHWARD_KEY, observations, turning_grid)

    assert northward[0] > 0.0
    numpy.testing.assert_allclose(eastward[0] ** 2 + northward[0] ** 2, 1.0)


def test_a_current_needs_both_components():
    dataset = _model_dataset(numpy.ones((1, 1, 4, 4)), name="uo")

    with pytest.raises(ValueError, match="one velocity component per"):
        interpolate_class4_native_model_to_observations(
            dataset, EASTWARD_KEY, _observations([40.1], [10.1]), _native_grid()
        )


def test_the_velocity_components_are_found_under_either_name():
    store_names = _model_dataset(numpy.ones((1, 1, 4, 4)), name="uo")
    store_names = store_names.assign({"vo": xarray.zeros_like(store_names["uo"])})
    standard_names = store_names.rename({"uo": "sea_water_x_velocity", "vo": "sea_water_y_velocity"})

    assert velocity_component_names(store_names) == ("uo", "vo")
    assert velocity_component_names(standard_names) == ("sea_water_x_velocity", "sea_water_y_velocity")


def test_sea_level_is_refused_on_the_native_grid():
    dataset = _model_dataset(numpy.zeros((1, 1, 4, 4)), name=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key())

    with pytest.raises(ValueError, match="mean dynamic topography"):
        interpolate_class4_native_model_to_observations(
            dataset,
            Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
            _observations([40.1], [10.1]),
            _native_grid(),
        )


def test_a_challenger_that_is_not_declared_curvilinear_has_no_native_grid():
    dataset = with_dataset_source(_model_dataset(numpy.ones((1, 1, 4, 4))), kind="challenger", name="regular")

    assert native_grid_of_dataset(dataset) is None


def test_a_declared_challenger_that_was_regridded_has_no_native_grid(monkeypatch):
    latitude, longitude = _tracer_grid()
    _declare(monkeypatch, latitude, longitude)
    regridded = xarray.Dataset(
        {TEMPERATURE_KEY: ((Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()), numpy.zeros((4, 4)))},
        coords={Dimension.LATITUDE.key(): numpy.arange(4.0), Dimension.LONGITUDE.key(): numpy.arange(4.0)},
    )

    assert native_grid_of_dataset(with_dataset_source(regridded, kind="challenger", name="curvy")) is None


def test_a_declared_challenger_still_on_its_grid_has_a_native_grid(monkeypatch):
    latitude, longitude = _tracer_grid()
    _declare(monkeypatch, latitude, longitude)
    dataset = with_dataset_source(_model_dataset(numpy.ones((1, 1, 4, 4))), kind="challenger", name="curvy")

    native_grid = native_grid_of_dataset(dataset)

    assert native_grid is not None
    numpy.testing.assert_array_equal(native_grid.latitude, latitude)
    assert native_grid.ocean_mask.all()


def _declare(monkeypatch, latitude: numpy.ndarray, longitude: numpy.ndarray) -> None:
    declaration = CurvilinearChallenger(
        tracer_grid=lambda dataset: (latitude, longitude),
        tracer_ocean_mask=lambda dataset: numpy.ones(latitude.shape, dtype=bool),
    )
    monkeypatch.setattr(
        "oceanbench.core.curvilinear_staging.CURVILINEAR_CHALLENGERS",
        {"curvy": declaration},
    )
