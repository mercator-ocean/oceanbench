# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
import xarray

from oceanbench.core.curvilinear_staging import (
    STANDARD_QUARTER_DEGREE_LATITUDE,
    STANDARD_QUARTER_DEGREE_LONGITUDE,
    CurvilinearChallenger,
    curvilinear_mapping,
    maybe_regridded_curvilinear_dataset,
    ocean_mask_from_land_sentinel,
    regridded_curvilinear_dataset,
    with_common_depth_axis,
)

TARGET_LATITUDE = numpy.arange(40.0, 41.01, 0.25)
TARGET_LONGITUDE = numpy.arange(10.0, 11.01, 0.25)


def _tracer_grid(rows: int = 9, columns: int = 9) -> tuple[numpy.ndarray, numpy.ndarray]:
    """A slightly turning grid over the target box, fine enough to cover every target cell."""
    row = numpy.arange(rows, dtype="float64")
    column = numpy.arange(columns, dtype="float64")
    latitude = 39.9 + 0.15 * row[:, numpy.newaxis] + 0.01 * column[numpy.newaxis, :]
    longitude = 9.9 + 0.15 * column[numpy.newaxis, :] + 0.01 * row[:, numpy.newaxis]
    return latitude, longitude


def _tracer_dataset(values: numpy.ndarray) -> xarray.Dataset:
    latitude, longitude = _tracer_grid()
    return xarray.Dataset(
        {"thetao": (("y", "x"), values)},
        coords={"nav_lat": (("y", "x"), latitude), "nav_lon": (("y", "x"), longitude)},
    )


def _regridded(dataset: xarray.Dataset, ocean_mask: numpy.ndarray | None = None, **keywords) -> xarray.Dataset:
    latitude, longitude = _tracer_grid()
    return regridded_curvilinear_dataset(
        dataset,
        latitude,
        longitude,
        numpy.ones(latitude.shape, dtype=bool) if ocean_mask is None else ocean_mask,
        target_latitude=TARGET_LATITUDE,
        target_longitude=TARGET_LONGITUDE,
        **keywords,
    )


def test_the_regridded_challenger_has_one_dimensional_axes():
    dataset = _tracer_dataset(numpy.zeros((9, 9)))

    regridded = _regridded(dataset)

    assert regridded["thetao"].dims == ("latitude", "longitude")
    numpy.testing.assert_array_equal(regridded["latitude"].values, TARGET_LATITUDE)
    numpy.testing.assert_array_equal(regridded["longitude"].values, TARGET_LONGITUDE)
    assert "nav_lat" not in regridded.variables
    assert "nav_lon" not in regridded.variables
    assert "y" not in regridded.dims
    assert "x" not in regridded.dims


def test_every_target_cell_takes_the_value_of_its_nearest_native_cell():
    latitude, longitude = _tracer_grid()
    dataset = _tracer_dataset(numpy.arange(81.0).reshape(9, 9))

    regridded = _regridded(dataset)

    for latitude_index, target_latitude in enumerate(TARGET_LATITUDE):
        for longitude_index, target_longitude in enumerate(TARGET_LONGITUDE):
            distance = (latitude - target_latitude) ** 2 + (
                (longitude - target_longitude) * numpy.cos(numpy.radians(target_latitude))
            ) ** 2
            nearest = float(numpy.arange(81.0).reshape(9, 9).ravel()[numpy.argmin(distance)])
            assert regridded["thetao"].values[latitude_index, longitude_index] == nearest


def test_a_land_block_never_reaches_the_regridded_field():
    values = numpy.full((9, 9), 5.0)
    values[3:6, 3:6] = 17.5
    ocean_mask = ocean_mask_from_land_sentinel(values, 17.5)
    dataset = _tracer_dataset(values)

    regridded = _regridded(dataset, ocean_mask)

    assert not (regridded["thetao"].values == 17.5).any()
    assert numpy.isnan(regridded["thetao"].values).any()
    assert numpy.nanmin(regridded["thetao"].values) == 5.0


def test_a_target_cell_beyond_the_neighbour_cutoff_is_dropped():
    dataset = _tracer_dataset(numpy.ones((9, 9)))
    latitude, longitude = _tracer_grid()

    regridded = regridded_curvilinear_dataset(
        dataset,
        latitude,
        longitude,
        numpy.ones(latitude.shape, dtype=bool),
        target_latitude=numpy.array([40.5, 60.0]),
        target_longitude=numpy.array([10.5]),
    )

    assert regridded["thetao"].values[0, 0] == 1.0
    assert numpy.isnan(regridded["thetao"].values[1, 0])


def test_the_ensemble_and_depth_dimensions_cross_the_regrid_untouched():
    latitude, longitude = _tracer_grid()
    values = numpy.arange(3 * 2 * 81.0).reshape(3, 2, 9, 9)
    dataset = xarray.Dataset(
        {"thetao": (("member", "deptht", "y", "x"), values)},
        coords={
            "member": [0, 1, 2],
            "deptht": [0.5, 10.0],
            "nav_lat": (("y", "x"), latitude),
            "nav_lon": (("y", "x"), longitude),
        },
    )

    regridded = _regridded(dataset, depth_values=numpy.array([0.494, 9.573]))

    assert regridded["thetao"].dims == ("member", "depth", "latitude", "longitude")
    assert regridded.sizes["member"] == 3
    numpy.testing.assert_array_equal(regridded["depth"].values, [0.494, 9.573])
    assert len(numpy.unique(regridded["thetao"].values)) == regridded["thetao"].size


def test_the_velocity_components_are_sampled_through_their_own_faces():
    latitude, longitude = _tracer_grid()
    zonal_values = numpy.arange(81.0).reshape(9, 9)
    dataset = xarray.Dataset(
        {
            "uo": (("y", "x"), zonal_values),
            "vo": (("y", "x"), numpy.zeros((9, 9))),
        },
        coords={"nav_lat": (("y", "x"), latitude), "nav_lon": (("y", "x"), longitude)},
    )

    through_own_faces = _regridded(dataset)["uo"].values
    tracer_dataset = dataset.rename({"uo": "thetao"}).drop_vars("vo")
    through_tracer_points = _regridded(tracer_dataset)["thetao"].values

    assert not numpy.allclose(through_own_faces, through_tracer_points, equal_nan=True)


def test_the_velocity_components_come_out_east_and_north_under_standard_names():
    latitude, longitude = _tracer_grid()
    dataset = xarray.Dataset(
        {
            "uo": (("y", "x"), numpy.ones((9, 9))),
            "vo": (("y", "x"), numpy.zeros((9, 9))),
        },
        coords={"nav_lat": (("y", "x"), latitude), "nav_lon": (("y", "x"), longitude)},
    )

    regridded = _regridded(dataset)

    assert regridded["uo"].attrs["standard_name"] == "eastward_sea_water_velocity"
    assert regridded["vo"].attrs["standard_name"] == "northward_sea_water_velocity"
    # The grid turns north as it runs east, so a current along the i-axis has a northward part.
    assert (regridded["vo"].values > 0.0).all()
    numpy.testing.assert_allclose(regridded["uo"].values ** 2 + regridded["vo"].values ** 2, 1.0)


def _folding_grid() -> tuple[numpy.ndarray, numpy.ndarray]:
    """A grid whose i-axis runs east on the southern rows and west on the northern ones."""
    row = numpy.arange(9, dtype="float64")
    column = numpy.arange(9, dtype="float64")
    latitude = numpy.broadcast_to(40.0 + 0.1 * row[:, numpy.newaxis], (9, 9)).copy()
    longitude = numpy.where(
        row[:, numpy.newaxis] < 4.5,
        10.0 + 0.1 * column[numpy.newaxis, :],
        10.0 + 0.1 * (8.0 - column[numpy.newaxis, :]),
    )
    return latitude, numpy.asarray(longitude, dtype="float64")


def test_a_target_cell_whose_two_velocity_faces_disagree_loses_its_current():
    latitude, longitude = _folding_grid()
    dataset = xarray.Dataset(
        {
            "uo": (("y", "x"), numpy.ones((9, 9))),
            "vo": (("y", "x"), numpy.zeros((9, 9))),
            "thetao": (("y", "x"), numpy.full((9, 9), 12.0)),
        }
    )
    target_latitude = numpy.arange(40.0, 40.81, 0.05)
    target_longitude = numpy.arange(10.0, 10.81, 0.1)

    regridded = regridded_curvilinear_dataset(
        dataset,
        latitude,
        longitude,
        numpy.ones((9, 9), dtype=bool),
        target_latitude=target_latitude,
        target_longitude=target_longitude,
    )

    fold_row = int(numpy.argmin(numpy.abs(target_latitude - 40.45)))
    assert numpy.isnan(regridded["uo"].values[fold_row]).all()
    assert numpy.isnan(regridded["vo"].values[fold_row]).all()
    numpy.testing.assert_allclose(regridded["uo"].values[0], 1.0, atol=1e-12)
    numpy.testing.assert_allclose(regridded["vo"].values[0], 0.0, atol=1e-12)
    numpy.testing.assert_allclose(regridded["uo"].values[-1], -1.0, atol=1e-12)
    numpy.testing.assert_allclose(regridded["thetao"].values, 12.0)


def test_one_velocity_component_on_its_own_is_refused():
    latitude, longitude = _tracer_grid()
    dataset = xarray.Dataset(
        {"uo": (("y", "x"), numpy.ones((9, 9)))},
        coords={"nav_lat": (("y", "x"), latitude), "nav_lon": (("y", "x"), longitude)},
    )

    with pytest.raises(ValueError, match="without the other velocity component"):
        _regridded(dataset)


def test_a_field_of_another_shape_than_the_declared_grid_is_refused():
    dataset = xarray.Dataset({"thetao": (("y", "x"), numpy.zeros((9, 8)))})

    with pytest.raises(ValueError, match="point at other cells"):
        _regridded(dataset)


def test_the_mapping_of_one_grid_pair_is_built_once():
    latitude, longitude = _tracer_grid()
    ocean_mask = numpy.ones(latitude.shape, dtype=bool)

    first = curvilinear_mapping(latitude, longitude, ocean_mask, TARGET_LATITUDE, TARGET_LONGITUDE)
    second = curvilinear_mapping(latitude, longitude, ocean_mask, TARGET_LATITUDE, TARGET_LONGITUDE)

    assert first is second


def test_a_different_land_mask_is_a_different_mapping():
    latitude, longitude = _tracer_grid()
    ocean_mask = numpy.ones(latitude.shape, dtype=bool)
    with_land = ocean_mask.copy()
    with_land[4, 4] = False

    first = curvilinear_mapping(latitude, longitude, ocean_mask, TARGET_LATITUDE, TARGET_LONGITUDE)
    second = curvilinear_mapping(latitude, longitude, with_land, TARGET_LATITUDE, TARGET_LONGITUDE)

    assert first is not second
    assert second.usable.sum() < first.usable.sum()


def test_the_staggered_vertical_axis_takes_the_tracer_levels():
    dataset = xarray.Dataset({"uo": (("depthu",), numpy.zeros(2))}, coords={"depthu": [1.0, 2.0]})

    renamed = with_common_depth_axis(dataset, numpy.array([0.494, 1.541]))

    assert "depthu" not in renamed.dims
    numpy.testing.assert_array_equal(renamed["depth"].values, [0.494, 1.541])


def test_every_staggered_vertical_axis_lands_on_the_common_one():
    dataset = xarray.Dataset(
        {
            "uo": (("depthu",), numpy.zeros(2)),
            "vo": (("depthv",), numpy.ones(2)),
            "thetao": (("deptht",), numpy.full(2, 3.0)),
        },
        coords={"depthu": [1.0, 2.0], "depthv": [1.0, 2.0], "deptht": [1.0, 2.0]},
    )

    renamed = with_common_depth_axis(dataset)

    assert set(renamed.dims) == {"depth"}
    numpy.testing.assert_array_equal(renamed["depth"].values, [1.0, 2.0])
    for name in ("uo", "vo", "thetao"):
        assert renamed[name].dims == ("depth",)


def test_a_three_dimensional_store_on_three_staggered_axes_is_regridded():
    latitude, longitude = _tracer_grid()
    dimensions = ("depthu", "y", "x")
    dataset = xarray.Dataset(
        {
            "uo": (dimensions, numpy.ones((2, 9, 9))),
            "vo": (("depthv", "y", "x"), numpy.zeros((2, 9, 9))),
            "thetao": (("deptht", "y", "x"), numpy.full((2, 9, 9), 12.0)),
        },
        coords={
            "depthu": [0.5, 10.0],
            "depthv": [0.5, 10.0],
            "deptht": [0.5, 10.0],
            "nav_lat": (("y", "x"), latitude),
            "nav_lon": (("y", "x"), longitude),
        },
    )

    regridded = _regridded(dataset, depth_values=numpy.array([0.494, 9.573]))

    assert set(regridded.dims) == {"depth", "latitude", "longitude"}
    numpy.testing.assert_array_equal(regridded["depth"].values, [0.494, 9.573])
    assert regridded["uo"].attrs["standard_name"] == "eastward_sea_water_velocity"
    assert (regridded["vo"].values > 0.0).all()
    numpy.testing.assert_allclose(regridded["uo"].values ** 2 + regridded["vo"].values ** 2, 1.0)
    numpy.testing.assert_allclose(regridded["thetao"].values, 12.0)


def test_a_challenger_that_is_not_declared_curvilinear_is_left_alone():
    dataset = _tracer_dataset(numpy.zeros((9, 9)))

    assert maybe_regridded_curvilinear_dataset(dataset, "not-a-curvilinear-challenger") is dataset


def test_a_declared_challenger_goes_through_the_regrid(monkeypatch):
    latitude, longitude = _tracer_grid()
    declaration = CurvilinearChallenger(
        tracer_grid=lambda dataset: (latitude, longitude),
        tracer_ocean_mask=lambda dataset: numpy.ones(latitude.shape, dtype=bool),
        target_latitude=TARGET_LATITUDE,
        target_longitude=TARGET_LONGITUDE,
    )
    monkeypatch.setattr(
        "oceanbench.core.curvilinear_staging.CURVILINEAR_CHALLENGERS",
        {"a-curvilinear-challenger": declaration},
    )

    regridded = maybe_regridded_curvilinear_dataset(_tracer_dataset(numpy.zeros((9, 9))), "a-curvilinear-challenger")

    assert regridded["thetao"].dims == ("latitude", "longitude")


def test_no_challenger_is_declared_curvilinear_by_default():
    from oceanbench.core.curvilinear_staging import CURVILINEAR_CHALLENGERS

    assert CURVILINEAR_CHALLENGERS == {}


def test_the_standard_target_grid_is_the_quarter_degree_scoring_grid():
    assert STANDARD_QUARTER_DEGREE_LATITUDE.shape == (672,)
    assert STANDARD_QUARTER_DEGREE_LONGITUDE.shape == (1440,)
    assert STANDARD_QUARTER_DEGREE_LATITUDE[0] == -78.0
    assert STANDARD_QUARTER_DEGREE_LATITUDE[-1] == 89.75
    assert STANDARD_QUARTER_DEGREE_LONGITUDE[0] == -180.0
    assert STANDARD_QUARTER_DEGREE_LONGITUDE[-1] == 179.75
