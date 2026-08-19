# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
import xarray

from oceanbench.core.classIV_support import (
    CHALLENGER_INVERSE_BAROMETER_VARIABLES,
    CHALLENGER_MEAN_SEA_SURFACE_HEIGHT_SHIFTS,
)
from oceanbench.core.curvilinear_staging import (
    CLASS4_ROUTE_NATIVE,
    CLASS4_ROUTE_REGRIDDED,
    CURVILINEAR_CHALLENGERS,
    GLOENS_LAND_SENTINELS,
    GLOENS_SOURCE_NAME,
    STANDARD_QUARTER_DEGREE_LATITUDE,
    STANDARD_QUARTER_DEGREE_LONGITUDE,
    CurvilinearChallenger,
    gloens_tracer_grid,
    gloens_tracer_ocean_mask,
    curvilinear_mapping,
    curvilinear_stage_variant,
    maybe_regridded_curvilinear_dataset,
    ocean_mask_from_land_sentinel,
    regridded_curvilinear_dataset,
    with_common_depth_axis,
)
from oceanbench.core.weekly_stage import staged_weekly_dataset

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


def _declare(monkeypatch, class4_route: str) -> None:
    latitude, longitude = _tracer_grid()
    declaration = CurvilinearChallenger(
        tracer_grid=lambda dataset: (latitude, longitude),
        tracer_ocean_mask=lambda dataset: numpy.ones(latitude.shape, dtype=bool),
        target_latitude=TARGET_LATITUDE,
        target_longitude=TARGET_LONGITUDE,
        class4_route=class4_route,
    )
    monkeypatch.setattr(
        "oceanbench.core.curvilinear_staging.CURVILINEAR_CHALLENGERS",
        {"a-curvilinear-challenger": declaration},
    )


def test_the_class4_track_of_a_native_route_challenger_reads_the_native_grid(monkeypatch):
    _declare(monkeypatch, CLASS4_ROUTE_NATIVE)
    dataset = _tracer_dataset(numpy.zeros((9, 9)))

    for_class4 = maybe_regridded_curvilinear_dataset(dataset, "a-curvilinear-challenger", for_class4=True)
    for_the_grid = maybe_regridded_curvilinear_dataset(dataset, "a-curvilinear-challenger")

    assert for_class4 is dataset
    assert for_the_grid["thetao"].dims == ("latitude", "longitude")


def test_the_class4_track_of_a_regridded_route_challenger_reads_the_regular_grid(monkeypatch):
    _declare(monkeypatch, CLASS4_ROUTE_REGRIDDED)
    dataset = _tracer_dataset(numpy.zeros((9, 9)))

    for_class4 = maybe_regridded_curvilinear_dataset(dataset, "a-curvilinear-challenger", for_class4=True)
    for_the_grid = maybe_regridded_curvilinear_dataset(dataset, "a-curvilinear-challenger")

    assert for_class4["thetao"].dims == ("latitude", "longitude")
    assert for_the_grid["thetao"].dims == ("latitude", "longitude")


def test_the_native_route_is_what_a_challenger_gets_unless_it_says_otherwise():
    latitude, longitude = _tracer_grid()
    declaration = CurvilinearChallenger(
        tracer_grid=lambda dataset: (latitude, longitude),
        tracer_ocean_mask=lambda dataset: numpy.ones(latitude.shape, dtype=bool),
    )

    assert declaration.class4_route == CLASS4_ROUTE_NATIVE


def test_an_unknown_class4_route_is_refused():
    latitude, longitude = _tracer_grid()

    with pytest.raises(ValueError, match="unknown Class IV route"):
        CurvilinearChallenger(
            tracer_grid=lambda dataset: (latitude, longitude),
            tracer_ocean_mask=lambda dataset: numpy.ones(latitude.shape, dtype=bool),
            class4_route="whatever",
        )


# ---------------------------------------------------------------------------
# The GloEns registration
# ---------------------------------------------------------------------------


def _gloens_dataset(
    values: numpy.ndarray,
    *,
    variable_name: str = "tos",
    described: str | None = "latitude",
) -> xarray.Dataset:
    latitude, longitude = _tracer_grid()
    missing = numpy.full(latitude.shape, numpy.nan)
    coordinates = {
        "latitude": (("y", "x"), latitude if described == "latitude" else missing),
        "longitude": (("y", "x"), longitude if described == "latitude" else missing),
        "nav_lat": (("y", "x"), latitude if described == "nav_lat" else missing),
        "nav_lon": (("y", "x"), longitude if described == "nav_lat" else missing),
    }
    dimensions = ("time", "ens", "y", "x")[-values.ndim :]
    return xarray.Dataset({variable_name: (dimensions, values)}, coords=coordinates)


def test_gloens_is_the_declared_curvilinear_challenger():
    declaration = CURVILINEAR_CHALLENGERS[GLOENS_SOURCE_NAME]

    assert list(CURVILINEAR_CHALLENGERS) == [GLOENS_SOURCE_NAME]
    assert declaration.source_dimensions == ("y", "x")
    assert declaration.class4_route == CLASS4_ROUTE_NATIVE
    numpy.testing.assert_array_equal(declaration.target_latitude, STANDARD_QUARTER_DEGREE_LATITUDE)
    numpy.testing.assert_array_equal(declaration.target_longitude, STANDARD_QUARTER_DEGREE_LONGITUDE)


@pytest.mark.parametrize("described", ["latitude", "nav_lat"])
def test_the_gloens_grid_comes_from_whichever_pair_holds_positions(described):
    latitude, longitude = _tracer_grid()

    read_latitude, read_longitude = gloens_tracer_grid(_gloens_dataset(numpy.zeros((9, 9)), described=described))

    numpy.testing.assert_array_equal(read_latitude, latitude)
    numpy.testing.assert_array_equal(read_longitude, longitude)


def test_a_gloens_store_whose_coordinates_are_all_missing_is_refused():
    dataset = _gloens_dataset(numpy.zeros((9, 9)), described=None)

    with pytest.raises(ValueError, match="no usable tracer grid"):
        gloens_tracer_grid(dataset)


def test_the_gloens_land_sentinel_says_which_cells_are_ocean():
    values = numpy.full((3, 2, 9, 9), 5.0)
    values[:, :, 3:6, 3:6] = GLOENS_LAND_SENTINELS["tos"]

    ocean = gloens_tracer_ocean_mask(_gloens_dataset(values))

    assert ocean.shape == (9, 9)
    assert not ocean[3:6, 3:6].any()
    assert ocean.sum() == 81 - 9


def test_an_ocean_cell_that_sits_at_the_sentinel_once_stays_ocean():
    values = numpy.full((3, 2, 9, 9), 5.0)
    values[0, 0, 2, 2] = GLOENS_LAND_SENTINELS["tos"]

    ocean = gloens_tracer_ocean_mask(_gloens_dataset(values))

    assert ocean[2, 2]


def test_the_gloens_sea_level_sentinel_is_read_when_the_store_carries_no_temperature():
    values = numpy.full((9, 9), 0.3)
    values[0, 0] = GLOENS_LAND_SENTINELS["zos"]

    ocean = gloens_tracer_ocean_mask(_gloens_dataset(values, variable_name="zos"))

    assert not ocean[0, 0]
    assert ocean.sum() == 80


def test_a_gloens_store_with_no_field_of_a_known_land_value_is_refused():
    dataset = _gloens_dataset(numpy.zeros((9, 9)), variable_name="thetao")

    with pytest.raises(ValueError, match="land cells cannot be told"):
        gloens_tracer_ocean_mask(dataset)


def test_the_gloens_declaration_regrids_a_store_through_its_own_grid_and_mask():
    values = numpy.full((9, 9), 5.0)
    values[3:6, 3:6] = GLOENS_LAND_SENTINELS["tos"]
    dataset = _gloens_dataset(values)
    declaration = CURVILINEAR_CHALLENGERS[GLOENS_SOURCE_NAME]

    regridded = regridded_curvilinear_dataset(
        dataset,
        *declaration.tracer_grid(dataset),
        declaration.tracer_ocean_mask(dataset),
        source_dimensions=declaration.source_dimensions,
        target_latitude=TARGET_LATITUDE,
        target_longitude=TARGET_LONGITUDE,
    )

    assert regridded["tos"].dims == ("latitude", "longitude")
    assert not (regridded["tos"].values == GLOENS_LAND_SENTINELS["tos"]).any()
    assert numpy.nanmin(regridded["tos"].values) == 5.0


def test_every_curvilinear_challenger_with_an_inverse_barometer_declares_its_own_shift():
    for source_name in CURVILINEAR_CHALLENGERS:
        if source_name in CHALLENGER_INVERSE_BAROMETER_VARIABLES:
            assert source_name in CHALLENGER_MEAN_SEA_SURFACE_HEIGHT_SHIFTS


def test_the_standard_target_grid_is_the_quarter_degree_scoring_grid():
    assert STANDARD_QUARTER_DEGREE_LATITUDE.shape == (672,)
    assert STANDARD_QUARTER_DEGREE_LONGITUDE.shape == (1440,)
    assert STANDARD_QUARTER_DEGREE_LATITUDE[0] == -78.0
    assert STANDARD_QUARTER_DEGREE_LATITUDE[-1] == 89.75
    assert STANDARD_QUARTER_DEGREE_LONGITUDE[0] == -180.0
    assert STANDARD_QUARTER_DEGREE_LONGITUDE[-1] == 179.75


# ---------------------------------------------------------------------------
# The stage path of a regridded week
# ---------------------------------------------------------------------------


def _staged_directory_names(tmp_path, monkeypatch, dataset_name: str, dataset: xarray.Dataset, **keywords) -> list[str]:
    monkeypatch.setattr("oceanbench.core.weekly_stage.local_stage_directory", lambda: tmp_path)
    staged_weekly_dataset(
        dataset_kind="challenger",
        dataset_name=dataset_name,
        first_day_datetimes=numpy.array([numpy.datetime64("2024-01-04")]),
        lead_days_count=10,
        open_week_dataset=lambda first_day_datetime: dataset,
        **keywords,
    )
    return sorted(path.name for path in tmp_path.iterdir())


def test_a_regridded_week_is_staged_under_a_path_of_its_own(tmp_path, monkeypatch):
    _declare(monkeypatch, CLASS4_ROUTE_NATIVE)

    names = _staged_directory_names(
        tmp_path, monkeypatch, "a-curvilinear-challenger", _tracer_dataset(numpy.zeros((9, 9)))
    )

    assert len(names) == 1
    assert names[0].startswith("challenger-a-curvilinear-challenger-regridded-")
    assert names[0].endswith("-10d")


def test_a_challenger_left_on_its_native_grid_for_class4_stages_under_yet_another_path(tmp_path, monkeypatch):
    _declare(monkeypatch, CLASS4_ROUTE_NATIVE)
    dataset = _tracer_dataset(numpy.zeros((9, 9)))

    for_the_grid = _staged_directory_names(tmp_path / "gridded", monkeypatch, "a-curvilinear-challenger", dataset)
    for_class4 = _staged_directory_names(
        tmp_path / "class4", monkeypatch, "a-curvilinear-challenger", dataset, for_class4=True
    )

    assert for_class4 == ["challenger-a-curvilinear-challenger-10d"]
    assert for_the_grid != for_class4


def test_an_existing_stage_variant_keeps_its_place_in_front_of_the_regrid_marker(monkeypatch):
    _declare(monkeypatch, CLASS4_ROUTE_REGRIDDED)

    variant = curvilinear_stage_variant("a-curvilinear-challenger", "surface")

    assert variant.startswith("surface-regridded-")
    assert curvilinear_stage_variant("a-regular-challenger", "surface") == "surface"


def test_the_stage_path_of_a_challenger_that_is_not_curvilinear_does_not_move(tmp_path, monkeypatch):
    dataset = xarray.Dataset({"thetao": (("latitude", "longitude"), numpy.zeros((3, 4)))})

    names = _staged_directory_names(tmp_path, monkeypatch, "a-regular-challenger", dataset, resolution="quarter")

    assert names == ["challenger-a-regular-challenger-quarter-10d"]


def test_a_mapping_that_reaches_nothing_describes_itself_without_percentiles():
    latitude, longitude = _tracer_grid()
    mapping = curvilinear_mapping(
        latitude,
        longitude,
        numpy.zeros(latitude.shape, dtype=bool),
        TARGET_LATITUDE,
        TARGET_LONGITUDE,
    )

    assert mapping.usable.sum() == 0
    assert mapping.describe() == "nearest neighbour on the sphere: no target cell is usable"
