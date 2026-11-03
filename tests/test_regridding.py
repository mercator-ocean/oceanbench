# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import sys

import numpy
import pytest
import xarray

REGRIDDING_DIRECTORY = Path(__file__).resolve().parents[1] / "helper_scripts" / "regridding"
sys.path.insert(0, str(REGRIDDING_DIRECTORY))

from croco_to_standard import (  # noqa: E402
    _eastward_and_northward_velocities,
    _interpolate_columns,
    _level_depths,
    _trapezoidal_daily_means,
)
from regrid_to_glo12_grid import _glo12_coordinates, regrid_to_glo12_grid  # noqa: E402
from to_challenger_format import to_challenger_format  # noqa: E402


def _source_grid_with_a_column_on_a_glo12_meridian() -> tuple[numpy.ndarray, numpy.ndarray]:
    # Column 20 lies exactly on 23.5 degrees east, a GLO12 meridian, so target points sit on a source cell edge,
    # where the fractional distances of pyresample alone are off by up to one source cell.
    rows, columns = numpy.meshgrid(numpy.arange(30) - 15, numpy.arange(41) - 20, indexing="ij")
    return 23.5 + columns * 0.0868, -36.0 + rows * 0.0759


def test_regrid_reproduces_a_linear_field_including_target_points_on_a_source_cell_edge() -> None:
    longitudes, latitudes = _source_grid_with_a_column_on_a_glo12_meridian()
    source = xarray.Dataset(
        {"field": (("eta_rho", "xi_rho"), longitudes + 2 * latitudes)},
        coords={"latitude": (("eta_rho", "xi_rho"), latitudes), "longitude": (("eta_rho", "xi_rho"), longitudes)},
    )

    regridded = regrid_to_glo12_grid(source)["field"]
    expected = regridded["longitude"].astype(float) + 2 * regridded["latitude"].astype(float)
    on_the_shared_meridian = regridded.sel(longitude=numpy.float32(23.5))

    assert int(regridded.notnull().sum()) > 1000
    assert int(on_the_shared_meridian.notnull().sum()) > 20
    assert float(numpy.abs(regridded - expected).max()) < 1e-8


def test_regrid_drops_nan_corners_and_renormalises_the_remaining_weights() -> None:
    longitudes, latitudes = _source_grid_with_a_column_on_a_glo12_meridian()
    coordinates = {"latitude": (("eta_rho", "xi_rho"), latitudes), "longitude": (("eta_rho", "xi_rho"), longitudes)}
    without_land = xarray.Dataset({"field": (("eta_rho", "xi_rho"), numpy.full(longitudes.shape, 5.0))}, coordinates)
    with_land = without_land.where(longitudes >= 23.0)

    covered = regrid_to_glo12_grid(without_land)["field"].notnull()
    regridded = regrid_to_glo12_grid(with_land)["field"]

    numpy.testing.assert_allclose(regridded.values[numpy.isfinite(regridded.values)], 5.0)
    assert bool(regridded.where(covered & (regridded["longitude"] > 23.1)).notnull().sum() > 0)
    assert bool((regridded.notnull() | ~(covered & (regridded["longitude"] > 23.1))).all())


def test_glo12_coordinates_are_bit_identical_to_the_stored_glo12_coordinates() -> None:
    stored_longitudes = {
        0: -180.0,
        1: -179.9166717529297,
        1237: -76.91666412353516,
        2156: -0.3333282470703125,
        2158: -0.166656494140625,
        2161: 0.083343505859375,
        2376: 18.0,
        2524: 30.333343505859375,
        4319: 179.91668701171875,
    }
    stored_latitudes = {
        0: -80.0,
        1: -79.91666412353516,
        481: -39.916664123535156,
        577: -31.916664123535156,
        1000: 3.3333358764648438,
        1104: 12.0,
        2040: 90.0,
    }

    longitudes = _glo12_coordinates(numpy.array(list(stored_longitudes)), -180.0)
    latitudes = _glo12_coordinates(numpy.array(list(stored_latitudes)), -80.0)

    assert longitudes.dtype == numpy.float32
    assert longitudes.tolist() == list(stored_longitudes.values())
    assert latitudes.tolist() == list(stored_latitudes.values())


def test_vertical_interpolation_copies_the_top_value_interpolates_linearly_and_is_nan_below_the_deepest_level() -> None:
    depths = numpy.array([[2.0, 10.0, 30.0]])
    values = numpy.array([[20.0, 16.0, 6.0]])

    interpolated = _interpolate_columns(values, depths, numpy.array([0.5, 2.0, 6.0, 20.0, 30.0, 40.0]))

    numpy.testing.assert_allclose(interpolated, [[20.0, 20.0, 18.0, 11.0, 6.0, numpy.nan]])


def test_croco_level_depths_follow_the_vtransform_2_equation() -> None:
    croco = xarray.Dataset(
        {
            "hc": 200.0,
            "h": (("eta_rho", "xi_rho"), numpy.full((3, 3), 1000.0)),
            "zeta": (("time", "eta_rho", "xi_rho"), numpy.full((1, 3, 3), 0.5)),
            "s_rho": ("s_rho", [-0.75, -0.25]),
            "Cs_rho": ("s_rho", [-0.6, -0.1]),
        }
    )
    stretching = (200.0 * numpy.array([-0.75, -0.25]) + 1000.0 * numpy.array([-0.6, -0.1])) / 1200.0

    depths = _level_depths(croco).transpose("time", "s_rho", "eta_rho", "xi_rho")

    numpy.testing.assert_allclose(depths.values[0, :, 0, 0], -(0.5 + 1000.5 * stretching))


def test_croco_velocities_are_rotated_by_the_grid_angle() -> None:
    croco = xarray.Dataset(
        {
            "u": (("time", "s_rho", "eta_rho", "xi_u"), numpy.ones((1, 1, 4, 3))),
            "v": (("time", "s_rho", "eta_v", "xi_rho"), numpy.zeros((1, 1, 3, 4))),
            "angle": (("eta_rho", "xi_rho"), numpy.full((4, 4), numpy.pi / 2)),
        }
    )

    eastward, northward = _eastward_and_northward_velocities(croco)

    numpy.testing.assert_allclose(eastward.values, 0.0, atol=1e-12)
    numpy.testing.assert_allclose(northward.values, 1.0)


def test_trapezoidal_daily_means_weight_the_midnight_snapshots_by_one_eighth() -> None:
    snapshots = xarray.Dataset(
        {"field": ("time", numpy.arange(9.0) ** 2)},
        coords={"time": numpy.datetime64("2026-10-21T00:00") + numpy.arange(9) * numpy.timedelta64(6, "h")},
    )

    daily = _trapezoidal_daily_means(snapshots)

    numpy.testing.assert_allclose(daily["field"].values, [5.5, 37.5])
    numpy.testing.assert_array_equal(daily["time"].values, numpy.array(["2026-10-21", "2026-10-22"], "datetime64[ns]"))


def test_challenger_format_turns_daily_times_into_lead_days_of_one_forecast() -> None:
    times = numpy.array(["2026-10-21", "2026-10-22", "2026-10-23"], dtype="datetime64[ns]")
    dataset = xarray.Dataset({"thetao": ("time", [1.0, 2.0, 3.0])}, coords={"time": times})

    challenger = to_challenger_format(dataset)

    assert dict(challenger.sizes) == {"first_day_datetime": 1, "lead_day_index": 3}
    assert challenger["lead_day_index"].values.tolist() == [0, 1, 2]
    assert challenger["first_day_datetime"].values.tolist() == times[:1].tolist()


def test_challenger_format_rejects_times_that_are_not_consecutive_days() -> None:
    times = numpy.array(["2026-10-21T00", "2026-10-21T06"], dtype="datetime64[ns]")
    dataset = xarray.Dataset({"thetao": ("time", [1.0, 2.0])}, coords={"time": times})

    with pytest.raises(ValueError, match="consecutive days"):
        to_challenger_format(dataset)
