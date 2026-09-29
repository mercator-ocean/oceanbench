# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import pytest
import xarray

import oceanbench.core.classIV_support as classIV_support
from oceanbench.core.classIV_support import (
    REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT,
    _compute_rmsd_table,
    _convert_forecast_ssh_to_sla,
    _interpolate_vertically_bracket,
    format_class4_results,
)
from oceanbench.core.dataset_utils import Dimension, Variable

MODEL_DEPTHS = numpy.array([10.0, 20.0, 30.0])


def _profiles(*columns: list[float]) -> numpy.ndarray:
    return numpy.array(columns, dtype=float).T


def test_bracket_interpolation_clamps_observations_outside_the_model_column_to_the_end_levels() -> None:
    interpolated = _interpolate_vertically_bracket(
        _profiles([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]),
        MODEL_DEPTHS,
        numpy.array([5.0, 35.0]),
    )

    numpy.testing.assert_array_equal(interpolated, [1.0, 3.0])


def test_bracket_interpolation_propagates_nan_from_bracketing_levels_only() -> None:
    interpolated = _interpolate_vertically_bracket(
        _profiles([1.0, numpy.nan, 3.0], [numpy.nan, 2.0, 3.0]),
        MODEL_DEPTHS,
        numpy.array([15.0, 25.0]),
    )

    assert numpy.isnan(interpolated[0])
    assert interpolated[1] == 2.5


def test_bracket_interpolation_is_invariant_under_model_level_order() -> None:
    target_depths = numpy.array([15.0, 25.0])
    sorted_result = _interpolate_vertically_bracket(_profiles([1.0, 2.0, 3.0]), MODEL_DEPTHS, target_depths[:1])
    permuted = _interpolate_vertically_bracket(
        _profiles([3.0, 1.0, 2.0], [3.0, 1.0, 2.0]),
        numpy.array([30.0, 10.0, 20.0]),
        target_depths,
    )

    numpy.testing.assert_array_equal(sorted_result, [1.5])
    numpy.testing.assert_array_equal(permuted, [1.5, 2.5])


def test_formatted_results_report_the_first_lead_day_count_per_variable_and_depth_bin() -> None:
    results_dataframe = pandas.DataFrame(
        {
            "variable": [Variable.SEA_WATER_SALINITY.key()] * 4 + [Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()] * 2,
            "depth_bin": ["0-5m", "0-5m", "5-100m", "5-100m", "surface", "surface"],
            "lead_day": [0, 1, 0, 1, 0, 1],
            "rmsd": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "count": [10, 8, 20, 15, 30, 29],
        }
    )

    formatted = format_class4_results(results_dataframe, 2)

    assert formatted["Observations"].tolist() == [30, 10, 20]
    assert formatted["Observations"].dtype.kind == "i"
    assert formatted["Lead day 2"].tolist() == [0.6, 0.2, 0.4]


def test_rmsd_table_counts_only_pairs_where_both_model_and_observation_are_finite() -> None:
    dataframe = pandas.DataFrame(
        {
            "depth_bin": ["0-5m"] * 4,
            "lead_day": [0] * 4,
            "model_value": [1.0, numpy.nan, 2.0, 3.0],
            "observation_value": [1.0, 1.0, numpy.nan, 1.0],
        }
    )

    table = _compute_rmsd_table(dataframe, Variable.SEA_WATER_SALINITY.key())

    assert table["count"].tolist() == [2]
    assert table["rmsd"].tolist() == [numpy.sqrt(2.0)]


def test_forecast_sea_surface_height_becomes_sla_by_removing_mdt_and_the_reanalysis_shift(monkeypatch) -> None:
    coordinates = {
        Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
        Dimension.LEAD_DAY_INDEX.key(): [0],
        Dimension.LATITUDE.key(): [0.0, 1.0],
        Dimension.LONGITUDE.key(): [10.0, 11.0],
    }
    zos = xarray.DataArray(
        numpy.full((1, 1, 2, 2), 1.0),
        dims=list(coordinates),
        coords=coordinates,
        name=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    )
    mean_dynamic_topography = xarray.DataArray(
        [[0.3, 0.3], [0.5, 0.5]],
        dims=[Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={Dimension.LATITUDE.key(): [0.0, 1.0], Dimension.LONGITUDE.key(): [10.0, 11.0]},
    )
    monkeypatch.setattr(classIV_support, "load_mean_dynamic_topography", lambda _resolution: mean_dynamic_topography)

    sla = _convert_forecast_ssh_to_sla(zos, Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key())

    assert REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT == -0.1148
    numpy.testing.assert_allclose(sla.values[0, 0], [[0.8148, 0.8148], [0.6148, 0.6148]])


def _quarter_degree_sea_surface_height(latitudes: numpy.ndarray, longitudes: numpy.ndarray) -> xarray.DataArray:
    coordinates = {
        Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
        Dimension.LEAD_DAY_INDEX.key(): [0],
        Dimension.LATITUDE.key(): latitudes,
        Dimension.LONGITUDE.key(): longitudes,
    }
    return xarray.DataArray(
        numpy.full((1, 1, latitudes.size, longitudes.size), 1.0),
        dims=list(coordinates),
        coords=coordinates,
        name=Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
    )


def _quarter_degree_mean_dynamic_topography(latitudes: numpy.ndarray, longitudes: numpy.ndarray) -> xarray.DataArray:
    return xarray.DataArray(
        numpy.full((latitudes.size, longitudes.size), 0.3, dtype=numpy.float32),
        dims=[Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            Dimension.LATITUDE.key(): latitudes.astype(numpy.float32),
            Dimension.LONGITUDE.key(): longitudes.astype(numpy.float32),
        },
    )


def test_sla_keeps_the_full_challenger_grid_when_its_coordinates_differ_from_the_mdt_by_rounding(
    monkeypatch,
) -> None:
    latitudes = numpy.arange(-10.0, 10.0, 0.25)
    longitudes = numpy.arange(-20.0, 20.0, 0.25)
    zos = _quarter_degree_sea_surface_height(latitudes + 1e-6, longitudes + 1e-6)
    mean_dynamic_topography = _quarter_degree_mean_dynamic_topography(latitudes, longitudes)
    monkeypatch.setattr(classIV_support, "load_mean_dynamic_topography", lambda _resolution: mean_dynamic_topography)

    sla = _convert_forecast_ssh_to_sla(zos, Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key())

    assert sla.sizes == zos.sizes
    numpy.testing.assert_array_equal(sla[Dimension.LATITUDE.key()], zos[Dimension.LATITUDE.key()])
    numpy.testing.assert_array_equal(sla[Dimension.LONGITUDE.key()], zos[Dimension.LONGITUDE.key()])
    numpy.testing.assert_allclose(sla.values, 0.8148, rtol=1e-6)


def test_sla_is_missing_where_the_challenger_grid_extends_beyond_the_mdt(monkeypatch) -> None:
    latitudes = numpy.arange(-10.0, 10.0, 0.25)
    longitudes = numpy.arange(-20.0, 20.0, 0.25)
    zos = _quarter_degree_sea_surface_height(latitudes, longitudes)
    mean_dynamic_topography = _quarter_degree_mean_dynamic_topography(latitudes[4:], longitudes)
    monkeypatch.setattr(classIV_support, "load_mean_dynamic_topography", lambda _resolution: mean_dynamic_topography)

    sla = _convert_forecast_ssh_to_sla(zos, Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key())

    assert sla.sizes == zos.sizes
    assert numpy.isnan(sla.values[..., :4, :]).all()
    numpy.testing.assert_allclose(sla.values[..., 4:, :], 0.8148, rtol=1e-6)


def test_sla_conversion_fails_when_the_challenger_grid_is_shifted_from_the_mdt(monkeypatch) -> None:
    latitudes = numpy.arange(-10.0, 10.0, 0.25)
    longitudes = numpy.arange(-20.0, 20.0, 0.25)
    zos = _quarter_degree_sea_surface_height(latitudes + 0.125, longitudes)
    mean_dynamic_topography = _quarter_degree_mean_dynamic_topography(latitudes, longitudes)
    monkeypatch.setattr(classIV_support, "load_mean_dynamic_topography", lambda _resolution: mean_dynamic_topography)

    with pytest.raises(ValueError, match="latitude"):
        _convert_forecast_ssh_to_sla(zos, Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key())
