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
    class4_observations_in_shared_population,
)
from oceanbench.core.classIV import rmsd_class4_validation
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.ocean_mask import OCEAN_MASK_DEPTHS

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


def test_bracket_interpolation_propagates_nan_from_the_shallower_bracketing_level() -> None:
    interpolated = _interpolate_vertically_bracket(
        _profiles([numpy.nan, 2.0, 3.0], [1.0, 2.0, 3.0]),
        MODEL_DEPTHS,
        numpy.array([15.0, 25.0]),
    )

    assert numpy.isnan(interpolated[0])
    assert interpolated[1] == 2.5


def test_bracket_interpolation_propagates_nan_from_the_deeper_bracketing_level() -> None:
    interpolated = _interpolate_vertically_bracket(
        _profiles([1.0, 2.0, numpy.nan], [1.0, 2.0, 3.0]),
        MODEL_DEPTHS,
        numpy.array([25.0, 25.0]),
    )

    assert numpy.isnan(interpolated[0])
    assert interpolated[1] == 2.5


def test_bracket_interpolation_returns_nan_when_the_whole_column_is_missing() -> None:
    interpolated = _interpolate_vertically_bracket(
        _profiles([numpy.nan, numpy.nan, numpy.nan]),
        MODEL_DEPTHS,
        numpy.array([25.0]),
    )

    assert numpy.isnan(interpolated).all()


def test_bracket_interpolation_keeps_the_deep_and_top_clamps_unchanged() -> None:
    interpolated = _interpolate_vertically_bracket(
        _profiles([1.0, 2.0, numpy.nan], [1.0, 2.0, 3.0]),
        MODEL_DEPTHS,
        numpy.array([35.0, 5.0]),
    )

    assert numpy.isnan(interpolated[0])
    assert interpolated[1] == 1.0


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
            "missing": [1, 0, 2, 0, 3, 0],
        }
    )

    formatted = format_class4_results(results_dataframe, 2)

    assert formatted["Observations"].tolist() == [30, 10, 20]
    assert formatted["Observations"].dtype.kind == "i"
    assert formatted["Missing"].tolist() == [3, 1, 2]
    assert formatted["Lead day 2"].tolist() == [0.6, 0.2, 0.4]


def test_rmsd_table_counts_every_eligible_observation_and_reports_the_missing_ones() -> None:
    dataframe = pandas.DataFrame(
        {
            "depth_bin": ["0-5m"] * 4,
            "lead_day": [0] * 4,
            "model_value": [1.0, numpy.nan, 2.0, 3.0],
            "observation_value": [1.0, 1.0, numpy.nan, 1.0],
        }
    )

    table = _compute_rmsd_table(dataframe, Variable.SEA_WATER_SALINITY.key())

    assert table["count"].tolist() == [3]
    assert table["missing"].tolist() == [1]
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


LATITUDES = numpy.array([0.0, 1.0, 2.0])
LONGITUDES = numpy.array([10.0, 11.0, 12.0])
FIRST_DAYS = numpy.array(["2024-01-03"], dtype="datetime64[ns]")


def _salinity_dataset(depths: numpy.ndarray, missing_column: bool) -> xarray.Dataset:
    values = (
        numpy.broadcast_to(
            35.0 + depths[:, numpy.newaxis, numpy.newaxis] / 10.0,
            (len(depths), len(LATITUDES), len(LONGITUDES)),
        )
        .astype(float)
        .copy()
    )
    if missing_column:
        values[:, 2, 2] = numpy.nan
    return xarray.Dataset(
        {
            Variable.SEA_WATER_SALINITY.key(): (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.DEPTH.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                values[numpy.newaxis, numpy.newaxis, :, :, :],
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): FIRST_DAYS,
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.DEPTH.key(): depths,
            Dimension.LATITUDE.key(): LATITUDES,
            Dimension.LONGITUDE.key(): LONGITUDES,
        },
    )


def _twelfth_degree_ocean_mask(
    is_wet: numpy.ndarray,
    first_latitude: float = 0.0,
    first_longitude: float = 10.0,
) -> xarray.DataArray:
    _, latitude_count, longitude_count = is_wet.shape
    return xarray.DataArray(
        is_wet,
        dims=[Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            Dimension.DEPTH.key(): OCEAN_MASK_DEPTHS,
            Dimension.LATITUDE.key(): first_latitude + numpy.arange(latitude_count) / 12.0,
            Dimension.LONGITUDE.key(): first_longitude + numpy.arange(longitude_count) / 12.0,
        },
    )


def _all_wet(latitude_count: int, longitude_count: int) -> numpy.ndarray:
    return numpy.full((len(OCEAN_MASK_DEPTHS), latitude_count, longitude_count), True)


def _ocean_mask() -> xarray.DataArray:
    # Twelfth of a degree over the challenger grid, with one shallow cell in the quarter degree
    # cell next to the observation at (0.5, 11.5) but not among its twelfth of a degree corners.
    is_wet = _all_wet(25, 25)
    is_wet[1:, 5, 17] = False
    return _twelfth_degree_ocean_mask(is_wet)


def _salinity_observations_dataset() -> xarray.Dataset:
    observation_dimension = "observation"
    return xarray.Dataset(
        {
            Dimension.TIME.key(): (observation_dimension, numpy.repeat(FIRST_DAYS, 3)),
            Dimension.LATITUDE.key(): (observation_dimension, numpy.array([0.25, 1.5, 0.5])),
            Dimension.LONGITUDE.key(): (observation_dimension, numpy.array([10.25, 11.5, 11.5])),
            Dimension.FIRST_DAY_DATETIME.key(): (observation_dimension, numpy.repeat(FIRST_DAYS, 3)),
            Dimension.DEPTH.key(): (observation_dimension, numpy.array([20.0, 20.0, 20.0])),
            Variable.SEA_WATER_SALINITY.key(): (observation_dimension, numpy.array([35.0, 35.0, 35.0])),
        }
    )


def test_challengers_with_different_vertical_axes_share_the_scored_observation_population() -> None:
    observations_dataset = _salinity_observations_dataset()

    formatted_tables = [
        rmsd_class4_validation(
            challenger_dataset=_salinity_dataset(challenger_depths, missing_column=missing_column),
            reference_dataset=observations_dataset,
            ocean_mask=_ocean_mask(),
            variables=[Variable.SEA_WATER_SALINITY],
        )
        for challenger_depths, missing_column in [
            (numpy.array([0.5, 10.0, 50.0]), False),
            (numpy.array([0.5, 47.0]), True),
        ]
    ]

    assert [table["Observations"].tolist() for table in formatted_tables] == [[2], [2]]
    assert [table["Missing"].tolist() for table in formatted_tables] == [[0], [1]]
    assert [table.index.tolist() for table in formatted_tables] == [["Salinity (PSU) [sea_water_salinity]{5-100m}"]] * 2


def _gated_depths(latitudes: list[float], longitudes: list[float], depths: list[float]) -> list[float]:
    observations_dataframe = pandas.DataFrame(
        {
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
            Dimension.DEPTH.key(): depths,
        }
    )
    gated = class4_observations_in_shared_population(observations_dataframe, _ocean_mask())
    return gated.index.tolist()


def test_gate_keeps_a_surface_observation_where_the_first_mask_level_is_wet() -> None:
    # Sea level anomaly is scored on a single fake level at depth zero, so both bracketing indices
    # fall on the first mask level.
    assert _gated_depths([0.0, 0.0], [10.0, 12.0], [0.0, 0.0]) == [0, 1]


def test_gate_clamps_an_observation_deeper_than_the_last_mask_level_to_that_level() -> None:
    is_wet = _all_wet(25, 25)
    is_wet[-1, 18, 18] = False
    observations_dataframe = pandas.DataFrame(
        {
            Dimension.LATITUDE.key(): [0.25, 1.5],
            Dimension.LONGITUDE.key(): [10.25, 11.5],
            Dimension.DEPTH.key(): [700.0, 700.0],
        }
    )

    gated = class4_observations_in_shared_population(observations_dataframe, _twelfth_degree_ocean_mask(is_wet))

    assert gated.index.tolist() == [0]


def test_ocean_mask_depths_are_twelfth_degree_native_levels() -> None:
    expected_native_levels = [
        0.494025,
        47.37369,
        92.32607,
        222.47520,
        318.12741,
        541.08893,
        643.56677,
    ]

    numpy.testing.assert_allclose(OCEAN_MASK_DEPTHS, expected_native_levels, atol=1e-3)


def _mask_on_the_ocean_mask_depths(wet_below_600_meters: bool) -> xarray.DataArray:
    is_wet = _all_wet(25, 25)
    is_wet[OCEAN_MASK_DEPTHS > 600.0] = wet_below_600_meters
    return _twelfth_degree_ocean_mask(is_wet)


def test_gate_vets_an_observation_of_the_deepest_depth_bin_against_the_level_below_600_meters() -> None:
    observations_dataframe = pandas.DataFrame(
        {
            Dimension.LATITUDE.key(): [1.0],
            Dimension.LONGITUDE.key(): [11.0],
            Dimension.DEPTH.key(): [580.0],
        }
    )

    dry_below = class4_observations_in_shared_population(
        observations_dataframe,
        _mask_on_the_ocean_mask_depths(wet_below_600_meters=False),
    )
    wet_below = class4_observations_in_shared_population(
        observations_dataframe,
        _mask_on_the_ocean_mask_depths(wet_below_600_meters=True),
    )

    assert dry_below.index.tolist() == []
    assert wet_below.index.tolist() == [0]


def test_formatted_results_keep_a_variable_and_depth_bin_whose_scores_are_all_missing() -> None:
    results_dataframe = pandas.DataFrame(
        {
            "variable": [Variable.SEA_WATER_SALINITY.key()] * 2 + [Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()] * 2,
            "depth_bin": ["0-5m", "0-5m", "surface", "surface"],
            "lead_day": [0, 1, 0, 1],
            "rmsd": [numpy.nan, numpy.nan, 0.5, 0.6],
            "count": [10, 10, 30, 30],
            "missing": [10, 10, 0, 0],
        }
    )

    formatted = format_class4_results(results_dataframe, 2)

    assert formatted.index.tolist() == [
        "Temperature (\u00b0C) [sea_water_potential_temperature]{surface}",
        "Salinity (PSU) [sea_water_salinity]{0-5m}",
    ]
    assert formatted["Missing"].tolist() == [0, 10]
    assert numpy.isnan(formatted.loc["Salinity (PSU) [sea_water_salinity]{0-5m}", "Lead day 1"])


def test_formatted_results_keep_a_lead_day_with_no_scored_value_at_all() -> None:
    results_dataframe = pandas.DataFrame(
        {
            "variable": [Variable.SEA_WATER_SALINITY.key()] * 2,
            "depth_bin": ["0-5m", "0-5m"],
            "lead_day": [0, 1],
            "rmsd": [0.1, numpy.nan],
            "count": [10, 10],
            "missing": [0, 10],
        }
    )

    formatted = format_class4_results(results_dataframe, 3)

    assert formatted.columns.tolist() == ["Lead day 1", "Lead day 2", "Lead day 3", "Observations", "Missing"]
    assert formatted["Lead day 1"].tolist() == [0.1]
    assert numpy.isnan(formatted["Lead day 2"]).all()
    assert numpy.isnan(formatted["Lead day 3"]).all()


def test_gate_keeps_observations_across_the_dateline_on_a_global_mask() -> None:
    ocean_mask = _twelfth_degree_ocean_mask(_all_wet(25, 4320), first_latitude=-1.0, first_longitude=-180.0)
    observations_dataframe = pandas.DataFrame(
        {
            Dimension.LATITUDE.key(): [0.0, 0.0, 0.0, 0.0],
            Dimension.LONGITUDE.key(): [179.95, 180.0, -180.0, 0.0],
            Dimension.DEPTH.key(): [20.0, 20.0, 20.0, 20.0],
        }
    )

    gated = class4_observations_in_shared_population(observations_dataframe, ocean_mask)

    assert gated.index.tolist() == [0, 1, 2, 3]


def _surface_observations(latitudes: list[float], longitudes: list[float]) -> pandas.DataFrame:
    return pandas.DataFrame(
        {
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
            Dimension.DEPTH.key(): [0.0] * len(latitudes),
        }
    )


def _make_shallow(is_wet: numpy.ndarray, rows: slice, columns: slice) -> None:
    is_wet[2:, rows, columns] = False


def test_gate_drops_an_observation_with_a_shallow_corner_cell() -> None:
    is_wet = _all_wet(61, 61)
    _make_shallow(is_wet, slice(30, 31), slice(30, 31))
    observations_dataframe = _surface_observations([2.5 + 0.5 / 12, 1.0], [12.5 + 0.5 / 12, 11.0])

    gated = class4_observations_in_shared_population(observations_dataframe, _twelfth_degree_ocean_mask(is_wet))

    assert gated.index.tolist() == [1]


def test_gate_keeps_an_observation_in_a_shallow_region_larger_than_the_size_threshold() -> None:
    # 1600 cells of about 86 square kilometres at the equator is about 137,000 square kilometres,
    # while the four cell region is far below the threshold.
    is_wet = _all_wet(61, 61)
    _make_shallow(is_wet, slice(10, 50), slice(10, 50))
    _make_shallow(is_wet, slice(55, 57), slice(55, 57))
    observations_dataframe = _surface_observations(
        [2.5 + 0.5 / 12, 55 / 12 + 0.5 / 12], [12.5 + 0.5 / 12, 10 + 55.5 / 12]
    )

    gated = class4_observations_in_shared_population(observations_dataframe, _twelfth_degree_ocean_mask(is_wet))

    assert gated.index.tolist() == [0]


def test_gate_sums_the_area_of_a_shallow_region_across_the_dateline() -> None:
    # Each half of the region straddling the dateline is below the threshold and only their sum is
    # above it, like the lone region of the same size as one half.
    is_wet = _all_wet(61, 4320)
    _make_shallow(is_wet, slice(10, 50), slice(0, 20))
    _make_shallow(is_wet, slice(10, 50), slice(4300, 4320))
    _make_shallow(is_wet, slice(10, 50), slice(2000, 2020))
    observations_dataframe = _surface_observations(
        [2.5 + 0.5 / 12, 2.5 + 0.5 / 12, 2.5 + 0.5 / 12],
        [-180.0 + 5.5 / 12, 179.5 + 0.5 / 12, -180.0 + 2005.5 / 12],
    )
    ocean_mask = _twelfth_degree_ocean_mask(is_wet, first_latitude=0.0, first_longitude=-180.0)

    gated = class4_observations_in_shared_population(observations_dataframe, ocean_mask)

    assert gated.index.tolist() == [0, 1]


def _observation_at_100_meters_in_quarter_degree_cells_10_and_11() -> pandas.DataFrame:
    return pandas.DataFrame(
        {
            Dimension.LATITUDE.key(): [2.55],
            Dimension.LONGITUDE.key(): [12.55],
            Dimension.DEPTH.key(): [100.0],
        }
    )


def test_gate_drops_an_observation_next_to_a_dry_quarter_degree_cell() -> None:
    # The dry fine cell is not one of the observation twelfth of a degree corners, and is wet at
    # 92 metres so it is not shallow either: only the quarter degree cell centred on (33, 33) sees it.
    is_wet = _all_wet(61, 61)
    is_wet[3:, 34, 34] = False

    gated = class4_observations_in_shared_population(
        _observation_at_100_meters_in_quarter_degree_cells_10_and_11(),
        _twelfth_degree_ocean_mask(is_wet),
    )

    assert gated.index.tolist() == []


def test_gate_keeps_an_observation_whose_four_quarter_degree_cells_are_wet() -> None:
    is_wet = _all_wet(61, 61)
    is_wet[3:, 36, 36] = False

    gated = class4_observations_in_shared_population(
        _observation_at_100_meters_in_quarter_degree_cells_10_and_11(),
        _twelfth_degree_ocean_mask(is_wet),
    )

    assert gated.index.tolist() == [0]
