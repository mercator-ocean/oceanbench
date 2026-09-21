# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pandas
import xarray

import oceanbench.core.classIV_support as classIV_support
from oceanbench.core.classIV_support import (
    REANALYSIS_MEAN_SEA_SURFACE_HEIGHT_SHIFT,
    _compute_rmsd_table,
    _convert_forecast_ssh_to_sla,
    _interpolate_vertically_bracket,
    format_class4_results,
    gate_class4_observations_to_reference_population,
)
from oceanbench.core.classIV import rmsd_class4_validation
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.references.glo12 import OCEAN_MASK_DEPTHS

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


MASK_DEPTHS = numpy.array([0.5, 10.0, 50.0])


def _ocean_mask() -> xarray.DataArray:
    values = numpy.full((len(MASK_DEPTHS), len(LATITUDES), len(LONGITUDES)), True)
    values[2, 0, 2] = False
    return xarray.DataArray(
        values,
        dims=[Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            Dimension.DEPTH.key(): MASK_DEPTHS,
            Dimension.LATITUDE.key(): LATITUDES,
            Dimension.LONGITUDE.key(): LONGITUDES,
        },
        name=Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
    )


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
    gated = gate_class4_observations_to_reference_population(observations_dataframe, _ocean_mask())
    return gated.index.tolist()


def test_gate_keeps_a_surface_observation_where_the_first_mask_level_is_wet() -> None:
    # Sea level anomaly is scored on a single fake level at depth zero, so both bracketing indices
    # fall on the first mask level.
    assert _gated_depths([0.0, 0.0], [10.0, 12.0], [0.0, 0.0]) == [0, 1]


def test_gate_drops_an_observation_outside_the_mask_horizontal_range() -> None:
    assert _gated_depths([0.5, 5.0], [10.5, 10.5], [20.0, 20.0]) == [0]


def test_gate_clamps_an_observation_deeper_than_the_last_mask_level_to_that_level() -> None:
    assert _gated_depths([0.0, 0.0], [10.0, 12.0], [100.0, 100.0]) == [0]


def test_ocean_mask_depths_are_glo12_native_levels() -> None:
    expected_native_levels = [
        0.494025,
        47.37369,
        92.32607,
        155.85069,
        222.47520,
        318.12741,
        380.21301,
        453.93771,
        541.08893,
        643.56677,
    ]

    numpy.testing.assert_allclose(OCEAN_MASK_DEPTHS, expected_native_levels, atol=1e-3)
