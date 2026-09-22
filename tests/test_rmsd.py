# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.ocean_mask import OCEAN_MASK_DEPTHS
from oceanbench.core.rmsd import (
    MISSING_COUNT_COLUMN,
    MISSING_FRACTION_COLUMN,
    _rmsd,
    rmsd,
)


def _dataset_with_spatial_coordinates(
    latitudes: numpy.ndarray,
    longitudes: numpy.ndarray,
    values: numpy.ndarray,
) -> xarray.Dataset:
    variable_key = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
    return xarray.Dataset(
        {
            variable_key: (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                values,
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )


def test_rmsd_uses_area_weights_without_land_in_denominator() -> None:
    variable_key = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
    values = numpy.array(
        [
            [
                [
                    [1.0, numpy.nan],
                    [3.0, 5.0],
                ]
            ],
            [
                [
                    [2.0, 4.0],
                    [numpy.nan, 6.0],
                ]
            ],
        ]
    )
    coordinates = {
        Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]"),
        Dimension.LEAD_DAY_INDEX.key(): [0],
        Dimension.LATITUDE.key(): [0.0, 60.0],
        Dimension.LONGITUDE.key(): [10.0, 11.0],
    }
    challenger_dataset = xarray.Dataset(
        {
            variable_key: (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                values,
            )
        },
        coords=coordinates,
    )
    reference_dataset = xarray.zeros_like(challenger_dataset)

    rmsd_dataset = _rmsd(challenger_dataset, reference_dataset)

    expected_first_day_rmsd = numpy.sqrt((1.0**2 * 1.0 + 3.0**2 * 0.5 + 5.0**2 * 0.5) / (1.0 + 0.5 + 0.5))
    expected_second_day_rmsd = numpy.sqrt((2.0**2 * 1.0 + 4.0**2 * 1.0 + 6.0**2 * 0.5) / (1.0 + 1.0 + 0.5))
    expected_rmsd = (expected_first_day_rmsd + expected_second_day_rmsd) / 2.0
    naive_land_weighted_rmsd = numpy.sqrt((1.0**2 * 1.0 + 3.0**2 * 0.5 + 5.0**2 * 0.5) / (1.0 + 1.0 + 0.5 + 0.5))
    actual_rmsd = float(rmsd_dataset[variable_key].sel({Dimension.LEAD_DAY_INDEX.key(): 0}))

    assert numpy.isclose(actual_rmsd, expected_rmsd)
    assert not numpy.isclose(actual_rmsd, naive_land_weighted_rmsd)


def test_rmsd_snaps_nearly_matching_spatial_coordinates_before_xarray_alignment() -> None:
    variable_key = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
    challenger_latitudes = numpy.array([0.0, 1.00001, 2.0], dtype=numpy.float32)
    challenger_longitudes = numpy.array([10.0, 11.00001, 12.0], dtype=numpy.float32)
    reference_latitudes = numpy.array([0.0, 1.0, 2.0], dtype=numpy.float32)
    reference_longitudes = numpy.array([10.0, 11.0, 12.0], dtype=numpy.float32)
    challenger_values = numpy.array(
        [
            [
                [
                    [1.0, 10.0, 2.0],
                    [100.0, 200.0, 300.0],
                    [3.0, 400.0, 4.0],
                ]
            ]
        ]
    )
    dimension_names = [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    base_coordinates = {
        Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
        Dimension.LEAD_DAY_INDEX.key(): [0],
    }
    challenger_dataset = xarray.Dataset(
        {
            variable_key: (
                dimension_names,
                challenger_values,
            )
        },
        coords={
            **base_coordinates,
            Dimension.LATITUDE.key(): challenger_latitudes,
            Dimension.LONGITUDE.key(): challenger_longitudes,
        },
    )
    reference_dataset = xarray.Dataset(
        {
            variable_key: (
                dimension_names,
                numpy.zeros_like(challenger_values),
            )
        },
        coords={
            **base_coordinates,
            Dimension.LATITUDE.key(): reference_latitudes,
            Dimension.LONGITUDE.key(): reference_longitudes,
        },
    )

    rmsd_dataset = _rmsd(challenger_dataset, reference_dataset)

    latitude_weights = numpy.cos(numpy.deg2rad(challenger_latitudes))[:, numpy.newaxis]
    expected_rmsd = numpy.sqrt(
        numpy.sum(challenger_values[0, 0] ** 2 * latitude_weights)
        / numpy.sum(numpy.ones_like(challenger_values[0, 0]) * latitude_weights)
    )
    legacy_inner_join_values = challenger_values[0, 0][[0, 2]][:, [0, 2]]
    legacy_inner_join_latitudes = reference_latitudes[[0, 2]]
    legacy_latitude_weights = numpy.cos(numpy.deg2rad(legacy_inner_join_latitudes))[:, numpy.newaxis]
    legacy_inner_join_rmsd = numpy.sqrt(
        numpy.sum(legacy_inner_join_values**2 * legacy_latitude_weights)
        / numpy.sum(numpy.ones_like(legacy_inner_join_values) * legacy_latitude_weights)
    )
    actual_rmsd = float(rmsd_dataset[variable_key].sel({Dimension.LEAD_DAY_INDEX.key(): 0}))

    assert numpy.isclose(actual_rmsd, expected_rmsd)
    assert not numpy.isclose(actual_rmsd, legacy_inner_join_rmsd)


def test_rmsd_snaps_reference_to_challenger_when_challenger_has_one_extra_coordinate() -> None:
    variable_key = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
    matched_challenger_latitudes = numpy.linspace(-50.0, 50.0, 1000, dtype=numpy.float32)
    challenger_latitudes = numpy.concatenate(
        [
            matched_challenger_latitudes,
            numpy.array([51.0], dtype=numpy.float32),
        ]
    )
    challenger_longitudes = numpy.array([10.0, 20.0], dtype=numpy.float32)
    reference_latitudes = matched_challenger_latitudes + numpy.float32(1e-5)
    reference_longitudes = challenger_longitudes + numpy.float32(1e-5)
    challenger_values = numpy.arange(challenger_latitudes.size * challenger_longitudes.size, dtype=float).reshape(
        1,
        1,
        challenger_latitudes.size,
        challenger_longitudes.size,
    )
    dimension_names = [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    base_coordinates = {
        Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
        Dimension.LEAD_DAY_INDEX.key(): [0],
    }
    challenger_dataset = xarray.Dataset(
        {
            variable_key: (
                dimension_names,
                challenger_values,
            )
        },
        coords={
            **base_coordinates,
            Dimension.LATITUDE.key(): challenger_latitudes,
            Dimension.LONGITUDE.key(): challenger_longitudes,
        },
    )
    reference_dataset = xarray.Dataset(
        {
            variable_key: (
                dimension_names,
                numpy.zeros(
                    (
                        1,
                        1,
                        reference_latitudes.size,
                        reference_longitudes.size,
                    )
                ),
            )
        },
        coords={
            **base_coordinates,
            Dimension.LATITUDE.key(): reference_latitudes,
            Dimension.LONGITUDE.key(): reference_longitudes,
        },
    )

    rmsd_dataset = _rmsd(challenger_dataset, reference_dataset)

    latitude_weights = numpy.cos(numpy.deg2rad(matched_challenger_latitudes))[:, numpy.newaxis]
    matched_challenger_values = challenger_values[0, 0, : matched_challenger_latitudes.size]
    expected_rmsd = numpy.sqrt(
        numpy.sum(matched_challenger_values**2 * latitude_weights)
        / numpy.sum(numpy.ones_like(matched_challenger_values) * latitude_weights)
    )
    legacy_inner_join_squared_error = (challenger_dataset - reference_dataset) ** 2
    actual_rmsd = float(rmsd_dataset[variable_key].sel({Dimension.LEAD_DAY_INDEX.key(): 0}))

    assert legacy_inner_join_squared_error.sizes[Dimension.LATITUDE.key()] == 0
    assert legacy_inner_join_squared_error.sizes[Dimension.LONGITUDE.key()] == 0
    assert numpy.isclose(actual_rmsd, expected_rmsd)


def test_rmsd_raises_when_spatial_coordinate_alignment_is_ambiguous() -> None:
    challenger_dataset = _dataset_with_spatial_coordinates(
        latitudes=numpy.array([0.0, 0.00001], dtype=numpy.float32),
        longitudes=numpy.array([10.0], dtype=numpy.float32),
        values=numpy.zeros((1, 1, 2, 1), dtype=float),
    )
    reference_dataset = _dataset_with_spatial_coordinates(
        latitudes=numpy.array([0.0], dtype=numpy.float32),
        longitudes=numpy.array([10.0], dtype=numpy.float32),
        values=numpy.zeros((1, 1, 1, 1), dtype=float),
    )

    with pytest.raises(ValueError, match="latitude coordinates: multiple challenger coordinates match"):
        _rmsd(challenger_dataset, reference_dataset)


def test_rmsd_raises_when_too_much_spatial_grid_is_unmatched() -> None:
    challenger_latitudes = numpy.linspace(-50.0, 50.0, 1000, dtype=numpy.float32)
    challenger_longitudes = numpy.array([10.0, 20.0], dtype=numpy.float32)
    reference_latitudes = challenger_latitudes[:998] + numpy.float32(1e-5)
    challenger_dataset = _dataset_with_spatial_coordinates(
        latitudes=challenger_latitudes,
        longitudes=challenger_longitudes,
        values=numpy.zeros((1, 1, challenger_latitudes.size, challenger_longitudes.size), dtype=float),
    )
    reference_dataset = _dataset_with_spatial_coordinates(
        latitudes=reference_latitudes,
        longitudes=challenger_longitudes,
        values=numpy.zeros((1, 1, reference_latitudes.size, challenger_longitudes.size), dtype=float),
    )

    with pytest.raises(
        ValueError,
        match="matched 99.8000%.*required at least 99.9000%.*latitude=99.8000%.*longitude=100.0000%",
    ):
        _rmsd(challenger_dataset, reference_dataset)


def test_rmsd_takes_the_square_root_per_first_day_and_depth_before_averaging_over_first_days() -> None:
    variable_key = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
    dimension_names = [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.DEPTH.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    # Latitude weights are cos(0) = 1 and cos(60) = 0.5, so a column [a, b] has weighted mean square
    # (a^2 + b^2 / 2) / 1.5: [1, 5] -> 9, [2, 10] -> 36, [a, a] -> a^2.
    values = numpy.array(
        [
            [[[[1.0], [5.0]], [[1.0], [1.0]]], [[[2.0], [2.0]], [[2.0], [10.0]]]],
            [[[[2.0], [10.0]], [[1.0], [5.0]]], [[[4.0], [4.0]], [[6.0], [6.0]]]],
        ]
    )
    coordinates = {
        Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]"),
        Dimension.LEAD_DAY_INDEX.key(): [0, 1],
        Dimension.DEPTH.key(): [0.5, 50.0],
        Dimension.LATITUDE.key(): [0.0, 60.0],
        Dimension.LONGITUDE.key(): [10.0],
    }
    challenger_dataset = xarray.Dataset({variable_key: (dimension_names, values)}, coords=coordinates)

    rmsd_dataset = _rmsd(challenger_dataset, xarray.zeros_like(challenger_dataset))

    numpy.testing.assert_allclose(
        rmsd_dataset[variable_key].transpose(Dimension.LEAD_DAY_INDEX.key(), Dimension.DEPTH.key()).values,
        [[(3.0 + 6.0) / 2, (1.0 + 3.0) / 2], [(2.0 + 4.0) / 2, (6.0 + 6.0) / 2]],
    )


MASK_TEST_LATITUDES = numpy.array([0.0, 30.0, 60.0])
MASK_TEST_LONGITUDES = numpy.array([10.0])


def _temperature_dataset(surface_values: list[float], deep_value: float) -> xarray.Dataset:
    variable_key = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
    depths = OCEAN_MASK_DEPTHS
    values = numpy.full((1, 1, len(depths), len(MASK_TEST_LATITUDES), len(MASK_TEST_LONGITUDES)), deep_value)
    values[0, 0, 0, :, 0] = surface_values
    return xarray.Dataset(
        {
            variable_key: (
                [
                    Dimension.FIRST_DAY_DATETIME.key(),
                    Dimension.LEAD_DAY_INDEX.key(),
                    Dimension.DEPTH.key(),
                    Dimension.LATITUDE.key(),
                    Dimension.LONGITUDE.key(),
                ],
                values,
            )
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.DEPTH.key(): depths,
            Dimension.LATITUDE.key(): MASK_TEST_LATITUDES,
            Dimension.LONGITUDE.key(): MASK_TEST_LONGITUDES,
        },
    )


def _surface_dry_at_sixty_degrees_mask() -> xarray.DataArray:
    values = numpy.ones(
        (len(OCEAN_MASK_DEPTHS), len(MASK_TEST_LATITUDES), len(MASK_TEST_LONGITUDES)),
        dtype=bool,
    )
    values[0, 2, 0] = False
    return xarray.DataArray(
        values,
        dims=[Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            Dimension.DEPTH.key(): OCEAN_MASK_DEPTHS,
            Dimension.LATITUDE.key(): MASK_TEST_LATITUDES,
            Dimension.LONGITUDE.key(): MASK_TEST_LONGITUDES,
        },
    )


SURFACE_TEMPERATURE_LABEL = "Temperature (°C) [sea_water_potential_temperature]{surface}"
FIFTY_METERS_TEMPERATURE_LABEL = "Temperature (°C) [sea_water_potential_temperature]{50m}"


def test_rmsd_scores_only_the_mask_wet_cells_and_reports_the_missing_ones() -> None:
    challenger_dataset = _temperature_dataset(surface_values=[numpy.nan, 4.0, 100.0], deep_value=1.0)
    reference_dataset = xarray.zeros_like(challenger_dataset)

    table = rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=reference_dataset,
        variables=[Variable.SEA_WATER_POTENTIAL_TEMPERATURE],
        ocean_mask=_surface_dry_at_sixty_degrees_mask(),
    )

    assert table.loc[SURFACE_TEMPERATURE_LABEL, "Lead day 1"] == 4.0
    assert table.loc[SURFACE_TEMPERATURE_LABEL, MISSING_COUNT_COLUMN] == 1
    assert numpy.isclose(
        table.loc[SURFACE_TEMPERATURE_LABEL, MISSING_FRACTION_COLUMN],
        1.0 / (1.0 + numpy.cos(numpy.deg2rad(30.0))),
    )
    assert table.loc[FIFTY_METERS_TEMPERATURE_LABEL, "Lead day 1"] == 1.0
    assert table.loc[FIFTY_METERS_TEMPERATURE_LABEL, MISSING_COUNT_COLUMN] == 0
    assert table.loc[FIFTY_METERS_TEMPERATURE_LABEL, MISSING_FRACTION_COLUMN] == 0.0


def test_rmsd_excludes_a_mask_dry_cell_even_when_both_sides_are_finite() -> None:
    challenger_dataset = _temperature_dataset(surface_values=[4.0, 4.0, 100.0], deep_value=1.0)
    reference_dataset = xarray.zeros_like(challenger_dataset)

    masked_table = rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=reference_dataset,
        variables=[Variable.SEA_WATER_POTENTIAL_TEMPERATURE],
        ocean_mask=_surface_dry_at_sixty_degrees_mask(),
    )
    fully_wet_mask = _surface_dry_at_sixty_degrees_mask()
    fully_wet_mask.values[0, 2, 0] = True
    unmasked_table = rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=reference_dataset,
        variables=[Variable.SEA_WATER_POTENTIAL_TEMPERATURE],
        ocean_mask=fully_wet_mask,
    )

    assert masked_table.loc[SURFACE_TEMPERATURE_LABEL, "Lead day 1"] == 4.0
    assert masked_table.loc[SURFACE_TEMPERATURE_LABEL, MISSING_COUNT_COLUMN] == 0
    assert unmasked_table.loc[SURFACE_TEMPERATURE_LABEL, "Lead day 1"] > 40.0


def _depth_free_dataset(variable_values: dict[str, list[float]]) -> xarray.Dataset:
    dimensions = [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    return xarray.Dataset(
        {
            variable_key: (
                dimensions,
                numpy.array(values).reshape(1, 1, len(MASK_TEST_LATITUDES), len(MASK_TEST_LONGITUDES)),
            )
            for variable_key, values in variable_values.items()
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): numpy.array(["2024-01-03"], dtype="datetime64[ns]"),
            Dimension.LEAD_DAY_INDEX.key(): [0],
            Dimension.DEPTH.key(): OCEAN_MASK_DEPTHS,
            Dimension.LATITUDE.key(): MASK_TEST_LATITUDES,
            Dimension.LONGITUDE.key(): MASK_TEST_LONGITUDES,
        },
    )


MIXED_LAYER_DEPTH_LABEL = "Mixed layer depth (m) [ocean_mixed_layer_thickness]{surface}"
MERIDIONAL_GEOSTROPHIC_LABEL = (
    "Meridional geostrophic current (m/s) [geostrophic_northward_sea_water_velocity]{surface}"
)
ZONAL_GEOSTROPHIC_LABEL = "Zonal geostrophic current (m/s) [geostrophic_eastward_sea_water_velocity]{surface}"


def test_rmsd_scores_a_dataset_whose_only_variable_has_no_depth() -> None:
    variable_key = Variable.MIXED_LAYER_DEPTH.key()
    challenger_dataset = _depth_free_dataset({variable_key: [numpy.nan, 4.0, 100.0]})
    reference_dataset = xarray.zeros_like(challenger_dataset)

    table = rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=reference_dataset,
        variables=[Variable.MIXED_LAYER_DEPTH],
        ocean_mask=_surface_dry_at_sixty_degrees_mask(),
    )

    assert list(table.index) == [MIXED_LAYER_DEPTH_LABEL]
    assert table.loc[MIXED_LAYER_DEPTH_LABEL, "Lead day 1"] == 4.0
    assert table.loc[MIXED_LAYER_DEPTH_LABEL, MISSING_COUNT_COLUMN] == 1
    assert numpy.isclose(
        table.loc[MIXED_LAYER_DEPTH_LABEL, MISSING_FRACTION_COLUMN],
        1.0 / (1.0 + numpy.cos(numpy.deg2rad(30.0))),
    )


def test_rmsd_scores_a_dataset_whose_two_variables_have_no_depth() -> None:
    northward_key = Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key()
    eastward_key = Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key()
    challenger_dataset = _depth_free_dataset(
        {
            northward_key: [0.0, 4.0, 100.0],
            eastward_key: [0.0, 4.0, 100.0],
        }
    )
    reference_dataset = xarray.zeros_like(challenger_dataset)

    table = rmsd(
        challenger_dataset=challenger_dataset,
        reference_dataset=reference_dataset,
        variables=[
            Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
            Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
        ],
        ocean_mask=_surface_dry_at_sixty_degrees_mask(),
    )

    assert sorted(table.index) == sorted([MERIDIONAL_GEOSTROPHIC_LABEL, ZONAL_GEOSTROPHIC_LABEL])
    for label in (MERIDIONAL_GEOSTROPHIC_LABEL, ZONAL_GEOSTROPHIC_LABEL):
        assert numpy.isfinite(table.loc[label, "Lead day 1"])
        assert table.loc[label, MISSING_COUNT_COLUMN] == 0
        assert table.loc[label, MISSING_FRACTION_COLUMN] == 0.0
