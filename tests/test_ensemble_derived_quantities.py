# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import pytest
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents, compute_mixed_layer_depth
from oceanbench.core.ensemble_derived_quantities import (
    DERIVED_VARIABLE_KEYS,
    derived_quantity_statistics,
    per_member_geostrophic_currents,
    per_member_mixed_layer_depth,
    reference_geostrophic_currents,
    reference_mixed_layer_depth,
)
from oceanbench.core.ensemble_gridded import ENSEMBLE_DIMENSION, ensemble_field_statistics

DEPTHS = [0.5, 10.0, 47.0, 100.0]
LATITUDES = [-20.0, -10.0, 10.0, 20.0]
LONGITUDES = [0.0, 1.0, 2.0]
FIRST_DAYS = [numpy.datetime64("2024-01-03")]
LEAD_DAYS = [0, 1]

MIXED_LAYER_DEPTH_KEY = Variable.MIXED_LAYER_DEPTH.key()
SEA_SURFACE_HEIGHT_KEY = Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key()
GEOSTROPHIC_EASTWARD_KEY = Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY.key()
GEOSTROPHIC_NORTHWARD_KEY = Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY.key()


def _profile_coordinates() -> dict:
    return {
        Dimension.FIRST_DAY_DATETIME.key(): FIRST_DAYS,
        Dimension.LEAD_DAY_INDEX.key(): LEAD_DAYS,
        Dimension.DEPTH.key(): DEPTHS,
        Dimension.LATITUDE.key(): LATITUDES,
        Dimension.LONGITUDE.key(): LONGITUDES,
    }


def _profile_dimensions() -> list[str]:
    return [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.DEPTH.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]


def _profile_dataset(member_count: int, seed: int = 0) -> xarray.Dataset:
    """Temperature and salinity profiles that stratify with depth, one set per member."""
    generator = numpy.random.default_rng(seed)
    shape = (member_count, len(FIRST_DAYS), len(LEAD_DAYS), len(DEPTHS), len(LATITUDES), len(LONGITUDES))
    stratification = numpy.array([0.0, -0.2, -1.5, -4.0]).reshape(1, 1, 1, len(DEPTHS), 1, 1)
    temperature = 18.0 + stratification + generator.normal(scale=0.05, size=shape)
    salinity = 35.0 + generator.normal(scale=0.02, size=shape)
    dimensions = [ENSEMBLE_DIMENSION] + _profile_dimensions()
    return xarray.Dataset(
        data_vars={
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (dimensions, temperature),
            Variable.SEA_WATER_SALINITY.key(): (dimensions, salinity),
        },
        coords={ENSEMBLE_DIMENSION: numpy.arange(member_count), **_profile_coordinates()},
    )


def _sea_surface_height_dataset(member_count: int, seed: int = 1) -> xarray.Dataset:
    generator = numpy.random.default_rng(seed)
    shape = (member_count, len(FIRST_DAYS), len(LEAD_DAYS), len(LATITUDES), len(LONGITUDES))
    dimensions = [
        ENSEMBLE_DIMENSION,
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    return xarray.Dataset(
        data_vars={SEA_SURFACE_HEIGHT_KEY: (dimensions, generator.normal(scale=0.2, size=shape))},
        coords={
            ENSEMBLE_DIMENSION: numpy.arange(member_count),
            Dimension.FIRST_DAY_DATETIME.key(): FIRST_DAYS,
            Dimension.LEAD_DAY_INDEX.key(): LEAD_DAYS,
            Dimension.LATITUDE.key(): LATITUDES,
            Dimension.LONGITUDE.key(): LONGITUDES,
        },
    )


def test_geostrophic_kernel_does_not_accept_a_member_dimension():
    with pytest.raises(ValueError):
        compute_geostrophic_currents(_sea_surface_height_dataset(member_count=3))


def test_per_member_mixed_layer_depth_matches_the_kernel_on_every_member():
    dataset = _profile_dataset(member_count=4)

    derived = per_member_mixed_layer_depth(dataset)

    assert derived.sizes[ENSEMBLE_DIMENSION] == 4
    for member_index in range(4):
        expected = compute_mixed_layer_depth(dataset.isel({ENSEMBLE_DIMENSION: member_index}))
        numpy.testing.assert_allclose(
            derived[MIXED_LAYER_DEPTH_KEY].isel({ENSEMBLE_DIMENSION: member_index}).values,
            expected[MIXED_LAYER_DEPTH_KEY].values,
        )


def test_per_member_mixed_layer_depth_keeps_the_other_dimensions():
    derived = per_member_mixed_layer_depth(_profile_dataset(member_count=2))

    assert derived[MIXED_LAYER_DEPTH_KEY].sizes == {
        ENSEMBLE_DIMENSION: 2,
        Dimension.FIRST_DAY_DATETIME.key(): len(FIRST_DAYS),
        Dimension.LEAD_DAY_INDEX.key(): len(LEAD_DAYS),
        Dimension.LATITUDE.key(): len(LATITUDES),
        Dimension.LONGITUDE.key(): len(LONGITUDES),
    }


def test_per_member_mixed_layer_depth_differs_from_the_ensemble_mean_derivation():
    dataset = _profile_dataset(member_count=6, seed=7)

    per_member = per_member_mixed_layer_depth(dataset)[MIXED_LAYER_DEPTH_KEY]

    assert per_member.std(dim=ENSEMBLE_DIMENSION).max().values > 0


def test_per_member_geostrophic_currents_match_the_kernel_on_every_member():
    dataset = _sea_surface_height_dataset(member_count=3)

    derived = per_member_geostrophic_currents(dataset)

    assert derived.sizes[ENSEMBLE_DIMENSION] == 3
    for member_index in range(3):
        expected = compute_geostrophic_currents(dataset.isel({ENSEMBLE_DIMENSION: member_index}))
        for variable_key in (GEOSTROPHIC_EASTWARD_KEY, GEOSTROPHIC_NORTHWARD_KEY):
            numpy.testing.assert_array_equal(
                derived[variable_key].isel({ENSEMBLE_DIMENSION: member_index}).values,
                expected[variable_key].values,
            )


def test_single_member_geostrophic_currents_match_the_deterministic_kernel():
    dataset = _sea_surface_height_dataset(member_count=1)

    derived = per_member_geostrophic_currents(dataset)
    expected = compute_geostrophic_currents(dataset.isel({ENSEMBLE_DIMENSION: 0}))

    assert derived.sizes[ENSEMBLE_DIMENSION] == 1
    for variable_key in (GEOSTROPHIC_EASTWARD_KEY, GEOSTROPHIC_NORTHWARD_KEY):
        numpy.testing.assert_array_equal(
            derived[variable_key].isel({ENSEMBLE_DIMENSION: 0}).values,
            expected[variable_key].values,
        )


def test_geostrophic_currents_keep_the_equator_exclusion_of_the_kernel():
    latitude_key = Dimension.LATITUDE.key()
    derived = per_member_geostrophic_currents(_sea_surface_height_dataset(member_count=2))

    assert numpy.all(numpy.abs(derived[latitude_key].values) > 0.5)


def test_derived_quantities_are_scored_by_the_ensemble_gridded_metrics():
    dataset = _sea_surface_height_dataset(member_count=5, seed=11)
    reference_dataset = _sea_surface_height_dataset(member_count=1, seed=99).isel({ENSEMBLE_DIMENSION: 0}, drop=True)
    field_selection = {Dimension.FIRST_DAY_DATETIME.key(): 0, Dimension.LEAD_DAY_INDEX.key(): 0}

    per_member = per_member_geostrophic_currents(dataset).isel(field_selection)
    reference = reference_geostrophic_currents(reference_dataset).isel(field_selection)
    statistics = derived_quantity_statistics(per_member, reference)

    assert set(statistics) == {GEOSTROPHIC_EASTWARD_KEY, GEOSTROPHIC_NORTHWARD_KEY}
    assert set(statistics).issubset(DERIVED_VARIABLE_KEYS)
    expected = ensemble_field_statistics(per_member[GEOSTROPHIC_EASTWARD_KEY], reference[GEOSTROPHIC_EASTWARD_KEY])
    assert statistics[GEOSTROPHIC_EASTWARD_KEY] == expected


def test_reference_mixed_layer_depth_carries_no_member_dimension():
    dataset = _profile_dataset(member_count=2).isel({ENSEMBLE_DIMENSION: 0}, drop=True)

    derived = reference_mixed_layer_depth(dataset)

    assert ENSEMBLE_DIMENSION not in derived.dims
