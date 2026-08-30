# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import math

import numpy
import pytest
import xarray

from oceanbench.core.curvilinear_grid import nearest_neighbour_mapping, sample_onto_target_grid
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.ensemble_gridded import (
    ENSEMBLE_DIMENSION,
    METRIC_CRPS_BIASED,
    METRIC_CRPS_FAIR,
    METRIC_ENSEMBLE_MEAN_RMSD,
    METRIC_ENSEMBLE_SPREAD,
    METRIC_MEMBER_RMSD,
    METRIC_SPREAD_ERROR_RATIO,
    _aggregate_metric_values,
    area_weighted_mean,
    continuous_ranked_probability_score,
    ensemble_field_statistics,
    ensemble_gridded_records,
    ensemble_spread,
    field_metric_values,
    finite_ensemble_correction,
)
from oceanbench.core.score_records import RunContext, records_to_dataframe

LATITUDE_KEY = Dimension.LATITUDE.key()
LONGITUDE_KEY = Dimension.LONGITUDE.key()


def _ensemble(values: numpy.ndarray) -> xarray.DataArray:
    return xarray.DataArray(
        values,
        dims=[ENSEMBLE_DIMENSION, LATITUDE_KEY, LONGITUDE_KEY],
        coords={
            ENSEMBLE_DIMENSION: numpy.arange(values.shape[0]),
            LATITUDE_KEY: numpy.linspace(-60, 60, values.shape[1]),
            LONGITUDE_KEY: numpy.linspace(-180, 170, values.shape[2]),
        },
    )


def _field(values: numpy.ndarray) -> xarray.DataArray:
    return xarray.DataArray(
        values,
        dims=[LATITUDE_KEY, LONGITUDE_KEY],
        coords={
            LATITUDE_KEY: numpy.linspace(-60, 60, values.shape[0]),
            LONGITUDE_KEY: numpy.linspace(-180, 170, values.shape[1]),
        },
    )


def _random_ensemble(member_count: int, seed: int = 0) -> tuple[xarray.DataArray, xarray.DataArray]:
    generator = numpy.random.default_rng(seed)
    truth = _field(generator.normal(size=(8, 12)))
    members = _ensemble(truth.values + generator.normal(scale=0.7, size=(member_count, 8, 12)))
    return members, truth


def test_single_member_crps_is_the_mean_absolute_error():
    members, truth = _random_ensemble(member_count=1)
    crps = continuous_ranked_probability_score(members, truth)
    expected = abs(members.isel({ENSEMBLE_DIMENSION: 0}) - truth)
    numpy.testing.assert_allclose(crps.values, expected.values)


def test_fair_crps_matches_the_explicit_double_sum():
    members, truth = _random_ensemble(member_count=6, seed=3)
    values = members.values
    member_count = values.shape[0]
    mean_absolute_error = numpy.abs(values - truth.values[None]).mean(axis=0)
    pairwise = numpy.abs(values[:, None] - values[None, :]).sum(axis=(0, 1))
    expected = mean_absolute_error - pairwise / (2 * member_count * (member_count - 1))
    numpy.testing.assert_allclose(continuous_ranked_probability_score(members, truth).values, expected, rtol=1e-12)


def test_biased_crps_is_above_the_fair_crps():
    members, truth = _random_ensemble(member_count=20, seed=5)
    fair = area_weighted_mean(continuous_ranked_probability_score(members, truth, fair=True))
    biased = area_weighted_mean(continuous_ranked_probability_score(members, truth, fair=False))
    assert fair < biased


def _perfect_model_ensemble(member_count: int, seed: int) -> tuple[xarray.DataArray, xarray.DataArray]:
    """A calibrated ensemble: the truth is one more draw from the same distribution.

    Every reliability property of the metrics is stated under exchangeability of the truth
    with the members, so this, and not an arbitrary spread, is the construction the sanity
    checks must use.
    """
    generator = numpy.random.default_rng(seed)
    background = generator.normal(size=(40, 90))
    draws = background[None] + generator.normal(scale=1.0, size=(member_count + 1, 40, 90))
    return _ensemble(draws[:member_count]), _field(draws[member_count])


def test_fair_crps_is_below_the_ensemble_mean_absolute_error():
    members, truth = _perfect_model_ensemble(member_count=30, seed=7)
    statistics = ensemble_field_statistics(members, truth)
    assert statistics.crps_fair < statistics.ensemble_mean_absolute_error


def test_ensemble_mean_beats_the_typical_single_member():
    members, truth = _random_ensemble(member_count=30, seed=11)
    statistics = ensemble_field_statistics(members, truth)
    assert statistics.ensemble_mean_squared_error < statistics.member_squared_error


def test_spread_error_ratio_is_one_for_a_perfect_model_ensemble():
    members, held_out = _perfect_model_ensemble(member_count=40, seed=42)
    statistics = ensemble_field_statistics(members, held_out)
    ratio = field_metric_values(statistics)[METRIC_SPREAD_ERROR_RATIO]
    assert ratio == pytest.approx(1.0, abs=0.03)


def test_ensemble_spread_carries_the_finite_size_correction():
    members, _truth = _random_ensemble(member_count=10, seed=13)
    member_count = members.sizes[ENSEMBLE_DIMENSION]
    expected = numpy.sqrt((member_count + 1) / member_count * members.var(dim=ENSEMBLE_DIMENSION, ddof=1))
    numpy.testing.assert_allclose(ensemble_spread(members).values, expected.values)


def test_area_weighted_mean_ignores_missing_cells():
    values = numpy.ones((8, 12))
    values[0, 0] = numpy.nan
    assert area_weighted_mean(_field(values)) == pytest.approx(1.0)


def test_area_weighted_mean_weights_by_cosine_latitude():
    latitudes = numpy.array([0.0, 60.0])
    field = xarray.DataArray(
        numpy.array([[1.0], [3.0]]),
        dims=[LATITUDE_KEY, LONGITUDE_KEY],
        coords={LATITUDE_KEY: latitudes, LONGITUDE_KEY: [0.0]},
    )
    weights = numpy.cos(numpy.deg2rad(latitudes))
    assert area_weighted_mean(field) == pytest.approx(float((weights * [1.0, 3.0]).sum() / weights.sum()))


def test_records_carry_every_metric_per_lead_day_and_an_aggregate():
    members, truth = _random_ensemble(member_count=12, seed=17)
    variable = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
    statistics = {
        (start_date, lead_day, variable): ensemble_field_statistics(members, truth)
        for start_date in ("2024-01-04", "2024-01-11")
        for lead_day in (1, 2)
    }
    records = ensemble_gridded_records(
        statistics,
        context=RunContext(
            challenger="gloens", challenger_version="test", year=2024, region="global", oceanbench_version="0.0.0"
        ),
        reference="glorys",
    )
    dataframe = records_to_dataframe(records)
    expected_metrics = {
        METRIC_CRPS_FAIR,
        METRIC_CRPS_BIASED,
        METRIC_ENSEMBLE_MEAN_RMSD,
        METRIC_MEMBER_RMSD,
        METRIC_ENSEMBLE_SPREAD,
        METRIC_SPREAD_ERROR_RATIO,
    }
    assert set(dataframe["metric"]) == expected_metrics
    assert len(dataframe) == len(expected_metrics) * (4 + 2)
    assert dataframe["start_date"].isna().sum() == len(expected_metrics) * 2
    assert set(dataframe["reference"]) == {"glorys"}
    assert set(dataframe[dataframe["metric"] == METRIC_SPREAD_ERROR_RATIO]["unit"]) == {"1"}


def test_records_are_ordered_by_lead_day_as_a_number():
    members, truth = _random_ensemble(member_count=6, seed=23)
    variable = Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key()
    lead_days = (1, 2, 10)
    statistics = {
        ("2024-01-04", lead_day, variable): ensemble_field_statistics(members, truth) for lead_day in lead_days
    }
    records = ensemble_gridded_records(
        statistics,
        context=RunContext(
            challenger="gloens", challenger_version="test", year=2024, region="global", oceanbench_version="0.0.0"
        ),
        reference="glorys",
    )
    dataframe = records_to_dataframe(records)
    per_start = dataframe[dataframe["start_date"].notna()]

    assert list(dict.fromkeys(per_start["lead_day"])) == list(lead_days)


def test_nearest_neighbour_mapping_drops_land_and_out_of_range_cells():
    source_latitude, source_longitude = numpy.meshgrid(
        numpy.arange(-10.0, 10.5, 0.5), numpy.arange(-10.0, 10.5, 0.5), indexing="ij"
    )
    ocean_mask = numpy.ones(source_latitude.shape, dtype=bool)
    ocean_mask[0, :] = False
    target_latitude = numpy.arange(-12.0, 12.5, 0.5)
    target_longitude = numpy.arange(-10.0, 10.5, 0.5)
    mapping = nearest_neighbour_mapping(
        source_latitude,
        source_longitude,
        ocean_mask,
        target_latitude,
        target_longitude,
        maximum_distance_kilometres=60.0,
    )
    assert not mapping.usable[0].any()
    assert not mapping.usable[-1].any()
    assert mapping.usable[len(target_latitude) // 2].all()


def test_sample_onto_target_grid_returns_the_nearest_source_value():
    source_latitude, source_longitude = numpy.meshgrid(
        numpy.arange(-10.0, 10.5, 0.5), numpy.arange(-10.0, 10.5, 0.5), indexing="ij"
    )
    ocean_mask = numpy.ones(source_latitude.shape, dtype=bool)
    mapping = nearest_neighbour_mapping(
        source_latitude,
        source_longitude,
        ocean_mask,
        numpy.arange(-9.0, 9.5, 0.5),
        numpy.arange(-9.0, 9.5, 0.5),
        maximum_distance_kilometres=60.0,
    )
    sampled = sample_onto_target_grid(source_latitude, mapping)
    expected_latitude, _ = numpy.meshgrid(numpy.arange(-9.0, 9.5, 0.5), numpy.arange(-9.0, 9.5, 0.5), indexing="ij")
    numpy.testing.assert_allclose(sampled.values, expected_latitude)


def test_the_finite_ensemble_correction_is_the_factor_that_makes_a_small_ensemble_reliable() -> None:
    assert finite_ensemble_correction(2) == 1.5
    assert finite_ensemble_correction(50) == 1.02
    assert finite_ensemble_correction(1000000) == pytest.approx(1.0, abs=1e-5)

    # A perfectly reliable ensemble draws its members and the truth from one distribution, so the
    # corrected spread equals the ensemble mean error however few members it carries. Without the
    # correction a small ensemble reads as under dispersive when it is not.
    generator = numpy.random.default_rng(20260830)
    for member_count in (3, 8, 50):
        members = generator.standard_normal((20000, member_count))
        truth = generator.standard_normal(20000)
        variance = members.var(axis=1, ddof=1).mean()
        squared_error = ((truth - members.mean(axis=1)) ** 2).mean()
        corrected = finite_ensemble_correction(member_count) * variance
        assert corrected / squared_error == pytest.approx(1.0, abs=0.05)
        assert variance / squared_error < 0.99


def test_aggregating_a_lead_day_averages_the_roots_and_rebuilds_the_ratio() -> None:
    first = {
        METRIC_CRPS_FAIR: 0.2,
        METRIC_ENSEMBLE_MEAN_RMSD: 1.0,
        METRIC_ENSEMBLE_SPREAD: 1.0,
        METRIC_SPREAD_ERROR_RATIO: 1.0,
    }
    second = {
        METRIC_CRPS_FAIR: 0.4,
        METRIC_ENSEMBLE_MEAN_RMSD: 0.01,
        METRIC_ENSEMBLE_SPREAD: 0.5,
        METRIC_SPREAD_ERROR_RATIO: 50.0,
    }

    aggregated = _aggregate_metric_values([first, second])

    assert aggregated[METRIC_CRPS_FAIR] == pytest.approx(0.3)
    assert aggregated[METRIC_ENSEMBLE_MEAN_RMSD] == pytest.approx(0.505)
    # The one start whose error nearly vanished carries a ratio of 50, which a plain mean would
    # turn into a year of 25.5. The ratio is rebuilt from the averaged spread and error instead.
    assert aggregated[METRIC_SPREAD_ERROR_RATIO] == pytest.approx(0.75 / 0.505)


def test_aggregating_a_lead_day_propagates_a_missing_start_rather_than_dropping_it() -> None:
    present = {
        METRIC_CRPS_FAIR: 0.2,
        METRIC_ENSEMBLE_MEAN_RMSD: 1.0,
        METRIC_ENSEMBLE_SPREAD: 1.0,
        METRIC_SPREAD_ERROR_RATIO: 1.0,
    }
    missing = {**present, METRIC_CRPS_FAIR: float("nan")}

    aggregated = _aggregate_metric_values([present, missing])

    assert math.isnan(aggregated[METRIC_CRPS_FAIR])
    assert aggregated[METRIC_ENSEMBLE_MEAN_RMSD] == pytest.approx(1.0)


def test_a_lead_day_whose_error_vanished_carries_no_ratio() -> None:
    values = {
        METRIC_CRPS_FAIR: 0.0,
        METRIC_ENSEMBLE_MEAN_RMSD: 0.0,
        METRIC_ENSEMBLE_SPREAD: 0.0,
        METRIC_SPREAD_ERROR_RATIO: float("nan"),
    }

    assert math.isnan(_aggregate_metric_values([values])[METRIC_SPREAD_ERROR_RATIO])


def test_the_spread_is_averaged_over_the_cells_the_error_is_averaged_over() -> None:
    generator = numpy.random.default_rng(7)
    truth = _field(generator.normal(size=(8, 12)))
    members = _ensemble(truth.values + generator.normal(scale=0.7, size=(10, 8, 12)))
    # A band the reference does not cover, an ice edge or an unusable target cell, where the
    # members happen to disagree far more than anywhere else.
    wide = members.values.copy()
    wide[:, :2, :] = truth.values[None, :2, :] + generator.normal(scale=8.0, size=(10, 2, 12))
    members = _ensemble(wide)
    uncovered = truth.values.copy()
    uncovered[:2, :] = numpy.nan
    reference = _field(uncovered)

    statistics = ensemble_field_statistics(members, reference)
    scored_only = ensemble_field_statistics(
        members.isel({LATITUDE_KEY: slice(2, None)}), reference.isel({LATITUDE_KEY: slice(2, None)})
    )

    assert statistics.scored_cell_count == 6 * 12
    assert statistics.ensemble_variance == pytest.approx(scored_only.ensemble_variance)
    assert statistics.crps_fair == pytest.approx(scored_only.crps_fair)
    assert statistics.ensemble_mean_squared_error == pytest.approx(scored_only.ensemble_mean_squared_error)
    # The uncovered band alone would have multiplied the spread of the field it is not part of.
    assert area_weighted_mean(members.var(dim=ENSEMBLE_DIMENSION, ddof=1)) > 3 * statistics.ensemble_variance
