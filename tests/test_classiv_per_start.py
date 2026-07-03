# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Prove the per-start Class-4 emission recombines into the exact pooled RMSD.

The published Class-4 value is a single RMSD pooled over every observation at a
given (variable, depth_bin, lead_day). The runner instead emits one row per
forecast start (``first_day``); these tests prove that the n-weighted
recombination ``sqrt(sum(rmsd**2 * count) / sum(count))`` reproduces the pooled
value bit-for-bit within float tolerance, including uneven observation counts
per start and starts that contribute no observations to some bins.
"""

import numpy
import pandas
import pytest

from oceanbench.core.classIV_support import (
    compute_class4_rmsd_table,
    compute_class4_rmsd_table_per_start,
    recombine_class4_pooled_from_per_start,
)
from oceanbench.runner import parity, records


def _matchups(rows: list[tuple[str, str, int, float, float]]) -> pandas.DataFrame:
    """Build a synthetic match-up frame from (first_day, depth_bin, lead_day, model, obs) tuples."""
    return pandas.DataFrame(
        rows,
        columns=["first_day", "depth_bin", "lead_day", "model_value", "observation_value"],
    )


def _random_matchups(seed: int) -> pandas.DataFrame:
    generator = numpy.random.default_rng(seed)
    first_days = ["2024-01-03", "2024-01-10", "2024-01-17", "2024-01-24"]
    depth_bins = ["surface", "0-5m", "100-300m"]
    lead_days = [0, 1, 4, 9]
    rows = []
    for first_day in first_days:
        for depth_bin in depth_bins:
            for lead_day in lead_days:
                # Uneven observation counts per (start, bin, lead), including zero.
                observation_count = int(generator.integers(0, 7))
                for _ in range(observation_count):
                    observation_value = float(generator.normal())
                    model_value = observation_value + float(generator.normal(scale=0.3))
                    rows.append((first_day, depth_bin, lead_day, model_value, observation_value))
    return _matchups(rows)


def _assert_pooled_matches_recombined(matchups: pandas.DataFrame) -> pandas.DataFrame:
    pooled = compute_class4_rmsd_table(matchups, "sea_water_potential_temperature")
    per_start = compute_class4_rmsd_table_per_start(matchups, "sea_water_potential_temperature")
    recombined = recombine_class4_pooled_from_per_start(per_start)

    keys = ["variable", "depth_bin", "lead_day"]
    merged = pooled.sort_values(keys).merge(recombined.sort_values(keys), on=keys, suffixes=("_pooled", "_recombined"))
    assert len(merged) == len(pooled)
    assert len(merged) == len(recombined)
    numpy.testing.assert_allclose(
        merged["rmsd_recombined"].to_numpy(),
        merged["rmsd_pooled"].to_numpy(),
        rtol=0.0,
        atol=1e-12,
    )
    assert (merged["count_pooled"] == merged["count_recombined"]).all()
    return merged


def test_recombination_equals_pooled_uneven_counts_and_zero_obs_bins():
    for seed in range(8):
        merged = _assert_pooled_matches_recombined(_random_matchups(seed))
        assert len(merged) > 0


def test_recombination_hand_worked_example():
    # Two starts, one bin/lead. Start A: errors [3, 4] -> rmsd sqrt(12.5), n=2.
    # Start B: error [0] -> rmsd 0, n=1. Pooled over [3,4,0]: sqrt(25/3).
    matchups = _matchups(
        [
            ("2024-01-03", "surface", 0, 3.0, 0.0),
            ("2024-01-03", "surface", 0, 4.0, 0.0),
            ("2024-01-10", "surface", 0, 0.0, 0.0),
        ]
    )
    per_start = compute_class4_rmsd_table_per_start(matchups, "sea_surface_height_above_geoid")
    per_start_by_day = per_start.set_index("first_day")["rmsd"]
    assert per_start_by_day.loc["2024-01-03"] == pytest.approx(numpy.sqrt(12.5))
    assert per_start_by_day.loc["2024-01-10"] == pytest.approx(0.0)

    recombined = recombine_class4_pooled_from_per_start(per_start)
    assert recombined["rmsd"].iloc[0] == pytest.approx(numpy.sqrt(25.0 / 3.0))
    assert int(recombined["count"].iloc[0]) == 3


def test_start_missing_from_a_bin_does_not_bias_recombination():
    # The "100-300m" bin only has observations on the first start; the second start
    # contributes zero observations there. The pooled RMSD must equal the first
    # start's RMSD, and recombination must reproduce it.
    matchups = _matchups(
        [
            ("2024-01-03", "100-300m", 4, 2.0, 0.0),
            ("2024-01-03", "100-300m", 4, 2.0, 0.0),
            ("2024-01-10", "surface", 4, 1.0, 0.0),
        ]
    )
    per_start = compute_class4_rmsd_table_per_start(matchups, "sea_water_salinity")
    deep = per_start[per_start["depth_bin"] == "100-300m"]
    assert len(deep) == 1  # only the first start contributed
    _assert_pooled_matches_recombined(matchups)


def _context() -> records.RunContext:
    return records.RunContext(
        challenger="glonet_1_degree",
        challenger_version="0.2.1",
        year=2024,
        region="global",
        oceanbench_version="0.2.1",
    )


def test_class4_per_start_records_carry_n_lead_day_and_unit():
    per_start = pandas.DataFrame(
        {
            "variable": ["sea_surface_height_above_geoid", "sea_water_potential_temperature"],
            "first_day": ["2024-01-03", "2024-01-03"],
            "depth_bin": ["surface", "0-5m"],
            "lead_day": [0, 9],  # 0-based in the table
            "rmsd": [0.05, 0.4],
            "count": [12, 30],
        }
    )
    emitted = records.class4_per_start_records(per_start, context=_context())
    assert all(record["metric"] == "class4_rmsd" and record["reference"] == "observations" for record in emitted)
    surface = next(record for record in emitted if record["depth"] == "surface")
    assert surface["lead_day"] == 1  # 0-based table lead day 0 -> 1-based artifact lead day 1
    assert surface["unit"] == "m"
    assert surface["n"] == 12
    assert surface["value"] == pytest.approx(0.05)
    deep = next(record for record in emitted if record["depth"] == "0-5m")
    assert deep["lead_day"] == 10
    assert deep["unit"] == "°C"


def test_parity_aggregation_recombines_class4_not_mean():
    # Per-start class4 rows whose n-weighted recombination differs from a plain mean.
    # Start A rmsd 3 (n=2), start B rmsd 0 (n=1) -> pooled sqrt((9*2 + 0)/3) = sqrt(6),
    # which is not the plain mean of the per-start RMSDs (1.5).
    def _row(**overrides):
        row = {
            "challenger": "glonet_1_degree",
            "challenger_version": "0.2.1",
            "year": 2024,
            "region": "global",
            "metric": "class4_rmsd",
            "reference": "observations",
            "variable": "sea_surface_height_above_geoid",
            "depth": "surface",
            "lead_day": 1,
            "start_date": None,
            "band": None,
            "polarity": None,
            "value": 0.0,
            "unit": "m",
            "n": 1,
            "oceanbench_version": "0.2.1",
        }
        row.update(overrides)
        return row

    runner = pandas.DataFrame(
        [
            _row(start_date="2024-01-03", value=3.0, n=2),
            _row(start_date="2024-01-10", value=0.0, n=1),
        ]
    )
    aggregated = parity.aggregate_runner_scores(runner)
    assert len(aggregated) == 1
    assert aggregated["value"].iloc[0] == pytest.approx(numpy.sqrt(6.0))
    assert aggregated["value"].iloc[0] != pytest.approx(1.5)
    assert aggregated["golden_metric_key"].iloc[0] == "rmsd_variables_observations"
