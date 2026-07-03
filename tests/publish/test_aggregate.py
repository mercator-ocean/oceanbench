# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Statistical sanity for the display-time aggregation library (contracts.md §3.4)."""

from pathlib import Path

import numpy
import pandas
import pytest

from oceanbench.publish.aggregate import aggregate_scores
from oceanbench.runner.parity import recombine_class4_over_starts

_GOLDEN = Path(__file__).resolve().parents[1] / "parity" / "golden_scores_main_1degree.parquet"


def _row(**overrides) -> dict:
    row = {
        "challenger": "model",
        "year": 2024,
        "region": "global",
        "metric": "rmsd",
        "reference": "glorys",
        "variable": "sea_surface_height_above_geoid",
        "depth": "surface",
        "lead_day": 1,
        "band": None,
        "polarity": None,
        "unit": "m",
        "n": None,
        "start_date": pandas.Timestamp("2024-01-03"),
        "value": 0.1,
    }
    row.update(overrides)
    return row


def _gridded_frame(challenger: str, values: list[float]) -> pandas.DataFrame:
    starts = pandas.date_range("2024-01-03", periods=len(values), freq="7D")
    return pandas.DataFrame(
        [_row(challenger=challenger, start_date=start, value=value) for start, value in zip(starts, values)]
    )


def _class4_frame(values_and_counts: list[tuple[float, int]]) -> pandas.DataFrame:
    starts = pandas.date_range("2024-01-03", periods=len(values_and_counts), freq="7D")
    return pandas.DataFrame(
        [
            _row(
                challenger="model",
                metric="class4_rmsd",
                reference="observations",
                start_date=start,
                value=value,
                n=count,
            )
            for start, (value, count) in zip(starts, values_and_counts)
        ]
    )


def test_confidence_interval_brackets_the_mean():
    frame = _gridded_frame("model", [0.10, 0.12, 0.11, 0.13, 0.09, 0.14, 0.10, 0.12])
    aggregated = aggregate_scores(frame, seed=1)
    (row,) = aggregated.to_dict(orient="records")
    assert row["ci_low"] <= row["mean"] <= row["ci_high"]
    assert row["n_starts"] == 8
    assert row["mean"] == pytest.approx(numpy.mean([0.10, 0.12, 0.11, 0.13, 0.09, 0.14, 0.10, 0.12]))


def test_single_start_gives_degenerate_interval():
    aggregated = aggregate_scores(_gridded_frame("model", [0.123]), seed=1)
    (row,) = aggregated.to_dict(orient="records")
    assert row["n_starts"] == 1
    assert row["ci_low"] == pytest.approx(row["mean"])
    assert row["ci_high"] == pytest.approx(row["mean"])
    assert row["mean"] == pytest.approx(0.123)


def test_class4_point_estimate_matches_pooled_recombination():
    frame = _class4_frame([(0.20, 100), (0.30, 400), (0.25, 250), (0.10, 50)])
    aggregated = aggregate_scores(frame, seed=1)
    (row,) = aggregated.to_dict(orient="records")

    pooled = recombine_class4_over_starts(frame[["value", "n"]].assign(**{"group": 0}), grouping_columns=["group"])
    assert row["mean"] == pytest.approx(float(pooled["value"].iloc[0]))
    # A plain mean of the per-start RMSDs is a different (wrong) number, so the recombination matters.
    assert row["mean"] != pytest.approx(float(frame["value"].mean()))


def test_class4_bootstrap_reproduces_pooled_on_the_full_resample():
    frame = _class4_frame([(0.20, 100), (0.30, 400), (0.25, 250), (0.10, 50)])
    aggregated = aggregate_scores(frame, seed=7)
    (row,) = aggregated.to_dict(orient="records")
    # Every bootstrap draw pools with the same n-weighting, so the pooled value stays inside the CI.
    assert row["ci_low"] <= row["mean"] <= row["ci_high"]


def test_skill_versus_itself_is_zero_with_a_tight_interval():
    frame = _gridded_frame("model", [0.10, 0.12, 0.11, 0.13, 0.09, 0.14, 0.10, 0.12])
    aggregated = aggregate_scores(frame, baseline_challenger="model", seed=3)
    (row,) = aggregated.to_dict(orient="records")
    assert row["skill_vs_model"] == pytest.approx(0.0)
    assert row["skill_ci_low"] == pytest.approx(0.0)
    assert row["skill_ci_high"] == pytest.approx(0.0)
    assert row["n_starts_paired"] == 8


def test_paired_skill_interval_is_narrower_than_the_unpaired_one():
    generator = numpy.random.default_rng(0)
    baseline_values = 0.30 + generator.normal(0.0, 0.05, size=40)
    model_values = baseline_values * 0.8 + generator.normal(0.0, 0.005, size=40)  # strongly correlated with baseline

    frame = pandas.concat(
        [_gridded_frame("baseline", list(baseline_values)), _gridded_frame("model", list(model_values))],
        ignore_index=True,
    )
    aggregated = aggregate_scores(frame, baseline_challenger="baseline", n_bootstrap=2000, seed=11)
    model_row = aggregated[aggregated["challenger"] == "model"].iloc[0]
    paired_width = model_row["skill_ci_high"] - model_row["skill_ci_low"]

    starts = numpy.arange(len(model_values))
    generator_unpaired = numpy.random.default_rng(11)
    unpaired_skill = []
    for _ in range(2000):
        model_mean = model_values[generator_unpaired.choice(starts, size=len(starts), replace=True)].mean()
        baseline_mean = baseline_values[generator_unpaired.choice(starts, size=len(starts), replace=True)].mean()
        unpaired_skill.append(1.0 - model_mean / baseline_mean)
    unpaired_width = numpy.subtract(*numpy.percentile(unpaired_skill, [97.5, 2.5]))

    assert paired_width < unpaired_width


def test_golden_parquet_smoke():
    scores = pandas.read_parquet(_GOLDEN)
    aggregated = aggregate_scores(scores, seed=5)

    assert (
        len(aggregated)
        == scores.groupby(
            ["challenger", "year", "region", "metric", "reference", "variable", "depth", "lead_day"],
            dropna=False,
        ).ngroups
    )
    assert (aggregated["ci_low"] <= aggregated["mean"] + 1e-9).all()
    assert (aggregated["mean"] <= aggregated["ci_high"] + 1e-9).all()
    assert (aggregated["n_starts"] == 52).all()
    assert aggregated["mean"].notna().all()
