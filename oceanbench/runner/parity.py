# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Compare a v2 runner parquet against the published v0.2.1 parity golden.

The runner emits per-start records; the golden holds the published aggregated
(mean-over-starts) values keyed by the legacy metric keys. This module averages
the runner records over start dates, maps the ``(metric, reference, variable)``
triple onto the legacy golden metric key, joins on the shared keys, and reports
the maximum absolute and relative differences per metric with a tolerance gate.

Pure pandas/numpy — no dependency on the metric machinery.
"""

from dataclasses import dataclass

import numpy
import pandas

_MIXED_LAYER_DEPTH_VARIABLE = "ocean_mixed_layer_thickness"
_GEOSTROPHIC_VARIABLES = frozenset(
    {
        "geostrophic_northward_sea_water_velocity",
        "geostrophic_eastward_sea_water_velocity",
    }
)

_GROUPING_COLUMNS = [
    "challenger",
    "region",
    "metric",
    "reference",
    "variable",
    "depth",
    "lead_day",
]

_JOIN_NULL_SENTINEL = "__parity_null__"


def _fill_join_nulls(frame: pandas.DataFrame, columns: list[str]) -> pandas.DataFrame:
    """Replace nulls in join-key columns with a sentinel so null keys match null keys."""
    filled = frame.copy()
    for column in columns:
        filled[column] = filled[column].astype(object).where(filled[column].notna(), _JOIN_NULL_SENTINEL)
    return filled


def golden_metric_key(metric: str, reference: str | None, variable: str | None) -> str | None:
    """Map a ``(metric, reference, variable)`` triple onto the legacy golden metric key."""
    if metric == "lagrangian_deviation_km":
        return f"lagrangian_{reference}"
    if metric == "class4_rmsd":
        return "rmsd_variables_observations"
    if metric == "rmsd":
        if variable == _MIXED_LAYER_DEPTH_VARIABLE:
            return f"rmsd_mld_{reference}"
        if variable in _GEOSTROPHIC_VARIABLES:
            return f"rmsd_geostrophic_{reference}"
        return f"rmsd_variables_{reference}"
    return None


def recombine_class4_over_starts(
    class4_frame: pandas.DataFrame,
    grouping_columns: list[str] = _GROUPING_COLUMNS,
) -> pandas.DataFrame:
    """N-weighted recombination of per-start Class-4 rows into the pooled-over-observations RMSD.

    The published Class-4 value pools every observation at a lead day into one RMSD; a
    plain mean of the per-start RMSDs would not reproduce it. Because each per-start
    ``value`` is ``sqrt(sum_of_squares_of_that_start / n)``, ``sqrt(sum(value ** 2 * n) /
    sum(n))`` over the starts recovers the pooled RMSD exactly. ``grouping_columns`` selects
    the keys the recombination pools within (defaulting to the parity key set); the downstream
    aggregation library passes its own identity keys so there is a single recombination.
    """
    contributions = class4_frame.assign(sum_of_squares=(class4_frame["value"] ** 2) * class4_frame["n"])
    pooled = (
        contributions.groupby(grouping_columns, dropna=False)
        .agg(
            sum_of_squares=("sum_of_squares", "sum"),
            sample_size=("n", "sum"),
        )
        .reset_index()
    )
    pooled["value"] = numpy.sqrt(pooled["sum_of_squares"] / pooled["sample_size"])
    return pooled[grouping_columns + ["value"]]


def aggregate_runner_scores(runner_scores: pandas.DataFrame) -> pandas.DataFrame:
    """Aggregate the per-start runner records over start dates, per metric key.

    Gridded and Lagrangian metrics aggregate as a plain mean over starts. Class-4 RMSD
    instead recombines with :func:`_recombine_class4_over_starts` because its published
    value is a single RMSD pooled over every observation, not a mean of per-start RMSDs.
    """
    frame = runner_scores.copy()
    frame["variable"] = frame["variable"].astype(object).where(frame["variable"].notna(), None)
    frame["depth"] = frame["depth"].astype(object).where(frame["depth"].notna(), None)
    is_class4 = frame["metric"] == "class4_rmsd"
    mean_aggregated = frame[~is_class4].groupby(_GROUPING_COLUMNS, dropna=False)["value"].mean().reset_index()
    parts = [mean_aggregated]
    if is_class4.any():
        parts.append(recombine_class4_over_starts(frame[is_class4]))
    aggregated = pandas.concat(parts, ignore_index=True)
    aggregated["golden_metric_key"] = aggregated.apply(
        lambda row: golden_metric_key(row["metric"], row["reference"], row["variable"]),
        axis=1,
    )
    return aggregated


def _normalise_golden(golden_scores: pandas.DataFrame) -> pandas.DataFrame:
    frame = golden_scores.copy()
    frame["variable"] = frame["variable_standard_name"].astype(object)
    frame.loc[frame["variable"].isin([""]), "variable"] = None
    frame["variable"] = frame["variable"].where(frame["variable"].notna(), None)
    frame["depth"] = frame["depth_label"].astype(object).str.lower()
    frame["depth"] = frame["depth"].where(frame["depth"].notna(), None)
    return frame[["challenger", "region", "metric_key", "variable", "depth", "lead_day", "value"]].rename(
        columns={"metric_key": "golden_metric_key", "value": "golden_value"}
    )


@dataclass(frozen=True)
class MetricComparison:
    golden_metric_key: str
    matched: int
    runner_only: int
    golden_only: int
    max_absolute_difference: float
    max_relative_difference: float


def compare(
    runner_scores: pandas.DataFrame,
    golden_scores: pandas.DataFrame,
    *,
    exclude_golden_metrics: tuple[str, ...] = (),
    relative_difference_floor: float = 1e-6,
) -> list[MetricComparison]:
    """Return one :class:`MetricComparison` per golden metric key present on either side."""
    runner_aggregated = aggregate_runner_scores(runner_scores)
    golden_normalised = _normalise_golden(golden_scores)

    join_keys = ["challenger", "region", "golden_metric_key", "variable", "depth", "lead_day"]
    nullable_keys = ["golden_metric_key", "variable", "depth"]
    merged = _fill_join_nulls(runner_aggregated, nullable_keys).merge(
        _fill_join_nulls(golden_normalised, nullable_keys),
        on=join_keys,
        how="outer",
        indicator=True,
    )

    comparisons = []
    for metric_key, group in merged.groupby("golden_metric_key", dropna=False):
        if metric_key in exclude_golden_metrics:
            continue
        matched = group[group["_merge"] == "both"]
        both_present = matched.dropna(subset=["value", "golden_value"])
        absolute_difference = (both_present["value"] - both_present["golden_value"]).abs()
        denominator = both_present["golden_value"].abs().clip(lower=relative_difference_floor)
        relative_difference = absolute_difference / denominator
        comparisons.append(
            MetricComparison(
                golden_metric_key=metric_key,
                matched=int(len(matched)),
                runner_only=int((group["_merge"] == "left_only").sum()),
                golden_only=int((group["_merge"] == "right_only").sum()),
                max_absolute_difference=float(absolute_difference.max()) if len(absolute_difference) else 0.0,
                max_relative_difference=float(relative_difference.max()) if len(relative_difference) else 0.0,
            )
        )
    return comparisons


def gate(
    comparisons: list[MetricComparison],
    *,
    absolute_tolerance: float = 1e-4,
    require_matches: bool = True,
) -> tuple[bool, list[str]]:
    """Return ``(passed, failures)``: every compared metric must be within ``absolute_tolerance``."""
    failures = []
    for comparison in comparisons:
        if require_matches and comparison.matched == 0:
            failures.append(f"{comparison.golden_metric_key}: no matched rows")
            continue
        if comparison.max_absolute_difference > absolute_tolerance:
            failures.append(
                f"{comparison.golden_metric_key}: max abs diff "
                f"{comparison.max_absolute_difference:.2e} > {absolute_tolerance:.0e}"
            )
    return (len(failures) == 0, failures)


def comparison_report(comparisons: list[MetricComparison]) -> str:
    """Render a fixed-width per-metric table (max abs/rel diff, matched/unmatched counts)."""
    header = f"{'metric':30s} {'matched':>8s} {'run_only':>8s} {'gold_only':>9s} {'max_abs':>10s} {'max_rel':>10s}"
    lines = [header, "-" * len(header)]
    for comparison in sorted(comparisons, key=lambda item: str(item.golden_metric_key)):
        lines.append(
            f"{str(comparison.golden_metric_key):30s} {comparison.matched:8d} {comparison.runner_only:8d} "
            f"{comparison.golden_only:9d} {comparison.max_absolute_difference:10.2e} "
            f"{comparison.max_relative_difference:10.2e}"
        )
    return "\n".join(lines)
