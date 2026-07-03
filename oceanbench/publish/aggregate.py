# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Derive display statistics from the per-start score records (contracts.md §3.4).

``scores.parquet`` stores one value per forecast start; means, bootstrap confidence
intervals and skill-vs-baseline are *derived at aggregation/display time* here, never
stored per run. This keeps every aggregate recomputable against any baseline.

Three derivations, all over the forecast-start axis (the 52 weekly starts):

- **mean over starts** — a plain mean for gridded/Lagrangian metrics; for Class-4
  RMSD the n-weighted recombination ``sqrt(sum(value ** 2 * n) / sum(n))`` (the single
  pooled-over-observations RMSD), reusing
  :func:`oceanbench.runner.parity.recombine_class4_over_starts`.
- **bootstrap 95% CI** — resample the starts with replacement, ``n_bootstrap`` draws
  under a fixed seed, percentile interval. Class-4 carries each start's ``(value, n)``
  pair through the recombination inside every draw.
- **skill vs a named baseline** — ``1 - value_model / value_baseline`` per metric key,
  computed on the *same* starts (paired). One resample of the starts is drawn per bootstrap
  draw and applied to both the model and the baseline, so their correlation across starts
  narrows the interval.

The output is one tidy row per (challenger, year, region, metric key, lead_day).
"""

from dataclasses import dataclass

import numpy
import pandas

from oceanbench.runner.parity import recombine_class4_over_starts

CLASS4_METRIC = "class4_rmsd"

METRIC_KEY_COLUMNS = ["metric", "reference", "variable", "depth", "lead_day", "band", "polarity"]
IDENTITY_COLUMNS = ["challenger", "year", "region"] + METRIC_KEY_COLUMNS
_CARRIED_COLUMNS = ["unit"]

DEFAULT_BOOTSTRAP_DRAWS = 1000
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 20240703

_NULL_SENTINEL = "__aggregate_null__"
_NULLABLE_KEY_COLUMNS = ["reference", "variable", "depth", "band", "polarity"]


@dataclass(frozen=True)
class _StartAlignedGroup:
    """A single identity group's per-start values aligned to the global start axis."""

    values: numpy.ndarray
    counts: numpy.ndarray
    present: numpy.ndarray
    is_class4: bool


def _fill_key_nulls(frame: pandas.DataFrame, columns: list[str]) -> pandas.DataFrame:
    filled = frame.copy()
    for column in columns:
        filled[column] = filled[column].astype(object).where(filled[column].notna(), _NULL_SENTINEL)
    return filled


def _pooled_root_mean_square(values: numpy.ndarray, counts: numpy.ndarray, axis: int) -> numpy.ndarray:
    """N-weighted pooled RMSD along ``axis``, ignoring absent (NaN-value) starts.

    Same formula as :func:`recombine_class4_over_starts`, vectorised over bootstrap draws:
    absent starts are given zero weight so ``sqrt(sum(value**2 * n) / sum(n))`` pools only
    the starts actually present in the draw.
    """
    absent = numpy.isnan(values)
    weighted_values = numpy.where(absent, 0.0, values)
    weights = numpy.where(absent, 0.0, counts)
    weight_total = weights.sum(axis=axis)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        pooled = numpy.sqrt((weighted_values**2 * weights).sum(axis=axis) / weight_total)
    return numpy.where(weight_total > 0, pooled, numpy.nan)


def _mean_over_starts(values: numpy.ndarray, axis: int) -> numpy.ndarray:
    """Plain mean along ``axis`` over the present (non-NaN) starts."""
    absent = numpy.isnan(values)
    present_count = (~absent).sum(axis=axis)
    summed = numpy.where(absent, 0.0, values).sum(axis=axis)
    with numpy.errstate(invalid="ignore", divide="ignore"):
        mean = summed / present_count
    return numpy.where(present_count > 0, mean, numpy.nan)


def _bootstrap_aggregate(values: numpy.ndarray, counts: numpy.ndarray, is_class4: bool) -> numpy.ndarray:
    """Aggregate every bootstrap draw (rows of ``values``) at once, vectorised over draws."""
    if is_class4:
        return _pooled_root_mean_square(values, counts, axis=1)
    return _mean_over_starts(values, axis=1)


def _point_aggregate(values: numpy.ndarray, counts: numpy.ndarray, present: numpy.ndarray, is_class4: bool) -> float:
    """Aggregate the present starts into the single point estimate.

    Class-4 delegates to :func:`recombine_class4_over_starts` so the pooled RMSD comes from the
    exact same recombination the scoring/parity pipeline uses — one source of truth for the value,
    while the bootstrap inner loop uses the vectorised form for speed.
    """
    if not is_class4:
        return float(_mean_over_starts(values[present], axis=0))
    present_frame = pandas.DataFrame({"value": values[present], "n": counts[present], "__pool__": 0})
    pooled = recombine_class4_over_starts(present_frame, grouping_columns=["__pool__"])
    return float(pooled["value"].iloc[0])


def _bootstrap_start_indices(start_count: int, draws: int, seed: int) -> numpy.ndarray:
    """A ``(draws, start_count)`` matrix resampling the start axis with replacement."""
    generator = numpy.random.default_rng(seed)
    return generator.integers(0, start_count, size=(draws, start_count))


def _confidence_interval(bootstrap_values: numpy.ndarray, confidence: float) -> tuple[float, float]:
    finite = bootstrap_values[numpy.isfinite(bootstrap_values)]
    if finite.size == 0:
        return (float("nan"), float("nan"))
    lower_percentile = (1.0 - confidence) / 2.0 * 100.0
    upper_percentile = (1.0 + confidence) / 2.0 * 100.0
    low, high = numpy.percentile(finite, [lower_percentile, upper_percentile])
    return (float(low), float(high))


def _align_group_to_starts(
    group: pandas.DataFrame,
    start_position: dict,
    start_count: int,
    is_class4: bool,
) -> _StartAlignedGroup:
    values = numpy.full(start_count, numpy.nan)
    counts = numpy.zeros(start_count)
    present = numpy.zeros(start_count, dtype=bool)
    positions = group["start_date"].map(start_position).to_numpy()
    values[positions] = group["value"].to_numpy(dtype=float)
    present[positions] = True
    if is_class4:
        counts[positions] = group["n"].to_numpy(dtype=float)
    return _StartAlignedGroup(values=values, counts=counts, present=present, is_class4=is_class4)


def _build_aligned_groups(
    scores: pandas.DataFrame, start_position: dict, start_count: int
) -> tuple[dict[tuple, _StartAlignedGroup], dict[tuple, dict]]:
    aligned: dict[tuple, _StartAlignedGroup] = {}
    carried: dict[tuple, dict] = {}
    for identity, group in scores.groupby(IDENTITY_COLUMNS, dropna=False, sort=False):
        is_class4 = group["metric"].iloc[0] == CLASS4_METRIC
        aligned[identity] = _align_group_to_starts(group, start_position, start_count, is_class4)
        carried[identity] = {column: group[column].iloc[0] for column in _CARRIED_COLUMNS}
    return aligned, carried


def _summarise_group(group: _StartAlignedGroup, bootstrap_indices: numpy.ndarray, confidence: float) -> dict:
    mean = _point_aggregate(group.values, group.counts, group.present, group.is_class4)
    bootstrap = _bootstrap_aggregate(group.values[bootstrap_indices], group.counts[bootstrap_indices], group.is_class4)
    ci_low, ci_high = _confidence_interval(bootstrap, confidence)
    return {
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_starts": int(group.present.sum()),
    }


def _skill_for_pair(
    model: _StartAlignedGroup,
    baseline: _StartAlignedGroup,
    bootstrap_indices: numpy.ndarray,
    confidence: float,
) -> dict:
    """Paired skill ``1 - model / baseline`` over the starts common to both, with paired CI."""
    common = model.present & baseline.present
    model_values = numpy.where(common, model.values, numpy.nan)
    baseline_values = numpy.where(common, baseline.values, numpy.nan)
    model_point = _point_aggregate(model_values, model.counts, common, model.is_class4)
    baseline_point = _point_aggregate(baseline_values, baseline.counts, common, baseline.is_class4)
    skill_point = float(1.0 - model_point / baseline_point)

    model_boot = _bootstrap_aggregate(model_values[bootstrap_indices], model.counts[bootstrap_indices], model.is_class4)
    baseline_boot = _bootstrap_aggregate(
        baseline_values[bootstrap_indices], baseline.counts[bootstrap_indices], baseline.is_class4
    )
    with numpy.errstate(invalid="ignore", divide="ignore"):
        skill_boot = 1.0 - model_boot / baseline_boot
    ci_low, ci_high = _confidence_interval(skill_boot, confidence)
    return {
        "skill": skill_point,
        "skill_ci_low": ci_low,
        "skill_ci_high": ci_high,
        "n_starts_paired": int(common.sum()),
    }


def aggregate_scores(
    scores: pandas.DataFrame,
    *,
    baseline_challenger: str | None = None,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_DRAWS,
    seed: int = DEFAULT_SEED,
    confidence: float = DEFAULT_CONFIDENCE,
) -> pandas.DataFrame:
    """Aggregate per-start score records into a tidy mean/CI (and optional skill) table.

    One row per (challenger, year, region, metric key, lead_day). ``baseline_challenger``,
    when given, adds paired skill-vs-baseline columns for every challenger that shares a
    metric key with the baseline (including the baseline against itself, which is exactly 0).
    """
    if scores.empty:
        return pandas.DataFrame(columns=IDENTITY_COLUMNS + _CARRIED_COLUMNS + ["mean", "ci_low", "ci_high", "n_starts"])

    normalised = _fill_key_nulls(scores, _NULLABLE_KEY_COLUMNS)
    normalised["start_date"] = pandas.to_datetime(normalised["start_date"])
    starts = numpy.sort(normalised["start_date"].dropna().unique())
    start_position = {start: index for index, start in enumerate(starts)}
    start_count = len(starts)
    bootstrap_indices = _bootstrap_start_indices(start_count, n_bootstrap, seed)

    aligned_groups, carried = _build_aligned_groups(normalised, start_position, start_count)

    baseline_by_metric_key: dict[tuple, _StartAlignedGroup] = {}
    if baseline_challenger is not None:
        for identity, group in aligned_groups.items():
            if identity[0] == baseline_challenger:
                baseline_by_metric_key[_metric_key_of(identity)] = group

    rows = []
    for identity, group in aligned_groups.items():
        summary = _summarise_group(group, bootstrap_indices, confidence)
        row = dict(zip(IDENTITY_COLUMNS, identity))
        row.update(carried[identity])
        row.update(summary)
        if baseline_challenger is not None:
            baseline_group = baseline_by_metric_key.get(_metric_key_of(identity))
            if baseline_group is not None:
                skill = _skill_for_pair(group, baseline_group, bootstrap_indices, confidence)
                row[f"skill_vs_{baseline_challenger}"] = skill["skill"]
                row["skill_ci_low"] = skill["skill_ci_low"]
                row["skill_ci_high"] = skill["skill_ci_high"]
                row["n_starts_paired"] = skill["n_starts_paired"]
        rows.append(row)

    tidy = pandas.DataFrame(rows)
    for column in _NULLABLE_KEY_COLUMNS:
        tidy[column] = tidy[column].where(tidy[column] != _NULL_SENTINEL, None)
    return tidy.sort_values(IDENTITY_COLUMNS, na_position="last").reset_index(drop=True)


def _metric_key_of(identity: tuple) -> tuple:
    return identity[len(["challenger", "year", "region"]) :]


def summary_to_json_records(summary: pandas.DataFrame) -> list[dict]:
    """JSON-ready records for the aggregated table (NaN -> null, dates/ints as plain scalars)."""
    frame = summary.copy()
    for column in frame.columns:
        if pandas.api.types.is_float_dtype(frame[column]):
            frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    frame = frame.astype(object).where(pandas.notna(frame), None)
    return frame.to_dict(orient="records")
