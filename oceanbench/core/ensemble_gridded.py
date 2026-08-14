# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Gridded probabilistic scores for an ensemble challenger against a gridded reference.

The deterministic gridded axis reduces a forecast to one field per lead day and reports its
area-weighted RMSD against the GLO12 analysis. An ensemble carries a whole predictive
distribution at every grid point, so it is scored with three numbers per forecast start,
lead day, variable and region:

``crps_fair``
    The fair (finite-ensemble unbiased) continuous ranked probability score of Ferro (2014),

        CRPS_fair = mean_i |x_i - y| - sum_i sum_j |x_i - x_j| / (2 M (M - 1))

    It is a proper score of the entire distribution and, unlike the biased estimator, it is
    comparable across ensembles of different size. The biased estimator divides the spread
    term by ``2 M^2`` instead and is emitted alongside as an implementation check: their
    difference is O(1/M) and always signed the same way. A one-member ensemble has no fair
    estimator, since the spread term divides by ``M - 1``. By explicit convention this module
    then returns the mean absolute error, which is what the CRPS reduces to at ``M = 1``.

``spread_error_ratio``
    Ensemble spread over ensemble-mean error, with the ``(M + 1) / M`` finite-size correction
    applied to the spread:

        spread = sqrt( (M + 1) / M * mean(s^2) ),  error = sqrt( mean((xbar - y)^2) )

    One is reliable, below one under-dispersive, above one over-dispersive. No observation
    error variance is subtracted: the reference is a gridded analysis sampled on the scoring
    grid itself, not a point measurement, so the class-4 obs-sigma correction does not belong
    on this axis.

``ensemble_mean_rmsd``
    The area-weighted RMSD of the ensemble mean, computed exactly as
    :mod:`oceanbench.core.rmsd` computes a deterministic RMSD, so an ensemble challenger can
    be read in the same table as a deterministic one. ``member_rmsd``, the root of the mean
    squared error averaged over the members, is emitted next to it as the single-realisation
    comparator; averaging suppresses unpredictable scales, so the ensemble mean is expected
    to beat it at every lead.

Every spatial average uses the same cosine-of-latitude area weights as the deterministic
metric, and every score is computed per forecast start before being averaged over starts,
so the aggregation matches ``oceanbench.core.rmsd`` step for step.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import math

import numpy
import xarray

from oceanbench.core.dataset_utils import VARIABLE_METADATA, Dimension
from oceanbench.core.rmsd import spatial_area_weights
from oceanbench.core.score_records import RunContext, score_record

ENSEMBLE_DIMENSION = "member"

METRIC_CRPS_FAIR = "crps_fair"
METRIC_CRPS_BIASED = "crps_biased"
METRIC_SPREAD_ERROR_RATIO = "spread_error_ratio"
METRIC_ENSEMBLE_SPREAD = "ensemble_spread"
METRIC_ENSEMBLE_MEAN_RMSD = "ensemble_mean_rmsd"
METRIC_MEMBER_RMSD = "member_rmsd"

RATIO_UNIT = "1"


def _pairwise_absolute_difference_sum(members: numpy.ndarray) -> numpy.ndarray:
    """``sum_i sum_j |x_i - x_j|`` over the last axis of ``members``.

    Uses the sorted-ensemble identity ``sum_i sum_j |x_i - x_j| = 2 sum_k (2k - M - 1) x_(k)``
    for ``k = 1..M`` ascending, which costs O(M log M) instead of the O(M^2) double loop.
    """
    member_count = members.shape[-1]
    sorted_members = numpy.sort(members, axis=-1)
    rank_coefficients = 2.0 * numpy.arange(1, member_count + 1) - member_count - 1
    return 2.0 * numpy.einsum("...k,k->...", sorted_members, rank_coefficients)


def _crps_values(members: numpy.ndarray, reference: numpy.ndarray, fair: bool) -> numpy.ndarray:
    member_count = members.shape[-1]
    mean_absolute_error = numpy.abs(members - reference[..., None]).mean(axis=-1)
    if member_count == 1:
        return mean_absolute_error
    spread_divisor = 2 * member_count * (member_count - 1) if fair else 2 * member_count**2
    return mean_absolute_error - _pairwise_absolute_difference_sum(members) / spread_divisor


def continuous_ranked_probability_score(
    members: xarray.DataArray,
    reference: xarray.DataArray,
    *,
    fair: bool = True,
    ensemble_dimension: str = ENSEMBLE_DIMENSION,
) -> xarray.DataArray:
    """Per-grid-point CRPS of ``members`` against ``reference``, fair by default.

    At a single member the fair estimator is undefined and the mean absolute error is
    returned instead, which is the value the CRPS takes at ``M = 1``.
    """
    return xarray.apply_ufunc(
        _crps_values,
        members,
        reference,
        input_core_dims=[[ensemble_dimension], []],
        kwargs={"fair": fair},
        dask="parallelized",
        output_dtypes=[float],
    )


def ensemble_spread(
    members: xarray.DataArray,
    *,
    ensemble_dimension: str = ENSEMBLE_DIMENSION,
) -> xarray.DataArray:
    """Per-grid-point ensemble spread, ``sqrt((M + 1) / M * s^2)`` with ``s^2`` unbiased."""
    member_count = members.sizes[ensemble_dimension]
    variance = members.var(dim=ensemble_dimension, ddof=1)
    return numpy.sqrt(finite_ensemble_correction(member_count) * variance)


def finite_ensemble_correction(member_count: int) -> float:
    """``(M + 1) / M``, the factor relating the spread of a finite ensemble to its error."""
    return (member_count + 1) / member_count


def area_weighted_mean(field: xarray.DataArray) -> float:
    """Cosine-of-latitude weighted spatial mean, ignoring cells that are not finite."""
    spatial_dimensions = [Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()]
    return float(field.weighted(spatial_area_weights(field)).mean(dim=spatial_dimensions))


@dataclass(frozen=True)
class EnsembleFieldStatistics:
    """Area-weighted statistics of one ensemble field against one reference field.

    All members are spatial averages over the scored cells of a single (forecast start, lead
    day, variable, region) field, kept as means rather than roots so that averaging over
    forecast starts stays a plain arithmetic mean. ``scored_cell_count`` is the number of grid
    cells that carried a finite value in both the ensemble and the reference.
    """

    member_count: int
    scored_cell_count: int
    crps_fair: float
    crps_biased: float
    ensemble_mean_absolute_error: float
    ensemble_mean_squared_error: float
    ensemble_variance: float
    member_squared_error: float


def ensemble_field_statistics(
    members: xarray.DataArray,
    reference: xarray.DataArray,
    *,
    ensemble_dimension: str = ENSEMBLE_DIMENSION,
) -> EnsembleFieldStatistics:
    """Reduce one ensemble field and its reference to the area-weighted statistics above."""
    member_count = members.sizes[ensemble_dimension]
    ensemble_mean = members.mean(dim=ensemble_dimension)
    ensemble_mean_error = ensemble_mean - reference
    member_error = members - reference
    variance = members.var(dim=ensemble_dimension, ddof=1)
    return EnsembleFieldStatistics(
        member_count=member_count,
        scored_cell_count=int(numpy.isfinite(ensemble_mean_error).sum()),
        crps_fair=area_weighted_mean(continuous_ranked_probability_score(members, reference, fair=True)),
        crps_biased=area_weighted_mean(continuous_ranked_probability_score(members, reference, fair=False)),
        ensemble_mean_absolute_error=area_weighted_mean(abs(ensemble_mean_error)),
        ensemble_mean_squared_error=area_weighted_mean(ensemble_mean_error**2),
        ensemble_variance=float("nan") if member_count < 2 else area_weighted_mean(variance),
        member_squared_error=area_weighted_mean((member_error**2).mean(dim=ensemble_dimension)),
    )


def spread_error_ratio(mean_ensemble_variance: float, mean_squared_error: float, member_count: int) -> float:
    """Corrected ensemble spread divided by ensemble-mean error, both as roots of means."""
    if member_count < 2 or not math.isfinite(mean_ensemble_variance) or mean_squared_error <= 0:
        return float("nan")
    spread = math.sqrt(finite_ensemble_correction(member_count) * mean_ensemble_variance)
    return spread / math.sqrt(mean_squared_error)


def field_metric_values(statistics: EnsembleFieldStatistics) -> dict[str, float]:
    """The published metric values of one field, keyed by metric name."""
    ensemble_mean_rmsd = math.sqrt(statistics.ensemble_mean_squared_error)
    spread = math.sqrt(finite_ensemble_correction(statistics.member_count) * statistics.ensemble_variance)
    return {
        METRIC_CRPS_FAIR: statistics.crps_fair,
        METRIC_CRPS_BIASED: statistics.crps_biased,
        METRIC_ENSEMBLE_MEAN_RMSD: ensemble_mean_rmsd,
        METRIC_MEMBER_RMSD: math.sqrt(statistics.member_squared_error),
        METRIC_ENSEMBLE_SPREAD: spread,
        METRIC_SPREAD_ERROR_RATIO: spread_error_ratio(
            statistics.ensemble_variance,
            statistics.ensemble_mean_squared_error,
            statistics.member_count,
        ),
    }


def _aggregate_metric_values(per_start_values: list[dict[str, float]]) -> dict[str, float]:
    """Average a lead day's per-start metric values into the score published for that lead day.

    Every metric is averaged over the forecast starts after its own root is taken, which is
    how ``oceanbench.core.rmsd`` averages over ``first_day_datetime``: root per start, mean of
    the roots. The spread-error ratio is the only exception, recomputed from the averaged
    spread and the averaged ensemble-mean RMSD rather than averaged directly, so that one
    start with a near-zero error cannot dominate the ratio.
    """
    averaged = {
        metric: float(numpy.mean([values[metric] for values in per_start_values]))
        for metric in per_start_values[0]
        if metric != METRIC_SPREAD_ERROR_RATIO
    }
    error = averaged[METRIC_ENSEMBLE_MEAN_RMSD]
    return {
        **averaged,
        METRIC_SPREAD_ERROR_RATIO: averaged[METRIC_ENSEMBLE_SPREAD] / error if error > 0 else float("nan"),
    }


def _metric_records(
    metric_values: dict[str, float],
    *,
    context: RunContext,
    reference: str,
    variable: str,
    depth: str | None,
    lead_day: int,
    start_date: object,
    scored_cell_count: int,
) -> list[dict]:
    variable_unit = VARIABLE_METADATA[variable][1]
    return [
        score_record(
            context=context,
            metric=metric,
            value=value,
            unit=RATIO_UNIT if metric == METRIC_SPREAD_ERROR_RATIO else variable_unit,
            reference=reference,
            variable=variable,
            depth=depth,
            lead_day=lead_day,
            start_date=start_date,
            sample_count=scored_cell_count,
        )
        for metric, value in metric_values.items()
    ]


def ensemble_gridded_records(
    statistics: Mapping[tuple[object, int, str], EnsembleFieldStatistics],
    *,
    context: RunContext,
    reference: str,
    depth: str | None = "surface",
) -> list[dict]:
    """Long-format ``scores.parquet`` records for a whole ensemble gridded run.

    ``statistics`` is keyed on ``(start_date, lead_day, variable)``. One record per metric is
    emitted for every key, plus one aggregate record per ``(lead_day, variable)`` carrying a
    null ``start_date``, which is the value the published tables read.
    """
    per_start_records = [
        record
        for (start_date, lead_day, variable), field_statistics in sorted(
            statistics.items(), key=lambda item: (str(item[0][0]), item[0][1], item[0][2])
        )
        for record in _metric_records(
            field_metric_values(field_statistics),
            context=context,
            reference=reference,
            variable=variable,
            depth=depth,
            lead_day=lead_day,
            start_date=start_date,
            scored_cell_count=field_statistics.scored_cell_count,
        )
    ]

    lead_day_variable_keys = sorted({(lead_day, variable) for _start, lead_day, variable in statistics})
    aggregate_records = []
    for lead_day, variable in lead_day_variable_keys:
        matching = [
            field_statistics
            for (_start, key_lead_day, key_variable), field_statistics in statistics.items()
            if (key_lead_day, key_variable) == (lead_day, variable)
        ]
        aggregate_records += _metric_records(
            _aggregate_metric_values([field_metric_values(field_statistics) for field_statistics in matching]),
            context=context,
            reference=reference,
            variable=variable,
            depth=depth,
            lead_day=lead_day,
            start_date=None,
            scored_cell_count=int(numpy.sum([field_statistics.scored_cell_count for field_statistics in matching])),
        )
    return per_start_records + aggregate_records
