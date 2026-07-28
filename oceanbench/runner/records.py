# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Convert the legacy per-metric score dataframes into long-format records.

Each of the nine metric functions in ``oceanbench.core.metrics`` returns a
pretty dataframe whose row index is a label of the form
``"{Display} ({unit}) [{standard_name}]{{depth_or_bin}}"`` and whose columns are
``"Lead day N"``. This module turns such a dataframe into records matching the
``scores.parquet`` schema of ``docs/contracts.md`` §3.1. It has no dependency on
xarray or the metric machinery so it can be unit-tested in isolation.
"""

from dataclasses import dataclass
from datetime import date, datetime
import math
import re

import pandas

from oceanbench.core.dataset_utils import VARIABLE_METADATA

SCORE_COLUMNS = [
    "challenger",
    "challenger_version",
    "year",
    "region",
    "metric",
    "reference",
    "variable",
    "depth",
    "lead_day",
    "start_date",
    "band",
    "polarity",
    "value",
    "unit",
    "n",
    "oceanbench_version",
]

METRIC_ROOT_MEAN_SQUARE_DEVIATION = "rmsd"
METRIC_CLASS4_ROOT_MEAN_SQUARE_DEVIATION = "class4_rmsd"
METRIC_CLASS4_BIAS = "class4_bias"
METRIC_LAGRANGIAN_DEVIATION_KILOMETRES = "lagrangian_deviation_km"

METRIC_PSD_BAND_ENERGY_FRACTION = "psd_band_energy_fraction"
METRIC_EFFECTIVE_RESOLUTION_KILOMETRES = "effective_resolution_km"
METRIC_ERROR_SPECTRUM_BAND_ENERGY = "error_spectrum_band_energy"
METRIC_ACTIVITY_RATIO = "activity_ratio"
METRIC_EDDY_COUNT = "eddy_count"
METRIC_EDDY_HIT_RATE = "eddy_hit_rate"
METRIC_EDDY_MISS_RATE = "eddy_miss_rate"
METRIC_EDDY_MEAN_DISPLACEMENT_KILOMETRES = "eddy_mean_displacement_km"

METRIC_GRID_COVERAGE = "grid_coverage"

DIAGNOSTIC_METRICS = frozenset({METRIC_GRID_COVERAGE})

_LABEL_PATTERN = re.compile(r"^(.*?) \(([^)]*)\) \[([^\]]*)\](?:\{([^}]+)\})?$")
_LEAD_DAY_PATTERN = re.compile(r"(\d+)\s*$")


@dataclass(frozen=True)
class RunContext:
    challenger: str
    challenger_version: str
    year: int
    region: str
    oceanbench_version: str


def _parse_label(label: str) -> tuple[str, str | None, str | None, str | None]:
    match = _LABEL_PATTERN.match(label)
    if match is None:
        return label, None, None, None
    display_name, unit, standard_name, depth = match.group(1), match.group(2), match.group(3), match.group(4)
    return display_name, (unit or None), (standard_name or None), (depth or None)


def _lead_day_from_column(column: object) -> int | None:
    match = _LEAD_DAY_PATTERN.search(str(column))
    return int(match.group(1)) if match else None


def _clean_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    numeric_value = float(value)
    return None if math.isnan(numeric_value) else numeric_value


def _normalise_start_date(start_date: object) -> date | None:
    if start_date is None:
        return None
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        return start_date
    return pandas.Timestamp(start_date).date()


def _resolved_depth(raw_depth: str | None, depth_applicable: bool) -> str | None:
    if not depth_applicable or raw_depth is None:
        return None
    return raw_depth.lower()


def _records_from_score_frame(
    dataframe: pandas.DataFrame,
    *,
    metric: str,
    reference: str | None,
    context: RunContext,
    start_date: object,
    depth_applicable: bool,
    sample_counts: dict[str, int] | None = None,
) -> list[dict]:
    normalised_start_date = _normalise_start_date(start_date)
    records = []
    for label, row in dataframe.iterrows():
        display_name, unit, standard_name, raw_depth = _parse_label(str(label))
        depth = _resolved_depth(raw_depth, depth_applicable)
        sample_count = None if sample_counts is None else sample_counts.get(str(label))
        for column, value in row.items():
            records.append(
                {
                    "challenger": context.challenger,
                    "challenger_version": context.challenger_version,
                    "year": context.year,
                    "region": context.region,
                    "metric": metric,
                    "reference": reference,
                    "variable": standard_name,
                    "depth": depth,
                    "lead_day": _lead_day_from_column(column),
                    "start_date": normalised_start_date,
                    "band": None,
                    "polarity": None,
                    "value": _clean_value(value),
                    "unit": unit,
                    "n": sample_count,
                    "oceanbench_version": context.oceanbench_version,
                }
            )
    return records


def gridded_rmsd_records(
    dataframe: pandas.DataFrame,
    *,
    reference: str,
    context: RunContext,
    start_date: object,
    depth_applicable: bool = True,
) -> list[dict]:
    """Records for a gridded ``rmsd`` dataframe (variables, mixed layer depth, geostrophic).

    ``depth_applicable`` is ``True`` for the multi-depth variable table and
    ``False`` for the depth-agnostic mixed-layer-depth and geostrophic tables.
    """
    return _records_from_score_frame(
        dataframe,
        metric=METRIC_ROOT_MEAN_SQUARE_DEVIATION,
        reference=reference,
        context=context,
        start_date=start_date,
        depth_applicable=depth_applicable,
    )


def class4_records(
    dataframe: pandas.DataFrame,
    *,
    context: RunContext,
    start_date: object = None,
    sample_counts: dict[str, int] | None = None,
) -> list[dict]:
    """Records for a Class-4 ``class4_rmsd`` dataframe (reference is observations)."""
    return _records_from_score_frame(
        dataframe,
        metric=METRIC_CLASS4_ROOT_MEAN_SQUARE_DEVIATION,
        reference="observations",
        context=context,
        start_date=start_date,
        depth_applicable=True,
        sample_counts=sample_counts,
    )


def class4_per_start_records(
    per_start_table: pandas.DataFrame,
    *,
    context: RunContext,
) -> list[dict]:
    """Records for the per-start Class-4 table (columns: variable, first_day, depth_bin, lead_day, rmsd, count).

    Emits one row per (start_date, variable, depth_bin, lead_day): ``value`` is the RMSD
    over that forecast start's observations and ``n`` is that observation count. Consumers
    recover the published pooled-over-observations RMSD exactly via
    ``sqrt(sum(value ** 2 * n) / sum(n))``. ``lead_day`` is stored 1-based (the table's
    0-based lead day plus one), matching every other artifact.
    """
    return [
        {
            "challenger": context.challenger,
            "challenger_version": context.challenger_version,
            "year": context.year,
            "region": context.region,
            "metric": METRIC_CLASS4_ROOT_MEAN_SQUARE_DEVIATION,
            "reference": "observations",
            "variable": str(row.variable),
            "depth": str(row.depth_bin),
            "lead_day": int(row.lead_day) + 1,
            "start_date": _normalise_start_date(row.first_day),
            "band": None,
            "polarity": None,
            "value": _clean_value(row.rmsd),
            "unit": VARIABLE_METADATA[str(row.variable)][1],
            "n": int(row.count),
            "oceanbench_version": context.oceanbench_version,
        }
        for row in per_start_table.itertuples(index=False)
    ]


def lagrangian_records(
    dataframe: pandas.DataFrame,
    *,
    reference: str,
    context: RunContext,
    start_date: object,
) -> list[dict]:
    """Records for a ``lagrangian_deviation_km`` dataframe (variable/depth are null, starts at lead day 2)."""
    return _records_from_score_frame(
        dataframe,
        metric=METRIC_LAGRANGIAN_DEVIATION_KILOMETRES,
        reference=reference,
        context=context,
        start_date=start_date,
        depth_applicable=False,
    )


def _score_record(
    *,
    context: RunContext,
    metric: str,
    value: object,
    unit: str,
    reference: str | None = None,
    variable: str | None = None,
    depth: str | None = None,
    lead_day: int | None = None,
    band: str | None = None,
    polarity: str | None = None,
    start_date: object = None,
    sample_count: int | None = None,
) -> dict:
    return {
        "challenger": context.challenger,
        "challenger_version": context.challenger_version,
        "year": context.year,
        "region": context.region,
        "metric": metric,
        "reference": reference,
        "variable": variable,
        "depth": depth,
        "lead_day": lead_day,
        "start_date": _normalise_start_date(start_date),
        "band": band,
        "polarity": polarity,
        "value": _clean_value(value),
        "unit": unit,
        "n": sample_count,
        "oceanbench_version": context.oceanbench_version,
    }


def realism_record(
    *,
    context: RunContext,
    metric: str,
    value: object,
    unit: str,
    reference: str | None = None,
    variable: str | None = None,
    depth: str | None = None,
    lead_day: int | None = None,
    band: str | None = None,
    polarity: str | None = None,
    start_date: object = None,
    sample_count: int | None = None,
) -> dict:
    """Build one long-format ``scores.parquet`` record for a realism-battery metric (contracts.md §3.2).

    Realism metrics are computed directly (not parsed from the legacy pretty dataframes),
    so this fills every contract column explicitly. ``band`` carries the spectral band
    (``large`` / ``regional`` / ``mesoscale``) and ``polarity`` the eddy rotational sense
    (``cyclone`` / ``anticyclone``); both are ``None`` when the metric does not use them.
    Spectra and eddy metrics are aggregate over the forecast starts by nature, so
    ``start_date`` is ``None`` unless a per-start value is emitted.
    """
    return _score_record(
        context=context,
        metric=metric,
        value=value,
        unit=unit,
        reference=reference,
        variable=variable,
        depth=depth,
        lead_day=lead_day,
        band=band,
        polarity=polarity,
        start_date=start_date,
        sample_count=sample_count,
    )


def grid_coverage_record(
    *,
    context: RunContext,
    reference: str,
    coverage: float,
    matched_cell_count: int,
) -> dict:
    """Build the ``grid_coverage`` diagnostic record for one reference (contracts.md §3.2).

    ``coverage`` is the fraction of the challenger's cells the reference grid could supply,
    and ``n`` the count of those cells. A run that scores a snapped grid says so in its own
    scores file rather than only in the console, so a degraded run stays visible downstream
    (issue #305). Diagnostics carry no forecast start or lead day, and aggregation drops them.
    """
    return _score_record(
        context=context,
        metric=METRIC_GRID_COVERAGE,
        value=coverage,
        unit="1",
        reference=reference,
        sample_count=matched_cell_count,
    )


def records_to_dataframe(records: list[dict]) -> pandas.DataFrame:
    """Assemble records into a ``scores.parquet`` dataframe with the contract column order and dtypes."""
    dataframe = pandas.DataFrame(records, columns=SCORE_COLUMNS)
    if dataframe.empty:
        return dataframe
    dataframe = dataframe.astype(
        {
            "challenger": "string",
            "challenger_version": "string",
            "year": "int32",
            "region": "string",
            "metric": "string",
            "reference": "string",
            "variable": "string",
            "depth": "string",
            "lead_day": "Int8",
            "band": "string",
            "polarity": "string",
            "value": "float64",
            "unit": "string",
            "n": "Int32",
            "oceanbench_version": "string",
        }
    )
    dataframe["start_date"] = pandas.to_datetime(dataframe["start_date"])
    return dataframe
