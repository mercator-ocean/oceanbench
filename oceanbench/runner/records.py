# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Convert the legacy per-metric score dataframes into v2 long-format records.

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
METRIC_LAGRANGIAN_DEVIATION_KILOMETRES = "lagrangian_deviation_km"

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
