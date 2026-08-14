# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Long-format score records for the ensemble gridded axis.

The deterministic axes publish their scores as pretty dataframes indexed by a variable label
and columned by lead day. The ensemble axis carries six metrics over a (forecast start, lead
day, variable, region) key, which does not fit that shape, so it emits one flat record per
metric value instead and lets the consumer pivot.

The columns below are the contract the ensemble scoring scripts write to parquet. Only the
record builder and the frame assembler live here: nothing in this module knows about xarray
or the metric machinery, so it is unit-testable on its own.
"""

from dataclasses import dataclass
from datetime import date, datetime
import math

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
    "value",
    "unit",
    "n",
    "oceanbench_version",
]


@dataclass(frozen=True)
class RunContext:
    """The run-level identity every record of a scoring run repeats."""

    challenger: str
    challenger_version: str
    year: int
    region: str
    oceanbench_version: str


def _clean_value(value: object) -> float | None:
    if value is None:
        return None
    numeric_value = float(value)
    return None if math.isnan(numeric_value) else numeric_value


def _normalise_start_date(start_date: object) -> date | None:
    if start_date is None:
        return None
    if isinstance(start_date, date) and not isinstance(start_date, datetime):
        return start_date
    return pandas.Timestamp(start_date).date()


def score_record(
    *,
    context: RunContext,
    metric: str,
    value: object,
    unit: str,
    reference: str | None = None,
    variable: str | None = None,
    depth: str | None = None,
    lead_day: int | None = None,
    start_date: object = None,
    sample_count: int | None = None,
) -> dict:
    """Build one long-format score record.

    A ``start_date`` of ``None`` marks the record as the value aggregated over the forecast
    starts, which is what the published tables read.
    """
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
        "value": _clean_value(value),
        "unit": unit,
        "n": sample_count,
        "oceanbench_version": context.oceanbench_version,
    }


def records_to_dataframe(records: list[dict]) -> pandas.DataFrame:
    """Assemble records into a dataframe with the contract column order and dtypes."""
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
            "value": "float64",
            "unit": "string",
            "n": "Int32",
            "oceanbench_version": "string",
        }
    )
    dataframe["start_date"] = pandas.to_datetime(dataframe["start_date"])
    return dataframe
