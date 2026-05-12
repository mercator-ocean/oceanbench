# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime

from oceanbench.core.datetime_utils import generate_dates

DEFAULT_EVALUATION_YEAR = 2024
SUPPORTED_EVALUATION_YEARS = (2023, 2024, 2025)


def validate_evaluation_year(evaluation_year: int | str) -> int:
    try:
        parsed_evaluation_year = int(evaluation_year)
    except (TypeError, ValueError) as error:
        raise ValueError(f"evaluation_year must be one of {SUPPORTED_EVALUATION_YEARS}.") from error
    if parsed_evaluation_year not in SUPPORTED_EVALUATION_YEARS:
        raise ValueError(f"evaluation_year must be one of {SUPPORTED_EVALUATION_YEARS}, got {parsed_evaluation_year}.")
    return parsed_evaluation_year


def first_wednesday_of_year(evaluation_year: int) -> datetime:
    year_start = datetime(evaluation_year, 1, 1)
    days_until_wednesday = (2 - year_start.weekday()) % 7
    return year_start.replace(day=year_start.day + days_until_wednesday)


def last_wednesday_of_year(evaluation_year: int) -> datetime:
    year_end = datetime(evaluation_year, 12, 31)
    days_since_wednesday = (year_end.weekday() - 2) % 7
    return year_end.replace(day=year_end.day - days_since_wednesday)


def evaluation_year_first_day_datetimes(evaluation_year: int | str) -> list[datetime]:
    resolved_evaluation_year = validate_evaluation_year(evaluation_year)
    return generate_dates(
        first_wednesday_of_year(resolved_evaluation_year).strftime("%Y-%m-%d"),
        last_wednesday_of_year(resolved_evaluation_year).strftime("%Y-%m-%d"),
        7,
    )
