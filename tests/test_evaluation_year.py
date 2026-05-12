# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pytest

from oceanbench.core.evaluation_year import (
    evaluation_year_first_day_datetimes,
    validate_evaluation_year,
)
from oceanbench.core.challenger_datasets import _forecast_source_first_day_datetimes
from oceanbench.core.references.observations import observation_path


def test_evaluation_year_first_day_datetimes_use_wednesdays_in_supported_year() -> None:
    first_day_datetimes_2023 = evaluation_year_first_day_datetimes(2023)
    first_day_datetimes_2024 = evaluation_year_first_day_datetimes(2024)
    first_day_datetimes_2025 = evaluation_year_first_day_datetimes(2025)

    assert first_day_datetimes_2023[0].strftime("%Y-%m-%d") == "2023-01-04"
    assert first_day_datetimes_2023[-1].strftime("%Y-%m-%d") == "2023-12-27"
    assert len(first_day_datetimes_2023) == 52
    assert first_day_datetimes_2024[0].strftime("%Y-%m-%d") == "2024-01-03"
    assert first_day_datetimes_2024[-1].strftime("%Y-%m-%d") == "2024-12-25"
    assert len(first_day_datetimes_2024) == 52
    assert first_day_datetimes_2025[0].strftime("%Y-%m-%d") == "2025-01-01"
    assert first_day_datetimes_2025[-1].strftime("%Y-%m-%d") == "2025-12-31"
    assert len(first_day_datetimes_2025) == 53


def test_validate_evaluation_year_rejects_unsupported_year() -> None:
    with pytest.raises(ValueError, match="evaluation_year"):
        validate_evaluation_year(2022)


def test_forecast_source_first_day_datetimes_reuse_2024_forecasts_for_extra_start() -> None:
    source_first_day_datetimes = _forecast_source_first_day_datetimes(53)

    assert source_first_day_datetimes[0].strftime("%Y-%m-%d") == "2024-01-03"
    assert source_first_day_datetimes[51].strftime("%Y-%m-%d") == "2024-12-25"
    assert source_first_day_datetimes[52].strftime("%Y-%m-%d") == "2024-01-03"


def test_observation_path_uses_evaluation_year_bucket() -> None:
    assert observation_path("2024-01-04", 2023).endswith("/observations2023/20240104.zarr")
