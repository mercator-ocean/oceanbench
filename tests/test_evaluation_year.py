# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pytest

from oceanbench.core.evaluation_year import (
    evaluation_year_first_day_datetimes,
    validate_evaluation_year,
)
from oceanbench.core.challenger_datasets import (
    _glonet_dataset_path,
    _langya_dataset_path,
)
from oceanbench.core.references.glo12 import _glo12_1_degree_path, _glo12_1_4_path
from oceanbench.core.references.glorys import _glorys_1_degree_path
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


def test_ml_forecast_output_paths_use_true_evaluation_dates() -> None:
    first_day_datetimes_2023 = evaluation_year_first_day_datetimes(2023)
    first_day_datetimes_2025 = evaluation_year_first_day_datetimes(2025)

    assert _glonet_dataset_path(first_day_datetimes_2023[0]) == (
        "s3://oceanbench-bucket/public/ml-forecast-outputs/glonet/20230104.zarr"
    )
    assert _langya_dataset_path(first_day_datetimes_2025[-1]) == (
        "s3://oceanbench-bucket/public/ml-forecast-outputs/langya/20251231.zarr"
    )


def test_observation_path_uses_evaluation_year_bucket() -> None:
    assert observation_path("2023-01-04", 2023) == "s3://oceanbench-bucket/public/observations2023/20230104.zarr"
    assert observation_path("2025-01-04", 2025) == "s3://oceanbench-bucket/public/observations2025/20250104.zarr"
    assert observation_path("2024-01-04", 2024).endswith("/observations2024/20240104.zarr")


def test_reference_paths_use_cloudferro_for_true_2023_and_2025_evaluations() -> None:
    assert _glo12_1_degree_path("2023-01-04") == (
        "s3://oceanbench-bucket/public/references/glo12_one_degree_2023/20230104.zarr"
    )
    assert _glo12_1_4_path("2025-12-31") == (
        "s3://oceanbench-bucket/public/references/glo12_quarter_degree_2025/20251231.zarr"
    )
    assert _glorys_1_degree_path("2023-12-27") == (
        "s3://oceanbench-bucket/public/references/glorys_one_degree_2023/20231227.zarr"
    )
