# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import dask

from oceanbench import cli
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.runner.matchups import default_matchup_worker_count


def test_matchup_worker_count_is_bounded_by_default(monkeypatch) -> None:
    monkeypatch.delenv(OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_MATCHUP_WORKERS.value, raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 96)

    assert default_matchup_worker_count() == 12


def test_matchup_worker_count_honours_the_environment_override(monkeypatch) -> None:
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_MATCHUP_WORKERS.value, "20")

    assert default_matchup_worker_count() == 20


def test_dask_worker_count_is_bounded_by_default(monkeypatch) -> None:
    monkeypatch.delenv(OceanbenchEnvironmentVariable.OCEANBENCH_DASK_WORKERS.value, raising=False)
    monkeypatch.setattr("os.cpu_count", lambda: 96)

    with dask.config.set({"num_workers": None}):
        cli._apply_default_dask_concurrency()

        assert dask.config.get("num_workers") == 8


def test_dask_worker_count_honours_the_environment_override(monkeypatch) -> None:
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_DASK_WORKERS.value, "3")

    with dask.config.set({"num_workers": 64}):
        cli._apply_default_dask_concurrency()

        assert dask.config.get("num_workers") == 3


def test_dask_worker_count_leaves_an_existing_configuration_alone(monkeypatch) -> None:
    monkeypatch.delenv(OceanbenchEnvironmentVariable.OCEANBENCH_DASK_WORKERS.value, raising=False)

    with dask.config.set({"num_workers": 64}):
        cli._apply_default_dask_concurrency()

        assert dask.config.get("num_workers") == 64
