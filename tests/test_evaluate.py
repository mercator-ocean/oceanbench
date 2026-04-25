# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pytest

from oceanbench.core import evaluate
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.runtime_configuration import RuntimeConfiguration


def test_execute_evaluation_notebook_uses_large_iopub_timeout_by_default(monkeypatch) -> None:
    execute_notebook_calls = []

    def execute_notebook(*args, **kwargs):
        execute_notebook_calls.append((args, kwargs))

    monkeypatch.delenv(
        OceanbenchEnvironmentVariable.OCEANBENCH_NOTEBOOK_IOPUB_TIMEOUT_SECONDS.value,
        raising=False,
    )
    monkeypatch.setattr(evaluate, "execute_notebook", execute_notebook)

    evaluate._execute_evaluation_notebook_file(
        "report.ipynb",
        output_bucket=None,
        output_prefix=None,
        runtime_configuration=RuntimeConfiguration(),
    )

    assert execute_notebook_calls == [
        (
            ("report.ipynb", "report.ipynb"),
            {"iopub_timeout": evaluate.DEFAULT_NOTEBOOK_IOPUB_TIMEOUT_SECONDS},
        )
    ]


def test_execute_evaluation_notebook_uses_configured_iopub_timeout(monkeypatch) -> None:
    execute_notebook_calls = []

    def execute_notebook(*args, **kwargs):
        execute_notebook_calls.append((args, kwargs))

    monkeypatch.setenv(
        OceanbenchEnvironmentVariable.OCEANBENCH_NOTEBOOK_IOPUB_TIMEOUT_SECONDS.value,
        "12",
    )
    monkeypatch.setattr(evaluate, "execute_notebook", execute_notebook)

    evaluate._execute_evaluation_notebook_file(
        "report.ipynb",
        output_bucket=None,
        output_prefix=None,
        runtime_configuration=RuntimeConfiguration(),
    )

    assert execute_notebook_calls == [
        (
            ("report.ipynb", "report.ipynb"),
            {"iopub_timeout": 12},
        )
    ]


@pytest.mark.parametrize("configured_timeout", ["0", "-1"])
def test_execute_evaluation_notebook_rejects_invalid_iopub_timeout(
    monkeypatch,
    configured_timeout: str,
) -> None:
    monkeypatch.setenv(
        OceanbenchEnvironmentVariable.OCEANBENCH_NOTEBOOK_IOPUB_TIMEOUT_SECONDS.value,
        configured_timeout,
    )

    with pytest.raises(ValueError, match="must be greater than or equal to 1"):
        evaluate._execute_evaluation_notebook_file(
            "report.ipynb",
            output_bucket=None,
            output_prefix=None,
            runtime_configuration=RuntimeConfiguration(),
        )
