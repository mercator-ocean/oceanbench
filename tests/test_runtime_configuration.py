# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from oceanbench.cli import _build_parser, _runtime_configuration_from_args
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.runtime_configuration import runtime_configuration_from_environment


RUNTIME_ENVIRONMENT_VARIABLES = [
    OceanbenchEnvironmentVariable.OCEANBENCH_STAGE,
    OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_DIR,
    OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_MAX_WORKERS,
    OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES,
]


def _clear_runtime_environment(monkeypatch):
    for environment_variable in RUNTIME_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(environment_variable.value, raising=False)


def test_runtime_configuration_reads_environment(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_STAGE.value, "references, observations")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_DIR.value, "/tmp/oceanbench-stage-env")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_MAX_WORKERS.value, "2")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES.value, "7")

    runtime_configuration = runtime_configuration_from_environment()

    assert runtime_configuration.staged_components == ("references", "observations")
    assert runtime_configuration.stage_directory == "/tmp/oceanbench-stage-env"
    assert runtime_configuration.stage_max_workers == 2
    assert runtime_configuration.remote_retries == 7


def test_evaluate_cli_runtime_arguments_override_environment(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_STAGE.value, "references")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_DIR.value, "/tmp/oceanbench-stage-env")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_MAX_WORKERS.value, "2")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES.value, "7")
    parser, _evaluate_parser = _build_parser()
    args = parser.parse_args(
        [
            "evaluate",
            "challenger.py",
            "--stage",
            "challenger",
            "--stage",
            "observations",
            "--stage-dir",
            "/tmp/oceanbench-stage-cli",
            "--stage-max-workers",
            "3",
            "--remote-retries",
            "4",
        ]
    )

    runtime_configuration = _runtime_configuration_from_args(args)

    assert runtime_configuration.staged_components == ("challenger", "observations")
    assert runtime_configuration.stage_directory == "/tmp/oceanbench-stage-cli"
    assert runtime_configuration.stage_max_workers == 3
    assert runtime_configuration.remote_retries == 4


def test_evaluate_cli_uses_environment_runtime_configuration_by_default(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_STAGE.value, "all")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_DIR.value, "/tmp/oceanbench-stage-env")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_MAX_WORKERS.value, "2")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES.value, "7")
    parser, _evaluate_parser = _build_parser()
    args = parser.parse_args(["evaluate", "challenger.py"])

    runtime_configuration = _runtime_configuration_from_args(args)

    assert runtime_configuration.staged_components == ("all",)
    assert runtime_configuration.stage_directory == "/tmp/oceanbench-stage-env"
    assert runtime_configuration.stage_max_workers == 2
    assert runtime_configuration.remote_retries == 7
