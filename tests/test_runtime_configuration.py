# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.runtime_configuration import runtime_configuration_from_environment


RUNTIME_ENVIRONMENT_VARIABLES = [
    OceanbenchEnvironmentVariable.OCEANBENCH_STAGE,
    OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_DIR,
    OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_MAX_WORKERS,
    OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES,
    OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_FAST_INTERPOLATION,
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
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_FAST_INTERPOLATION.value, "1")

    runtime_configuration = runtime_configuration_from_environment()

    assert runtime_configuration.staged_components == ("references", "observations")
    assert runtime_configuration.stage_directory == "/tmp/oceanbench-stage-env"
    assert runtime_configuration.stage_max_workers == 2
    assert runtime_configuration.remote_retries == 7
    assert runtime_configuration.class4_fast_interpolation is True


def test_runtime_configuration_rejects_invalid_class4_fast_interpolation(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_FAST_INTERPOLATION.value, "true")

    try:
        runtime_configuration_from_environment()
    except ValueError as error:
        assert "OCEANBENCH_CLASS4_FAST_INTERPOLATION must be '0' or '1'." == str(error)
    else:
        raise AssertionError("Expected invalid class4 fast interpolation value to fail.")
