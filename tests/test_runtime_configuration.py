# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from oceanbench.cli import _build_parser, _runtime_configuration_from_args
from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable
from oceanbench.core.runtime_configuration import runtime_configuration_from_environment


RUNTIME_ENVIRONMENT_VARIABLES = [
    OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE,
    OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE_REVALIDATE,
    OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES,
    OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_FAST_INTERPOLATION,
]


def _clear_runtime_environment(monkeypatch):
    for environment_variable in RUNTIME_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(environment_variable.value, raising=False)


def test_runtime_configuration_reads_environment(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE.value, "/tmp/oceanbench-cache-env")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE_REVALIDATE.value, "0")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES.value, "7")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_FAST_INTERPOLATION.value, "1")

    runtime_configuration = runtime_configuration_from_environment()

    assert runtime_configuration.local_cache_directory_path == "/tmp/oceanbench-cache-env"
    assert runtime_configuration.local_cache_revalidate is False
    assert runtime_configuration.remote_retries == 7
    assert runtime_configuration.class4_fast_interpolation is True


def test_runtime_configuration_defaults_to_online_with_revalidation(monkeypatch):
    _clear_runtime_environment(monkeypatch)

    runtime_configuration = runtime_configuration_from_environment()

    assert runtime_configuration.local_cache_directory_path is None
    assert runtime_configuration.local_cache_revalidate is True


def test_runtime_configuration_rejects_invalid_class4_fast_interpolation(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_FAST_INTERPOLATION.value, "true")

    try:
        runtime_configuration_from_environment()
    except ValueError as error:
        assert "OCEANBENCH_CLASS4_FAST_INTERPOLATION must be '0' or '1'." == str(error)
    else:
        raise AssertionError("Expected invalid class4 fast interpolation value to fail.")


def test_evaluate_cli_runtime_arguments_override_environment(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE.value, "/tmp/oceanbench-cache-env")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES.value, "7")
    parser, _evaluate_parser = _build_parser()
    args = parser.parse_args(
        ["evaluate", "challenger.py", "--local-cache", "/tmp/oceanbench-cache-cli", "--remote-retries", "4"]
    )

    runtime_configuration = _runtime_configuration_from_args(args)

    assert runtime_configuration.local_cache_directory_path == "/tmp/oceanbench-cache-cli"
    assert runtime_configuration.remote_retries == 4


def test_evaluate_cli_uses_environment_runtime_configuration_by_default(monkeypatch):
    _clear_runtime_environment(monkeypatch)
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE.value, "/tmp/oceanbench-cache-env")
    monkeypatch.setenv(OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES.value, "7")
    parser, _evaluate_parser = _build_parser()
    args = parser.parse_args(["evaluate", "challenger.py"])

    runtime_configuration = _runtime_configuration_from_args(args)

    assert runtime_configuration.local_cache_directory_path == "/tmp/oceanbench-cache-env"
    assert runtime_configuration.remote_retries == 7
