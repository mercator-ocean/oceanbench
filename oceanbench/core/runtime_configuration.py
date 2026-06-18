# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
import os
from pathlib import Path

from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable

DEFAULT_REMOTE_HTTP_RETRIES = 5
FALSE_ENVIRONMENT_VALUE = "0"
TRUE_ENVIRONMENT_VALUE = "1"


@dataclass(frozen=True)
class RuntimeConfiguration:
    remote_retries: int = DEFAULT_REMOTE_HTTP_RETRIES
    class4_fast_interpolation: bool = False
    local_cache_directory_path: str | None = None
    local_cache_revalidate: bool = True

    def __post_init__(self):
        if self.remote_retries < 1:
            raise ValueError("remote_retries must be greater than or equal to 1.")

    def local_cache_directory(self) -> Path | None:
        if self.local_cache_directory_path is None:
            return None
        return Path(self.local_cache_directory_path)


def _parse_zero_one_environment_variable(environment_variable: OceanbenchEnvironmentVariable) -> bool:
    raw_value = os.environ.get(environment_variable.value, FALSE_ENVIRONMENT_VALUE)
    if raw_value == TRUE_ENVIRONMENT_VALUE:
        return True
    if raw_value == FALSE_ENVIRONMENT_VALUE:
        return False
    raise ValueError(f"{environment_variable.value} must be {FALSE_ENVIRONMENT_VALUE!r} or {TRUE_ENVIRONMENT_VALUE!r}.")


def _parse_runtime_configuration_from_environment() -> RuntimeConfiguration:
    local_cache_directory_path = os.environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE.value) or None
    local_cache_revalidate = (
        os.environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE_REVALIDATE.value, TRUE_ENVIRONMENT_VALUE)
        != FALSE_ENVIRONMENT_VALUE
    )
    remote_retries = int(
        os.environ.get(
            OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES.value,
            DEFAULT_REMOTE_HTTP_RETRIES,
        )
    )
    return RuntimeConfiguration(
        remote_retries=remote_retries,
        class4_fast_interpolation=_parse_zero_one_environment_variable(
            OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_FAST_INTERPOLATION
        ),
        local_cache_directory_path=local_cache_directory_path,
        local_cache_revalidate=local_cache_revalidate,
    )


def runtime_configuration_from_environment() -> RuntimeConfiguration:
    return _parse_runtime_configuration_from_environment()


_runtime_configuration: RuntimeConfiguration | None = None


def current_runtime_configuration() -> RuntimeConfiguration:
    return _runtime_configuration or runtime_configuration_from_environment()


def set_runtime_configuration(runtime_configuration: RuntimeConfiguration) -> None:
    global _runtime_configuration
    _runtime_configuration = runtime_configuration
