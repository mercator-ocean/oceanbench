# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

from oceanbench.core.environment_variables import OceanbenchEnvironmentVariable

STAGE_ALL_KEY = "all"
DEFAULT_STAGE_MAX_WORKERS = min(4, os.cpu_count() or 1)
DEFAULT_REMOTE_HTTP_RETRIES = 5
FALSE_ENVIRONMENT_VALUE = "0"
TRUE_ENVIRONMENT_VALUE = "1"


@dataclass(frozen=True)
class RuntimeConfiguration:
    staged_components: tuple[str, ...] = ()
    stage_directory: str | None = None
    stage_max_workers: int = DEFAULT_STAGE_MAX_WORKERS
    remote_retries: int = DEFAULT_REMOTE_HTTP_RETRIES
    class4_fast_interpolation: bool = False
    local_cache_directory_path: str | None = None

    def __post_init__(self):
        normalized_components = tuple(dict.fromkeys(component.strip().lower() for component in self.staged_components))
        if self.stage_max_workers < 1:
            raise ValueError("stage_max_workers must be greater than or equal to 1.")
        if self.remote_retries < 1:
            raise ValueError("remote_retries must be greater than or equal to 1.")
        object.__setattr__(self, "staged_components", normalized_components)

    def has_local_stage(self) -> bool:
        return bool(self.staged_components)

    def should_stage(self, stage_key: str) -> bool:
        normalized_stage_key = stage_key.strip().lower()
        return normalized_stage_key in self.staged_components or STAGE_ALL_KEY in self.staged_components

    def resolved_stage_directory(self) -> Path:
        if self.stage_directory is not None:
            return Path(self.stage_directory)
        return Path(tempfile.gettempdir()) / "oceanbench-stage"

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
    staged_components = tuple(
        component.strip()
        for component in os.environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_STAGE.value, "").split(",")
        if component.strip()
    )
    stage_directory = os.environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_DIR.value) or None
    local_cache_directory_path = os.environ.get(OceanbenchEnvironmentVariable.OCEANBENCH_LOCAL_CACHE.value) or None
    stage_max_workers = int(
        os.environ.get(
            OceanbenchEnvironmentVariable.OCEANBENCH_STAGE_MAX_WORKERS.value,
            DEFAULT_STAGE_MAX_WORKERS,
        )
    )
    remote_retries = int(
        os.environ.get(
            OceanbenchEnvironmentVariable.OCEANBENCH_REMOTE_RETRIES.value,
            DEFAULT_REMOTE_HTTP_RETRIES,
        )
    )
    return RuntimeConfiguration(
        staged_components=staged_components,
        stage_directory=stage_directory,
        stage_max_workers=stage_max_workers,
        remote_retries=remote_retries,
        class4_fast_interpolation=_parse_zero_one_environment_variable(
            OceanbenchEnvironmentVariable.OCEANBENCH_CLASS4_FAST_INTERPOLATION
        ),
        local_cache_directory_path=local_cache_directory_path,
    )


def runtime_configuration_from_environment() -> RuntimeConfiguration:
    return _parse_runtime_configuration_from_environment()


_runtime_configuration: RuntimeConfiguration | None = None


def current_runtime_configuration() -> RuntimeConfiguration:
    return _runtime_configuration or runtime_configuration_from_environment()


def set_runtime_configuration(runtime_configuration: RuntimeConfiguration) -> None:
    global _runtime_configuration
    _runtime_configuration = runtime_configuration
