# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import shutil
from os import environ
from pathlib import Path

LOCAL_STAGE_ENVIRONMENT_VARIABLE = "OCEANBENCH_LOCAL_STAGE"
LOCAL_STAGE_DIRECTORY_ENVIRONMENT_VARIABLE = "OCEANBENCH_LOCAL_STAGE_DIRECTORY"
LOCAL_STAGE_CLEANUP_ENVIRONMENT_VARIABLE = "OCEANBENCH_LOCAL_STAGE_CLEANUP"
LOCAL_STAGE_ALL_KEY = "all"
LOCAL_STAGE_CLEANUP_NEVER = "never"
LOCAL_STAGE_CLEANUP_SUCCESS = "success"
LOCAL_STAGE_CLEANUP_ALWAYS = "always"


def local_stage_keys() -> set[str]:
    raw_value = environ.get(LOCAL_STAGE_ENVIRONMENT_VARIABLE, "")
    return {key.strip().lower() for key in raw_value.split(",") if key.strip()}


def has_local_stage_configuration() -> bool:
    return bool(local_stage_keys())


def should_stage_locally(stage_key: str) -> bool:
    configured_keys = local_stage_keys()
    return stage_key.lower() in configured_keys or LOCAL_STAGE_ALL_KEY in configured_keys


def local_stage_directory() -> Path:
    default_stage_directory = Path(environ.get("TMPDIR", "/tmp")) / "oceanbench-stage"
    return Path(environ.get(LOCAL_STAGE_DIRECTORY_ENVIRONMENT_VARIABLE, str(default_stage_directory)))


def local_stage_cleanup_policy() -> str:
    raw_value = environ.get(LOCAL_STAGE_CLEANUP_ENVIRONMENT_VARIABLE, LOCAL_STAGE_CLEANUP_NEVER).strip().lower()
    cleanup_policy = raw_value or LOCAL_STAGE_CLEANUP_NEVER
    if cleanup_policy not in {
        LOCAL_STAGE_CLEANUP_NEVER,
        LOCAL_STAGE_CLEANUP_SUCCESS,
        LOCAL_STAGE_CLEANUP_ALWAYS,
    }:
        raise ValueError(
            f"Unsupported {LOCAL_STAGE_CLEANUP_ENVIRONMENT_VARIABLE} value: {cleanup_policy!r}. "
            f"Expected one of: {LOCAL_STAGE_CLEANUP_NEVER!r}, {LOCAL_STAGE_CLEANUP_SUCCESS!r}, "
            f"{LOCAL_STAGE_CLEANUP_ALWAYS!r}."
        )
    return cleanup_policy


def should_cleanup_local_stage_directory(operation_succeeded: bool) -> bool:
    cleanup_policy = local_stage_cleanup_policy()
    return cleanup_policy == LOCAL_STAGE_CLEANUP_ALWAYS or (
        cleanup_policy == LOCAL_STAGE_CLEANUP_SUCCESS and operation_succeeded
    )


def cleanup_local_stage_directory() -> None:
    shutil.rmtree(local_stage_directory(), ignore_errors=True)
