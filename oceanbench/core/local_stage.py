# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from os import environ
from pathlib import Path

LOCAL_STAGE_ENVIRONMENT_VARIABLE = "OCEANBENCH_LOCAL_STAGE"
LOCAL_STAGE_DIRECTORY_ENVIRONMENT_VARIABLE = "OCEANBENCH_LOCAL_STAGE_DIRECTORY"
LOCAL_STAGE_ALL_KEY = "all"


def local_stage_keys() -> set[str]:
    raw_value = environ.get(LOCAL_STAGE_ENVIRONMENT_VARIABLE, "")
    return {key.strip().lower() for key in raw_value.split(",") if key.strip()}


def should_stage_locally(stage_key: str) -> bool:
    configured_keys = local_stage_keys()
    return stage_key.lower() in configured_keys or LOCAL_STAGE_ALL_KEY in configured_keys


def local_stage_directory() -> Path:
    default_stage_directory = Path(environ.get("TMPDIR", "/tmp")) / "oceanbench-stage"
    return Path(environ.get(LOCAL_STAGE_DIRECTORY_ENVIRONMENT_VARIABLE, str(default_stage_directory)))
