# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import tomllib


def _version_from_pyproject() -> str:
    pyproject_file = Path(__file__).resolve().parents[2] / "pyproject.toml"
    with pyproject_file.open("rb") as file:
        return tomllib.load(file)["project"]["version"]


def _resolve_version() -> str:
    try:
        return version("oceanbench")
    except PackageNotFoundError:
        return _version_from_pyproject()


__version__ = _resolve_version()
