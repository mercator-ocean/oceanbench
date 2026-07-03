# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Load the repository JSON Schemas and validate artifacts against them.

The schemas under ``schemas/`` are the contract (contracts.md §4-6). Every writer
validates the artifact it produces before emitting it and refuses to write an
invalid one. This module locates the schema directory by walking up from this
file to the repository root; callers may override the directory explicitly.
"""

from functools import lru_cache
import json
from pathlib import Path

import jsonschema


def _repository_schema_directory() -> Path:
    for ancestor in Path(__file__).resolve().parents:
        candidate = ancestor / "schemas"
        if (candidate / "catalog.schema.json").exists():
            return candidate
    raise FileNotFoundError("Could not locate the repository schemas/ directory.")


@lru_cache(maxsize=None)
def load_schema(schema_name: str, schema_directory: str | None = None) -> dict:
    """Load ``<schema_name>.schema.json`` from the repository schema directory."""
    directory = Path(schema_directory) if schema_directory is not None else _repository_schema_directory()
    return json.loads((directory / f"{schema_name}.schema.json").read_text(encoding="utf-8"))


def validate_against_schema(instance: object, schema_name: str, schema_directory: str | None = None) -> None:
    """Validate ``instance`` against a named schema, raising on the first violation."""
    jsonschema.validate(instance=instance, schema=load_schema(schema_name, schema_directory))
