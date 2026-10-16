# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from functools import lru_cache
import json
from pathlib import Path

import jsonschema

SCHEMA_DIRECTORY = Path(__file__).resolve().parent.parent / "schemas"


@lru_cache(maxsize=None)
def load_schema(schema_name: str) -> dict:
    return json.loads((SCHEMA_DIRECTORY / f"{schema_name}.schema.json").read_text(encoding="utf-8"))


def validate_against_schema(instance: object, schema_name: str) -> None:
    jsonschema.validate(instance=instance, schema=load_schema(schema_name))
