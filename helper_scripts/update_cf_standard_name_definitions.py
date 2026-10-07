# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
from pathlib import Path
from urllib.request import urlopen
from xml.etree import ElementTree

from oceanbench.core.climate_forecast_standard_names import StandardVariable

CLIMATE_FORECAST_TABLE_URL = "https://cfconventions.org/Data/cf-standard-names/95/src/cf-standard-name-table.xml"
OUTPUT_PATH = Path(__file__).parent.parent / "website" / "cf_standard_name_definitions.json"


def _definitions_by_standard_name(table_xml: bytes) -> dict[str, str]:
    return {
        entry.get("id"): " ".join((entry.findtext("description") or "").split())
        for entry in ElementTree.fromstring(table_xml).findall("entry")
    }


def main() -> None:
    with urlopen(CLIMATE_FORECAST_TABLE_URL) as response:
        definitions = _definitions_by_standard_name(response.read())
    selected = {variable.value: definitions[variable.value] for variable in StandardVariable}
    OUTPUT_PATH.write_text(json.dumps(selected, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
