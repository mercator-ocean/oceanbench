# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Convert the score tables returned by the ``oceanbench.metrics`` functions into a scores file.

Each table has one row per variable and depth, labelled ``"{Display name} ({unit}) [{standard_name}]{{depth}}"``,
one ``"Lead day N"`` column per lead day, and optionally ``"Missing fraction"`` (gridded RMSD) or
``"Observations"`` and ``"Missing"`` (Class IV) columns. The scores file stores one row per number.
A metric that could not be computed returns a one-column ``"Message"`` table instead and has no rows.
"""

from datetime import datetime, timezone
import json
import math
import re

import pandas
import xarray

from oceanbench.core.classIV_support import MISSING_COUNT_COLUMN, OBSERVATION_COUNT_COLUMN
from oceanbench.core.dataset_utils import Dimension
from oceanbench.core.metrics import MESSAGE_COLUMN
from oceanbench.core.regions import RegionSpec, region_to_dict
from oceanbench.core.rmsd import MISSING_FRACTION_COLUMN
from oceanbench.core.version import __version__

METRIC_FUNCTION_NAMES = [
    "rmsd_of_variables_compared_to_glorys_reanalysis",
    "rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis",
    "rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis",
    "rmsd_of_variables_compared_to_observations",
    "deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis",
    "rmsd_of_variables_compared_to_glo12_analysis",
    "rmsd_of_mixed_layer_depth_compared_to_glo12_analysis",
    "rmsd_of_geostrophic_currents_compared_to_glo12_analysis",
    "deviation_of_lagrangian_trajectories_compared_to_glo12_analysis",
]

_LABEL_PATTERN = re.compile(r"^(.*?) \(([^)]*)\) \[([^\]]*)\]\{([^}]+)\}$")
_LEAD_DAY_PATTERN = re.compile(r"^Lead day (\d+)$")
_ROW_COLUMN_FIELDS = {
    MISSING_FRACTION_COLUMN: ("missing_fraction", float),
    OBSERVATION_COUNT_COLUMN: ("observations", int),
    MISSING_COUNT_COLUMN: ("missing", int),
}


def scores_file_name(challenger_name: str, region: RegionSpec) -> str:
    return f"{challenger_name}.{region.id}.scores.json"


def _json_number(value, number_type: type) -> float | int | None:
    return None if math.isnan(value) else number_type(value)


def score_rows(metric_function_name: str, table: pandas.DataFrame) -> list[dict]:
    if MESSAGE_COLUMN in table.columns:
        return []
    metric, reference = metric_function_name.split("_compared_to_")
    lead_day_columns = {
        column: int(match.group(1)) for column in table.columns if (match := _LEAD_DAY_PATTERN.match(column))
    }
    rows = []
    for label in table.index:
        display_name, unit, standard_name, depth = _LABEL_PATTERN.match(label).groups()
        row_fields = {
            field: _json_number(table.at[label, column], number_type)
            for column, (field, number_type) in _ROW_COLUMN_FIELDS.items()
            if column in table.columns
        }
        rows += [
            {
                "metric": metric,
                "reference": reference,
                "variable": display_name.lower(),
                "standard_name": standard_name or None,
                "depth": depth,
                "unit": unit,
                "lead_day": lead_day,
                "value": _json_number(table.at[label, column], float),
            }
            | row_fields
            for column, lead_day in lead_day_columns.items()
        ]
    return rows


def scores_document(
    challenger_name: str,
    challenger_python_code: str,
    challenger_dataset: xarray.Dataset,
    region: RegionSpec,
    tables: dict[str, pandas.DataFrame],
    timings: dict[str, float],
) -> dict:
    return {
        "oceanbench_version": __version__,
        "challenger": challenger_name,
        "region": region_to_dict(region),
        "start_dates": [
            pandas.Timestamp(start_date).date().isoformat()
            for start_date in challenger_dataset[Dimension.FIRST_DAY_DATETIME.key()].values
        ],
        "lead_days_count": challenger_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()],
        "challenger_code": challenger_python_code,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "timings": timings,
        "rows": [
            row for metric_function_name, table in tables.items() for row in score_rows(metric_function_name, table)
        ],
    }


def write_scores_file(document: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as file:
        json.dump(document, file, indent=2)
        file.write("\n")
