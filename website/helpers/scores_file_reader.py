# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

from helpers.notebook_score_parser import _DEPTH_VARIABLE_METRICS, _METRIC_PATTERNS
from helpers.type import ModelScore

# A scores file row names its metric function by "metric" and "reference", the notebook cell by its call.
_METRIC_KEYS = {
    pattern.removeprefix("oceanbench.metrics."): metric_key for metric_key, pattern in _METRIC_PATTERNS.items()
}


def _depth_key(metric_key: str, depth: str) -> str:
    return depth.capitalize() if metric_key in _DEPTH_VARIABLE_METRICS else "flat"


def get_all_model_scores_from_scores_file(scores_file_path: str, challenger_name: str) -> dict[str, ModelScore]:
    """Read a scores file into the same scores the notebook parser reads from a report notebook."""
    with open(scores_file_path, encoding="utf-8") as file:
        scores = json.load(file)
    # A metric that was run but has no rows, like Class IV without observations, still gets an empty score,
    # as its empty table does in a report notebook.
    depths_by_metric = {
        _METRIC_KEYS[function_name]: {} for function_name in scores["timings"] if function_name in _METRIC_KEYS
    }
    for row in scores["rows"]:
        metric_key = _METRIC_KEYS.get(f"{row['metric']}_compared_to_{row['reference']}")
        if metric_key is None:
            continue
        depth = depths_by_metric.setdefault(metric_key, {}).setdefault(
            _depth_key(metric_key, row["depth"]), {"variables": {}}
        )
        variable = depth["variables"].setdefault(
            row["variable"], {"standard_name": row["standard_name"] or "", "unit": row["unit"], "data": {}}
        )
        variable["data"][str(row["lead_day"])] = row["value"]
    return {
        metric_key: ModelScore.model_validate({"name": challenger_name, "depths": depths})
        for metric_key, depths in depths_by_metric.items()
    }
