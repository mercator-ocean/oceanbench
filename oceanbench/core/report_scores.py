# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import re

import numpy
import pandas

VARIABLE_LABEL_PATTERN = re.compile(r"^(.*?) \(([^)]+)\) \[([^\]]*)\](?:\{([^}]+)\})?$")
LEAD_DAY_NUMBER_PATTERN = re.compile(r"(\d+)$")
DISPLAY_NAME_RENAMES = {
    "height": "sea surface height",
    "surface height": "sea surface height",
    "northward velocity": "meridional current",
    "eastward velocity": "zonal current",
    "northward geostrophic velocity": "meridional geostrophic current",
    "eastward geostrophic velocity": "zonal geostrophic current",
}


def _parse_variable_label(label: str) -> tuple[str, str, str, str]:
    match = VARIABLE_LABEL_PATTERN.match(label)
    if match:
        return match.group(1), match.group(2), match.group(3), match.group(4) or ""
    return label, "", "unknown", ""


def _normalise_display_name(display_name: str) -> str:
    normalised_display_name = display_name.lower()
    return DISPLAY_NAME_RENAMES.get(normalised_display_name, normalised_display_name)


def _normalise_depth_label(depth_label: str) -> str:
    return depth_label.capitalize() if depth_label else "flat"


def _lead_day_key(column: object) -> str | None:
    match = LEAD_DAY_NUMBER_PATTERN.search(str(column))
    return match.group(1) if match else None


def _score_value(value) -> float | None:
    if pandas.isna(value):
        return None
    if isinstance(value, numpy.generic):
        value = value.item()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _score_row_data(row: pandas.Series, lead_day_columns: list[object]) -> dict[str, float | None]:
    return {
        lead_day: _score_value(row[column])
        for column in lead_day_columns
        if (lead_day := _lead_day_key(column)) is not None
    }


def dataframe_to_model_score(
    dataframe: pandas.DataFrame,
    challenger_name: str,
    *,
    has_depths: bool,
) -> dict | None:
    if dataframe.empty:
        return None
    lead_day_columns = [column for column in dataframe.columns if _lead_day_key(column) is not None]
    if not lead_day_columns:
        return None

    depths: dict[str, dict] = {}
    for label, row in dataframe.iterrows():
        display_name, unit, standard_name, depth_label = _parse_variable_label(str(label))
        depth = _normalise_depth_label(depth_label) if has_depths else "flat"
        variable_name = _normalise_display_name(display_name.removeprefix(f"{depth} "))
        depths.setdefault(depth, {"variables": {}})["variables"][variable_name] = {
            "standard_name": standard_name,
            "unit": unit,
            "data": _score_row_data(row, lead_day_columns),
        }

    return {"name": challenger_name, "depths": depths}


def model_scores_from_dataframes(
    score_dataframes: dict[str, pandas.DataFrame],
    challenger_name: str,
    *,
    depth_variable_metric_keys: set[str],
) -> dict[str, dict]:
    return {
        metric_key: score
        for metric_key, dataframe in score_dataframes.items()
        if (
            score := dataframe_to_model_score(
                dataframe,
                challenger_name,
                has_depths=metric_key in depth_variable_metric_keys,
            )
        )
        is not None
    }
