# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

from bs4 import BeautifulSoup
import requests

from oceanbench.core.rmsd import DEPTH_LABELS, VARIABLE_METADATA
from oceanbench.core.lagrangian_trajectory import LAGRANGIAN_LABEL, LAGRANGIAN_UNIT, LAGRANGIAN_CF_NAME
from oceanbench.core.lead_day_utils import LEAD_DAY_LABEL_PREFIX

from helpers.type import ModelScore


_METRICS = [
    {"key": "rmsd_variables", "title": "RMSD of Variables", "function": "rmsd_of_variables", "has_depths": True},
    {
        "key": "rmsd_mld",
        "title": "RMSD of Mixed Layer Depth",
        "function": "rmsd_of_mixed_layer_depth",
        "has_depths": False,
    },
    {
        "key": "rmsd_geostrophic",
        "title": "RMSD of Geostrophic Currents",
        "function": "rmsd_of_geostrophic_currents",
        "has_depths": False,
    },
    {
        "key": "lagrangian",
        "title": "Lagrangian Trajectory Deviation",
        "function": "deviation_of_lagrangian_trajectories",
        "has_depths": False,
    },
]

_REFERENCES = [
    {"key": "reanalysis", "suffix": "glorys", "function_suffix": "compared_to_glorys_reanalysis"},
    {"key": "analysis", "suffix": "glo12", "function_suffix": "compared_to_glo12_analysis"},
]

_METRIC_PATTERNS = {
    f"{metric['key']}_{reference['suffix']}": f"oceanbench.metrics.{metric['function']}_{reference['function_suffix']}"
    for metric in _METRICS
    for reference in _REFERENCES
}

_DEPTH_VARIABLE_METRICS = {
    f"{metric['key']}_{reference['suffix']}" for metric in _METRICS for reference in _REFERENCES if metric["has_depths"]
}

METRIC_TITLES = {
    f"{metric['key']}_{reference['suffix']}": metric["title"] for metric in _METRICS for reference in _REFERENCES
}

SECTIONS = {
    reference["key"]: {
        "depth_metric": next(f"{metric['key']}_{reference['suffix']}" for metric in _METRICS if metric["has_depths"]),
        "flat_metrics": [f"{metric['key']}_{reference['suffix']}" for metric in _METRICS if not metric["has_depths"]],
    }
    for reference in _REFERENCES
}

_LABEL_LOOKUP: dict[str, tuple[str, str]] = {
    **{label: (cf_name, unit) for cf_name, (label, unit) in VARIABLE_METADATA.items()},
    **{
        label.removeprefix(f"{depth} "): (cf_name, unit)
        for cf_name, (label, unit) in VARIABLE_METADATA.items()
        for depth in DEPTH_LABELS.values()
        if label.startswith(f"{depth} ")
    },
    LAGRANGIAN_LABEL: (LAGRANGIAN_CF_NAME, LAGRANGIAN_UNIT),
    f"{LAGRANGIAN_LABEL} ({LAGRANGIAN_UNIT})": (LAGRANGIAN_CF_NAME, LAGRANGIAN_UNIT),
}


def _get_variable_metadata(variable_name: str) -> tuple[str, str]:
    result = _LABEL_LOOKUP.get(variable_name) or _LABEL_LOOKUP.get(variable_name.lower())
    if result:
        return result
    return ("unknown", "")


def _get_cell_source(cell: dict) -> str:
    source = cell.get("source", [])
    if isinstance(source, list):
        return "".join(source)
    return source


def _get_cell_html_output(cell: dict) -> str | None:
    for output in cell.get("outputs", []):
        if "data" in output and "text/html" in output["data"]:
            html_parts = output["data"]["text/html"]
            if isinstance(html_parts, list):
                return "".join(line.removesuffix("\n") for line in html_parts)
            return html_parts
    return None


def _get_all_metrics_from_notebook(raw_notebook: dict) -> dict[str, str]:
    return {
        metric_key: html
        for cell in raw_notebook["cells"]
        for metric_key, pattern in _METRIC_PATTERNS.items()
        if pattern in _get_cell_source(cell)
        if (html := _get_cell_html_output(cell))
    }


def _get_depth_and_variable(variable: str) -> tuple[str, str] | None:
    for depth_label in DEPTH_LABELS.values():
        capitalized = depth_label.capitalize()
        if variable.startswith(capitalized):
            return (capitalized, variable.removeprefix(capitalized + " "))


def _parse_cell_value(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _parse_html_table_row(row, lead_days: list[str]) -> dict:
    label = row.find("th").get_text(strip=True)
    values = {day: _parse_cell_value(cell.get_text(strip=True)) for day, cell in zip(lead_days, row.find_all("td"))}
    return {"label": label, "data": values}


def _parse_html_table_rows(raw_table: str) -> list[dict]:
    soup = BeautifulSoup(raw_table, features="html.parser")
    headers = [th.get_text(strip=True) for th in soup.find("thead").find_all("th")]
    lead_days = [header.removeprefix(LEAD_DAY_LABEL_PREFIX) for header in headers[1:]]
    return [_parse_html_table_row(row, lead_days) for row in soup.find("tbody").find_all("tr")]


def _convert_depth_variable_table_to_model_score(raw_table: str, name: str) -> ModelScore:
    parsed_rows = [
        (depth, variable, _get_variable_metadata(variable), row["data"])
        for row in _parse_html_table_rows(raw_table)
        if (result := _get_depth_and_variable(row["label"])) is not None
        for depth, variable in [result]
    ]
    unique_depths = dict.fromkeys(depth for depth, _, _, _ in parsed_rows)
    depths = {
        depth: {
            "variables": {
                variable: {"cf_name": metadata[0], "unit": metadata[1], "data": data}
                for row_depth, variable, metadata, data in parsed_rows
                if row_depth == depth
            }
        }
        for depth in unique_depths
    }
    return ModelScore.model_validate({"name": name, "depths": depths})


def _convert_flat_table_to_model_score(raw_table: str, name: str) -> ModelScore:
    rows = _parse_html_table_rows(raw_table)
    variables = {
        row["label"]: {"cf_name": cf_name, "unit": unit, "data": row["data"]}
        for row in rows
        for cf_name, unit in [_get_variable_metadata(row["label"])]
    }
    return ModelScore.model_validate({"name": name, "depths": {"flat": {"variables": variables}}})


def _get_notebook(path: str) -> dict | None:
    if path.startswith("http"):
        response = requests.get(path)
        if response.status_code == 200:
            return response.json()
    else:
        with open(path) as file:
            return json.load(file)


def get_all_model_scores_from_notebook(notebook_path: str, name: str) -> dict[str, ModelScore]:
    raw_notebook = _get_notebook(notebook_path)
    if raw_notebook is None:
        return {}
    metrics_html = _get_all_metrics_from_notebook(raw_notebook)

    def _converter(key):
        return (
            _convert_depth_variable_table_to_model_score
            if key in _DEPTH_VARIABLE_METRICS
            else _convert_flat_table_to_model_score
        )

    return {metric_key: _converter(metric_key)(raw_table, name) for metric_key, raw_table in metrics_html.items()}
