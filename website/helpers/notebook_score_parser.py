# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

from bs4 import BeautifulSoup
import requests

from oceanbench.core.rmsd import DEPTH_LABELS, VARIABLE_METADATA
from oceanbench.core.lagrangian_trajectory import LAGRANGIAN_LABEL, LAGRANGIAN_UNIT, LAGRANGIAN_CF_NAME

from helpers.type import ModelScore


METRICS = [
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

REFERENCES = [
    {"key": "reanalysis", "suffix": "glorys", "function_suffix": "compared_to_glorys_reanalysis(challenger_dataset)"},
    {"key": "analysis", "suffix": "glo12", "function_suffix": "compared_to_glo12_analysis(challenger_dataset)"},
]

METRIC_PATTERNS = {
    f"{metric['key']}_{reference['suffix']}": f"oceanbench.metrics.{metric['function']}_{reference['function_suffix']}"
    for metric in METRICS
    for reference in REFERENCES
}

DEPTH_VARIABLE_METRICS = {
    f"{metric['key']}_{reference['suffix']}" for metric in METRICS for reference in REFERENCES if metric["has_depths"]
}

METRIC_TITLES = {
    f"{metric['key']}_{reference['suffix']}": metric["title"] for metric in METRICS for reference in REFERENCES
}

SECTIONS = {
    reference["key"]: {
        "depth_metric": next(f"{metric['key']}_{reference['suffix']}" for metric in METRICS if metric["has_depths"]),
        "flat_metrics": [f"{metric['key']}_{reference['suffix']}" for metric in METRICS if not metric["has_depths"]],
    }
    for reference in REFERENCES
}

_LABEL_LOOKUP: dict[str, tuple[str, str]] = {}
for _cf_name, (_label, _unit) in VARIABLE_METADATA.items():
    _LABEL_LOOKUP[_label] = (_cf_name, _unit)
    for _depth in DEPTH_LABELS.values():
        if _label.startswith(f"{_depth} "):
            _LABEL_LOOKUP[_label.removeprefix(f"{_depth} ")] = (_cf_name, _unit)
_LABEL_LOOKUP[f"{LAGRANGIAN_LABEL} ({LAGRANGIAN_UNIT})"] = (LAGRANGIAN_CF_NAME, LAGRANGIAN_UNIT)


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
    metrics = {}
    for cell in raw_notebook["cells"]:
        source = _get_cell_source(cell)
        for metric_key, pattern in METRIC_PATTERNS.items():
            if pattern in source:
                html = _get_cell_html_output(cell)
                if html:
                    metrics[metric_key] = html
    return metrics


def _get_depth_and_variable(variable: str) -> tuple[str, str] | None:
    for depth_label in DEPTH_LABELS.values():
        capitalized = depth_label.capitalize()
        if variable.startswith(capitalized):
            return (capitalized, variable.removeprefix(capitalized + " "))


def _parse_html_table_rows(raw_table: str) -> list[dict]:
    soup = BeautifulSoup(raw_table, features="html.parser")
    thead = soup.find("thead")
    headers = [th.get_text(strip=True) for th in thead.find_all("th")]
    lead_days = [header.replace("Lead day ", "") for header in headers[1:]]

    tbody = soup.find("tbody")
    rows = []
    for row in tbody.find_all("tr"):
        label = row.find("th").get_text(strip=True)
        values = {}
        for index, table_cell in enumerate(row.find_all("td")):
            text = table_cell.get_text(strip=True)
            try:
                values[lead_days[index]] = float(text)
            except (ValueError, IndexError):
                values[lead_days[index]] = None
        rows.append({"label": label, "data": values})
    return rows


def _convert_depth_variable_table_to_model_score(raw_table: str, name: str) -> ModelScore:
    scores = {"name": name, "depths": {}}
    for row in _parse_html_table_rows(raw_table):
        result = _get_depth_and_variable(row["label"])
        if result is None:
            continue
        depth, variable = result
        if depth not in scores["depths"]:
            scores["depths"][depth] = {"variables": {}}
        cf_name, unit = _get_variable_metadata(variable)
        scores["depths"][depth]["variables"][variable] = {
            "cf_name": cf_name,
            "unit": unit,
            "data": row["data"],
        }
    return ModelScore.model_validate(scores)


def _convert_flat_table_to_model_score(raw_table: str, name: str) -> ModelScore:
    scores = {"name": name, "depths": {}}
    for row in _parse_html_table_rows(raw_table):
        label = row["label"]
        if "flat" not in scores["depths"]:
            scores["depths"]["flat"] = {"variables": {}}
        cf_name, unit = _get_variable_metadata(label)
        scores["depths"]["flat"]["variables"][label] = {
            "cf_name": cf_name,
            "unit": unit,
            "data": row["data"],
        }
    return ModelScore.model_validate(scores)


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
    scores = {}
    for metric_key, raw_table in metrics_html.items():
        if metric_key in DEPTH_VARIABLE_METRICS:
            scores[metric_key] = _convert_depth_variable_table_to_model_score(raw_table, name)
        else:
            scores[metric_key] = _convert_flat_table_to_model_score(raw_table, name)
    return scores
