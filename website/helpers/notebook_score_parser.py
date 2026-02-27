# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json
import re

from bs4 import BeautifulSoup
import requests

from helpers.type import ModelScore

_VARIABLE_LABEL_PATTERN = re.compile(r"^(.*?) \(([^)]+)\) \[([^\]]+)\](?:\{([^}]+)\})?$")
_LEAD_DAY_NUMBER_PATTERN = re.compile(r"(\d+)$")


def _parse_variable_label(label: str) -> tuple[str, str, str, str]:
    match = _VARIABLE_LABEL_PATTERN.match(label)
    if match:
        return match.group(1), match.group(2), match.group(3), match.group(4) or ""
    return label, "", "unknown", ""


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

_OBSERVATIONS_METRIC_KEY = "rmsd_variables_observations"
_OBSERVATIONS_METRICS = {_OBSERVATIONS_METRIC_KEY}

_METRIC_PATTERNS = {
    f"{metric['key']}_{reference['suffix']}": f"oceanbench.metrics.{metric['function']}_{reference['function_suffix']}"
    for metric in _METRICS
    for reference in _REFERENCES
} | {_OBSERVATIONS_METRIC_KEY: "oceanbench.metrics.rmsd_of_variables_compared_to_observations"}

_DEPTH_VARIABLE_METRICS = {
    f"{metric['key']}_{reference['suffix']}" for metric in _METRICS for reference in _REFERENCES if metric["has_depths"]
}

METRIC_TITLES = {
    f"{metric['key']}_{reference['suffix']}": metric["title"] for metric in _METRICS for reference in _REFERENCES
} | {_OBSERVATIONS_METRIC_KEY: "RMSD vs. Observations"}

SECTIONS = {
    reference["key"]: {
        "depth_metric": next(f"{metric['key']}_{reference['suffix']}" for metric in _METRICS if metric["has_depths"]),
        "flat_metrics": [f"{metric['key']}_{reference['suffix']}" for metric in _METRICS if not metric["has_depths"]],
    }
    for reference in _REFERENCES
} | {"observations": {"depth_metric": _OBSERVATIONS_METRIC_KEY, "flat_metrics": []}}

_OBSERVATIONS_VARIABLE_METADATA = {
    "temperature": ("°C", "sea_water_potential_temperature"),
    "surface temperature": ("°C", "sea_surface_temperature"),
    "salinity": ("PSU", "sea_water_salinity"),
    "surface height": ("m", "sea_surface_height_above_geoid"),
    "eastward velocity": ("m/s", "eastward_sea_water_velocity"),
    "northward velocity": ("m/s", "northward_sea_water_velocity"),
}


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


def _parse_cell_value(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _parse_html_table_row(row, lead_days: list[str]) -> dict:
    label = row.find("th").get_text(strip=True)
    values = {day: _parse_cell_value(cell.get_text(strip=True)) for day, cell in zip(lead_days, row.find_all("td"))}
    return {"label": label, "data": values}


def _extract_lead_day_number(header: str) -> str:
    match = _LEAD_DAY_NUMBER_PATTERN.search(header)
    return match.group(1) if match else header


def _parse_html_table_rows(raw_table: str) -> list[dict]:
    soup = BeautifulSoup(raw_table, features="html.parser")
    headers = [th.get_text(strip=True) for th in soup.find("thead").find_all("th")]
    lead_days = [_extract_lead_day_number(header) for header in headers[1:]]
    return [_parse_html_table_row(row, lead_days) for row in soup.find("tbody").find_all("tr")]


def _convert_depth_variable_table_to_model_score(raw_table: str, name: str) -> ModelScore:
    parsed_rows = [
        (depth, variable_name, cf_name, unit, row["data"])
        for row in _parse_html_table_rows(raw_table)
        for display_name, unit, cf_name, depth_label in [_parse_variable_label(row["label"])]
        if depth_label
        for depth in [depth_label.capitalize()]
        for variable_name in [display_name.removeprefix(depth + " ")]
    ]
    unique_depths = dict.fromkeys(depth for depth, _, _, _, _ in parsed_rows)
    depths = {
        depth: {
            "variables": {
                variable_name: {"cf_name": cf_name, "unit": unit, "data": data}
                for row_depth, variable_name, cf_name, unit, data in parsed_rows
                if row_depth == depth
            }
        }
        for depth in unique_depths
    }
    return ModelScore.model_validate({"name": name, "depths": depths})


def _convert_flat_table_to_model_score(raw_table: str, name: str) -> ModelScore:
    rows = _parse_html_table_rows(raw_table)
    variables = {
        display_name: {"cf_name": cf_name, "unit": unit, "data": row["data"]}
        for row in rows
        for display_name, unit, cf_name, _depth_label in [_parse_variable_label(row["label"])]
    }
    return ModelScore.model_validate({"name": name, "depths": {"flat": {"variables": variables}}})


def _convert_observations_table_to_model_score(raw_table: str, name: str) -> ModelScore | None:
    soup = BeautifulSoup(raw_table, features="html.parser")
    headers = [th.get_text(strip=True) for th in soup.find("thead").find_all("th")]

    if len(headers) < 4 or headers[1] == "Message":
        return None

    lead_day_headers = [h for h in headers if h.startswith("Lead day")]
    lead_days = [_extract_lead_day_number(h) for h in lead_day_headers]

    parsed_rows = [
        (
            cells[0].get_text(strip=True),
            cells[1].get_text(strip=True),
            {day: _parse_cell_value(cells[2 + i].get_text(strip=True)) for i, day in enumerate(lead_days)},
        )
        for row in soup.find("tbody").find_all("tr")
        for cells in [row.find_all("td")]
        if len(cells) >= 2 + len(lead_days)
    ]

    if not parsed_rows:
        return None

    unique_depths = dict.fromkeys(depth for _, depth, _ in parsed_rows)
    depths = {
        depth: {
            "variables": {
                variable_name: {
                    "unit": _OBSERVATIONS_VARIABLE_METADATA.get(variable_name, ("", "unknown"))[0],
                    "cf_name": _OBSERVATIONS_VARIABLE_METADATA.get(variable_name, ("", "unknown"))[1],
                    "data": data,
                }
                for row_variable, row_depth, data in parsed_rows
                for variable_name in [row_variable]
                if row_depth == depth
            }
        }
        for depth in unique_depths
    }
    return ModelScore.model_validate({"name": name, "depths": depths})


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
        if key in _OBSERVATIONS_METRICS:
            return _convert_observations_table_to_model_score
        if key in _DEPTH_VARIABLE_METRICS:
            return _convert_depth_variable_table_to_model_score
        return _convert_flat_table_to_model_score

    return {
        metric_key: result
        for metric_key, raw_table in metrics_html.items()
        if (result := _converter(metric_key)(raw_table, name)) is not None
    }
