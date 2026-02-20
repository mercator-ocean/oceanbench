# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

from bs4 import BeautifulSoup
import requests

from oceanbench.core.rmsd import DEPTH_LABELS

from helpers.type import ModelScore


_GLORYS = "compared_to_glorys_reanalysis(challenger_dataset)"
_GLO12 = "compared_to_glo12_analysis(challenger_dataset)"

METRIC_PATTERNS = {
    "rmsd_variables_glorys": f"oceanbench.metrics.rmsd_of_variables_{_GLORYS}",
    "rmsd_mld_glorys": f"oceanbench.metrics.rmsd_of_mixed_layer_depth_{_GLORYS}",
    "rmsd_geostrophic_glorys": f"oceanbench.metrics.rmsd_of_geostrophic_currents_{_GLORYS}",
    "lagrangian_glorys": f"oceanbench.metrics.deviation_of_lagrangian_trajectories_{_GLORYS}",
    "rmsd_variables_glo12": f"oceanbench.metrics.rmsd_of_variables_{_GLO12}",
    "rmsd_mld_glo12": f"oceanbench.metrics.rmsd_of_mixed_layer_depth_{_GLO12}",
    "rmsd_geostrophic_glo12": f"oceanbench.metrics.rmsd_of_geostrophic_currents_{_GLO12}",
    "lagrangian_glo12": f"oceanbench.metrics.deviation_of_lagrangian_trajectories_{_GLO12}",
}

DEPTH_VARIABLE_METRICS = {"rmsd_variables_glorys", "rmsd_variables_glo12"}

VARIABLE_UNITS = {
    "height": ("sea_surface_height_above_geoid", "m"),
    "temperature": ("sea_water_potential_temperature", "°C"),
    "salinity": ("sea_water_salinity", "PSU"),
    "northward velocity": ("northward_sea_water_velocity", "m/s"),
    "eastward velocity": ("eastward_sea_water_velocity", "m/s"),
    "Mixed layer depth": ("ocean_mixed_layer_thickness", "m"),
    "Northward geostrophic velocity": ("northward_geostrophic_velocity", "m/s"),
    "Eastward geostrophic velocity": ("eastward_geostrophic_velocity", "m/s"),
    "Surface Lagrangian trajectory deviation (km)": ("lagrangian_trajectory_deviation", "km"),
}


def _get_variable_metadata(variable_name: str) -> tuple[str, str]:
    if variable_name in VARIABLE_UNITS:
        return VARIABLE_UNITS[variable_name]
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
