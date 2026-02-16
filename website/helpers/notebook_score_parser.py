# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json

from bs4 import BeautifulSoup
import requests

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


def get_raw_html_report_score_table(raw_notebook: dict) -> str | None:
    for cell in raw_notebook["cells"]:
        if METRIC_PATTERNS["rmsd_variables_glorys"] in _get_cell_source(cell):
            return _get_cell_html_output(cell)


def get_all_metrics_from_notebook(raw_notebook: dict) -> dict[str, str]:
    metrics = {}
    for cell in raw_notebook["cells"]:
        source = _get_cell_source(cell)
        for metric_key, pattern in METRIC_PATTERNS.items():
            if pattern in source:
                html = _get_cell_html_output(cell)
                if html:
                    metrics[metric_key] = html
    return metrics


def _get_depth_and_variable(variable: str) -> tuple[str, str]:
    for depth in ["Surface", "50m", "100m", "200m", "300m", "500m"]:
        if variable.startswith(depth):
            return (depth, variable.removeprefix(depth + " "))


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
        depth, variable = _get_depth_and_variable(row["label"])
        if depth not in scores["depths"]:
            scores["depths"][depth] = {"real_value": 0, "variables": {}}
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
            scores["depths"]["flat"] = {"real_value": 0, "variables": {}}
        cf_name, unit = _get_variable_metadata(label)
        scores["depths"]["flat"]["variables"][label] = {
            "cf_name": cf_name,
            "unit": unit,
            "data": row["data"],
        }
    return ModelScore.model_validate(scores)


def _convert_raw_html_report_score_table_to_model_score(raw_table: str, name: str) -> ModelScore:
    scores = {"name": name, "depths": {}}
    soup = BeautifulSoup(raw_table, features="html.parser")
    tbody = soup.find("tbody")
    rows = tbody.find_all("tr")
    for row in rows:
        depth, variable = _get_depth_and_variable(row.find("th").string)
        if depth not in scores["depths"]:
            scores["depths"][depth] = {"real_value": 0, "variables": {}}
        cf_name, unit = _get_variable_metadata(variable)
        scores["depths"][depth]["variables"][variable] = {
            "cf_name": cf_name,
            "unit": unit,
            "data": {str(index + 1): float(cell.string) for index, cell in enumerate(row.find_all("td"))},
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


def get_model_score_from_notebook(notebook_path: str, name: str) -> ModelScore:
    raw_report = _get_notebook(notebook_path)
    score_table = get_raw_html_report_score_table(raw_report)
    model_score = _convert_raw_html_report_score_table_to_model_score(score_table, name)
    return model_score


def get_all_model_scores_from_notebook(notebook_path: str, name: str) -> dict[str, ModelScore]:
    raw_notebook = _get_notebook(notebook_path)
    if raw_notebook is None:
        return {}
    metrics_html = get_all_metrics_from_notebook(raw_notebook)
    scores = {}
    for metric_key, raw_table in metrics_html.items():
        if metric_key in DEPTH_VARIABLE_METRICS:
            scores[metric_key] = _convert_depth_variable_table_to_model_score(raw_table, name)
        else:
            scores[metric_key] = _convert_flat_table_to_model_score(raw_table, name)
    return scores


def get_model_score_from_file(file_path: str) -> ModelScore:
    with open(file_path) as file:
        return ModelScore.model_validate(json.load(file))
