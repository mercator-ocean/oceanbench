# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from bs4 import BeautifulSoup
import requests
from helpers.type import ModelScore
import json


def get_raw_html_report_score_table(raw_notebook) -> str:
    for cell in raw_notebook["cells"]:
        if (
            "oceanbench.metrics.rmsd_of_variables_compared_to_glorys(challenger_datasets)"
            in cell["source"]
        ):
            html_output = cell["outputs"][0]["data"]["text/html"]
            cleaned_html_output = "".join(
                [line.removesuffix("\n") for line in html_output]
            )
            return cleaned_html_output


def _get_depth_and_variable(variable: str) -> tuple[str, str]:
    for depth in ["Surface", "50m", "100m", "200m", "300m", "500m"]:
        if variable.startswith(depth):
            return (depth, variable.removeprefix(depth + " "))


def _convert_raw_html_report_score_table_to_model_score(
    raw_table: str, name: str
) -> ModelScore:
    scores = {"name": name, "depths": {}}
    soup = BeautifulSoup(raw_table, features="html.parser")
    tbody = soup.find("tbody")
    rows = tbody.find_all("tr")
    for row in rows:
        depth, variable = _get_depth_and_variable(row.find("th").string)
        if depth not in scores["depths"]:
            scores["depths"][depth] = {"real_value": 0, "variables": {}}
        scores["depths"][depth]["variables"][variable] = {
            "cf_name": "TODO",
            "unit": "TODO",
            "data": {
                str(k + 1): float(v.string) for k, v in enumerate(row.find_all("td"))
            },
        }
    return ModelScore.model_validate(scores)


def _get_notebook(path: str):
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


def get_model_score_from_file(file_path: str) -> ModelScore:
    with open(file_path) as file:
        return ModelScore.model_validate(json.load(file))
