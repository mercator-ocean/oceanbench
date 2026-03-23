# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from io import StringIO
import json

import pandas
import requests

from helpers.type import ModelScore


TRACK_METRIC_CALLS = {
    "glorys_reanalysis": [
        "oceanbench.metrics.rmsd_of_variables_compared_to_glorys_reanalysis(",
        "oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(",
        "oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(",
    ],
    "glo12_analysis": [
        "oceanbench.metrics.rmsd_of_variables_compared_to_glo12_analysis(",
        "oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(",
        "oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glo12_analysis(",
    ],
    "observations": [
        "oceanbench.metrics.rmsd_of_variables_compared_to_observations(",
    ],
}

LEGACY_TRACK_METRIC_CALLS = {
    "glorys_reanalysis": [
        "oceanbench.metrics.rmsd_of_variables_compared_to_glorys_reanalysis(",
        "oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(",
        "oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(",
    ],
    "glo12_analysis": [
        "oceanbench.metrics.rmsd_of_variables_compared_to_glo12_analysis(",
        "oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(",
        "oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glo12_analysis(",
    ],
    "observations": [
        "oceanbench.metrics.rmsd_of_variables_compared_to_observations(",
    ],
}

SURFACE_DEPTH_LABEL = "Surface"


def _get_depth_and_variable(variable: str) -> tuple[str, str]:
    normalized_variable = variable.strip()
    for depth in [SURFACE_DEPTH_LABEL, "50m", "100m", "200m", "300m", "500m"]:
        if normalized_variable.startswith(depth):
            return depth, normalized_variable.removeprefix(depth).strip().lower()
    return SURFACE_DEPTH_LABEL, normalized_variable.lower()


def _load_dataframe_from_html(raw_table: str) -> pandas.DataFrame:
    return pandas.read_html(StringIO(raw_table))[0]


def _get_lead_day_data(row: pandas.Series) -> dict[str, float]:
    lead_day_data = {}
    for column_name, value in row.items():
        if not str(column_name).startswith("Lead day "):
            continue
        lead_day_index = column_name.removeprefix("Lead day ").strip()
        lead_day_data[lead_day_index] = float(value)
    return lead_day_data


def _convert_forecast_dataframe_to_model_score_fragment(
    dataframe: pandas.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    score_fragment: dict[str, dict[str, dict[str, float]]] = {}
    for _, row in dataframe.iterrows():
        depth, variable = _get_depth_and_variable(str(row.iloc[0]))
        if depth not in score_fragment:
            score_fragment[depth] = {}
        score_fragment[depth][variable] = _get_lead_day_data(row)
    return score_fragment


def _normalise_observation_depth_label(depth_label: str) -> str:
    stripped_depth_label = depth_label.strip()
    return SURFACE_DEPTH_LABEL if stripped_depth_label.lower() == "surface" else stripped_depth_label


def _convert_observation_dataframe_to_model_score_fragment(
    dataframe: pandas.DataFrame,
) -> dict[str, dict[str, dict[str, float]]]:
    score_fragment: dict[str, dict[str, dict[str, float]]] = {}
    for _, row in dataframe.iterrows():
        depth = _normalise_observation_depth_label(str(row["Depth Range"]))
        variable = str(row["Variable"]).strip().lower()
        if depth not in score_fragment:
            score_fragment[depth] = {}
        score_fragment[depth][variable] = _get_lead_day_data(row)
    return score_fragment


def _merge_model_score_fragments(
    name: str,
    fragments: list[dict[str, dict[str, dict[str, float]]]],
) -> ModelScore:
    scores = {"name": name, "depths": {}}
    for fragment in fragments:
        for depth, variables in fragment.items():
            if depth not in scores["depths"]:
                scores["depths"][depth] = {"real_value": 0, "variables": {}}
            scores["depths"][depth]["variables"].update(
                {
                    variable: {
                        "cf_name": "TODO",
                        "unit": "TODO",
                        "data": data,
                    }
                    for variable, data in variables.items()
                }
            )
    return ModelScore.model_validate(scores)


def _get_cell_source(cell: dict) -> str:
    source = cell.get("source", "")
    return "".join(source) if isinstance(source, list) else source


def _get_raw_html_report_score_table(
    raw_notebook: dict,
    expected_source: str,
) -> str:
    for cell in raw_notebook["cells"]:
        if _normalise_source(_get_cell_source(cell)).find(_normalise_source(expected_source)) == -1:
            continue
        return "".join([line.removesuffix("\n") for line in cell["outputs"][0]["data"]["text/html"]])
    raise ValueError(f"Notebook does not contain the expected score cell: {expected_source}")


def _normalise_source(source: str) -> str:
    return "".join(source.split())


def _get_track_metric_calls(track: str) -> list[str]:
    if track in TRACK_METRIC_CALLS:
        return TRACK_METRIC_CALLS[track]
    supported_tracks = ", ".join(TRACK_METRIC_CALLS.keys())
    raise ValueError(f"Unsupported score track: {track}. Supported values are: {supported_tracks}.")


def _convert_raw_html_report_score_tables_to_model_score(
    raw_tables: list[str],
    name: str,
    track: str,
) -> ModelScore:
    if track == "observations":
        fragments = [
            _convert_observation_dataframe_to_model_score_fragment(_load_dataframe_from_html(raw_tables[0])),
        ]
    else:
        fragments = [
            _convert_forecast_dataframe_to_model_score_fragment(_load_dataframe_from_html(raw_table))
            for raw_table in raw_tables
        ]
    return _merge_model_score_fragments(name, fragments)


def _get_notebook(path: str):
    if path.startswith("http"):
        response = requests.get(path)
        if response.status_code == 200:
            return response.json()
        raise ValueError(f"Failed to read notebook at {path}: HTTP {response.status_code}")
    with open(path) as file:
        return json.load(file)


def get_model_score_from_notebook(notebook_path: str, name: str, track: str) -> ModelScore:
    raw_report = _get_notebook(notebook_path)
    metric_calls = _get_track_metric_calls(track)

    raw_tables = []
    for metric_call in metric_calls:
        try:
            raw_tables.append(_get_raw_html_report_score_table(raw_report, metric_call))
        except ValueError:
            legacy_metric_calls = LEGACY_TRACK_METRIC_CALLS[track]
            legacy_metric_call = legacy_metric_calls[metric_calls.index(metric_call)]
            raw_tables.append(_get_raw_html_report_score_table(raw_report, legacy_metric_call))

    return _convert_raw_html_report_score_tables_to_model_score(raw_tables, name, track)


def get_model_score_from_file(file_path: str) -> ModelScore:
    with open(file_path) as file:
        return ModelScore.model_validate(json.load(file))
