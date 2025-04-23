# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from helpers.color import get_color
from helpers.type import ModelScore


LEAD_DAY_INDEXES = ["1", "3", "5", "7", "10"]


def _get_variables(score: ModelScore, depth_label: str) -> list[str]:
    return list(score.depths[depth_label].variables.keys())


def _get_score_value(
    score: ModelScore,
    depth_label: str,
    variable_label: str,
    index: str,
) -> float:
    return float(score.depths[depth_label].variables[variable_label].data[index])


def _get_model_names_table(
    depth_label: str,
    reference_score: ModelScore,
    other_scores: list[ModelScore],
) -> str:
    thead = "<thead><tr><th>Models</th></tr></thead>"
    model_names = [reference_score.name] + [score.name for score in other_scores if depth_label in score.depths]
    model_name_as_tr = [f"<tr><th>{name}</th></tr>" for name in model_names]
    tbody = "<tbody>" + "".join(model_name_as_tr) + "</tbody>"
    tfoot = "<tfoot><tr><th>Lead day</th></tr></tfoot>"
    table = f"<table class='model-names'>{thead}{tbody}{tfoot}</table>"
    return table


def _get_variable_table_body_row_cells(
    reference_score: ModelScore,
    depth_label: str,
    variable_label: str,
    data: dict[str, float],
) -> str:
    table_cells = []
    for lead_day_index in LEAD_DAY_INDEXES:
        value = data[lead_day_index]
        reference_value = _get_score_value(reference_score, depth_label, variable_label, lead_day_index)
        color = get_color(reference_value, value)
        table_cells.append(f"<td style='background-color:{color}'>{'{:.2f}'.format(value)}</td>")
    return "".join(table_cells)


def _get_variable_table_body_row(
    variable_label: str,
    reference_score: ModelScore,
    depth_label: str,
    score: ModelScore,
) -> str:
    if depth_label in score.depths:
        html_model_row = "<tr>"
        variables = score.depths[depth_label].variables
        if variable_label in variables:
            data = variables[variable_label].data
            html_model_row += _get_variable_table_body_row_cells(reference_score, depth_label, variable_label, data)
        else:
            html_model_row += "<td>NaN</td>" * len(LEAD_DAY_INDEXES)
        html_model_row += "</tr>"
        return html_model_row
    return ""


def _get_variable_table_body(
    variable_label: str,
    reference_score: ModelScore,
    depth_label: str,
    other_scores: list[ModelScore],
) -> str:
    html_table_body = "<tbody>"
    html_table_body += _get_variable_table_body_row(
        variable_label=variable_label,
        reference_score=reference_score,
        depth_label=depth_label,
        score=reference_score,
    )
    for other_score in other_scores:
        html_model_row = _get_variable_table_body_row(
            variable_label=variable_label,
            reference_score=reference_score,
            depth_label=depth_label,
            score=other_score,
        )
        html_table_body += html_model_row
    html_table_body += "</tbody>"
    return html_table_body


def _get_variable_table(
    variable_label: str,
    reference_score: ModelScore,
    depth_label: str,
    other_scores: list[ModelScore],
) -> str:
    thead = f"<thead><tr><th colspan='{len(LEAD_DAY_INDEXES)}'>{variable_label}</th></tr></thead>"
    tbody = _get_variable_table_body(
        variable_label=variable_label,
        reference_score=reference_score,
        depth_label=depth_label,
        other_scores=other_scores,
    )
    tfoot = "<tfoot><tr>" + "".join([f"<td>{index}</td>" for index in LEAD_DAY_INDEXES]) + "</tr></tfoot>"

    html_table = f"<table>{thead}{tbody}{tfoot}</table>"

    return html_table


def get_html_tables(
    reference_score: ModelScore,
    depth_label: str,
    other_scores: list[ModelScore],
) -> str:
    models = _get_model_names_table(depth_label, reference_score, other_scores)

    variables = _get_variables(reference_score, depth_label)
    variable_tables = ""
    for variable in variables:
        variable_tables += _get_variable_table(variable, reference_score, depth_label, other_scores)

    html_tables = f"<div class='row'>{models}<div class='variables'>{variable_tables}</div></div>"

    return html_tables
