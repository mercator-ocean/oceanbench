from helpers.color import get_color
from helpers.type import ModelScore


LEAD_DAY_INDEXES = ["1", "3", "5", "7", "10"]


def _get_value(
    reference_score: ModelScore,
    depth_label: str,
    variable_label: str,
    index: str,
) -> float:
    return float(reference_score.depths[depth_label].variables[variable_label].data[index])


def _get_variables(score: ModelScore, depth_label: str) -> list[str]:
    return list(score.depths[depth_label].variables.keys())


def _get_table_header(reference_score: ModelScore, depth_label: str) -> str:
    variables = _get_variables(reference_score, depth_label)
    table_cells = [f"<td colspan='{len(LEAD_DAY_INDEXES)}'>{variable_label}</td>" for variable_label in variables]
    return (
        "<thead><tr class='sticky'>"
        + "<th class='sticky' style='background: #f5f5f5'>Models</th>"
        + "".join(table_cells)
        + "</tr></thead>"
    )


def _get_table_footer(reference_score: ModelScore, depth_label: str) -> str:
    variables = _get_variables(reference_score, depth_label)
    return (
        "<tfoot><tr class='sticky'>"
        + "<th class='sticky' style='background: #f5f5f5'>Lead day</th>"
        + "".join([f"<td>{lead_day_index}</td>" for lead_day_index in LEAD_DAY_INDEXES] * len(variables))
        + "</tr></tfoot>"
    )


def _get_variable_cells(
    reference_score: ModelScore,
    depth_label: str,
    variable_label: str,
    data: dict[str, float],
) -> str:
    table_cells = []
    for lead_day_index in LEAD_DAY_INDEXES:
        value = data[lead_day_index]
        reference_value = _get_value(reference_score, depth_label, variable_label, lead_day_index)
        color = get_color(reference_value, value)
        table_cells.append(f"<td style='background-color:{color}'>{'{:.2f}'.format(value)}</td>")
    return "".join(table_cells)


def _get_table_body_row(
    reference_score: ModelScore,
    depth_label: str,
    score: ModelScore,
) -> str:
    if depth_label in score.depths:
        html_model_row = f"<tr><th class='sticky' style='background: #f5f5f5'>{score.name}</th>"
        variables = score.depths[depth_label].variables
        variable_labels = _get_variables(reference_score, depth_label)
        for variable_label in variable_labels:
            if variable_label in variables:
                data = variables[variable_label].data
                html_model_row += _get_variable_cells(reference_score, depth_label, variable_label, data)
            else:
                html_model_row += "<td></td>" * len(LEAD_DAY_INDEXES)
        html_model_row += "</tr>"
        return html_model_row
    return ""


def _get_table_body(
    reference_score: ModelScore,
    depth_label: str,
    other_scores: list[ModelScore],
) -> str:
    html_table_body = "<tbody>"
    html_table_body += _get_table_body_row(reference_score, depth_label, reference_score)
    for other_score in other_scores:
        html_model_row = _get_table_body_row(reference_score, depth_label, other_score)
        html_table_body += html_model_row
    html_table_body += "</tbody>"
    return html_table_body


def get_html_table(
    reference_score: ModelScore,
    depth_label: str,
    other_scores: list[ModelScore],
) -> str:
    html_table_head = _get_table_header(reference_score, depth_label)
    html_table_body = _get_table_body(reference_score, depth_label, other_scores)
    html_table_footer = _get_table_footer(reference_score, depth_label)

    html_table = f"<table>{html_table_head}{html_table_body}{html_table_footer}</table>"

    return html_table
