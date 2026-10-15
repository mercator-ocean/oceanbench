# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import re

import nbformat

METRIC_CALL_PATTERN = re.compile(r"oceanbench\.metrics\.(\w+)\(")


def _cell_label(cell: nbformat.NotebookNode, previous_heading: str | None) -> str:
    metric_call = METRIC_CALL_PATTERN.search(cell.source)
    if metric_call:
        return metric_call.group(1)
    if previous_heading:
        return previous_heading
    return cell.source.splitlines()[0]


def evaluation_step_timings(notebook: nbformat.NotebookNode) -> list[tuple[str, float]]:
    """
    Read the duration of each code cell of an executed evaluation notebook, as recorded by papermill.

    Metric cells are labelled by their metric function name, other cells by the heading above them.
    """
    timings = []
    previous_heading = None
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            previous_heading = cell.source.splitlines()[0].lstrip("# ")
        else:
            timings.append((_cell_label(cell, previous_heading), cell.metadata.papermill.duration))
            previous_heading = None
    return timings


def format_evaluation_timings(notebook: nbformat.NotebookNode, notebook_name: str) -> str:
    lines = [f"OceanBench step timings for {notebook_name}:"]
    lines += [f"{duration:10.1f} s  {label}" for label, duration in evaluation_step_timings(notebook)]
    lines.append(f"{notebook.metadata.papermill.duration:10.1f} s  total")
    return "\n".join(lines)
