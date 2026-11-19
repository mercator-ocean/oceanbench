# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Execute the example notebook to refresh its stored outputs, or check that the notebook still works.

Contributor housekeeping, run through ``make execute-example-notebook`` and ``make check-example-notebook``.

Refreshing executes every cell. Checking is cheap: every code cell must parse, every metric it calls must exist,
only a few cells are executed and compared with their stored outputs, and every stored score table must hold the
scores of the sample scores file, which the sample evaluation of the 1 degree challenger keeps up to date.
"""

import argparse
import ast
import copy
import json
import re
import sys
from enum import Enum
from pathlib import Path

import nbformat
from bs4 import BeautifulSoup
from nbclient import NotebookClient

import oceanbench

REPOSITORY_PATH = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPOSITORY_PATH / "assets" / "evaluate_challenger_example.ipynb"
SAMPLE_SCORES_PATH = REPOSITORY_PATH / "assets" / "glonet_1_degree_sample.global.scores.json"
EXECUTED_CELL_IDS_WHEN_CHECKING = ("challenger-dataset", "metric-3")
METRIC_CALL_PATTERN = re.compile(r"oceanbench\.metrics\.(\w+?)_compared_to_(\w+)\(")
LEAD_DAY_HEADER_PATTERN = re.compile(r"^Lead day \d+$")
CELL_TIMEOUT_SECONDS = 1800
NUMBER_PATTERN = re.compile(r"-?\d+\.\d+(?:e[+-]?\d+)?", re.IGNORECASE)
NUMBER_PLACEHOLDER = "#"
ABSOLUTE_TOLERANCE = 1e-5
RELATIVE_TOLERANCE = 1e-5
STANDARD_ERROR_STREAM = "stderr"


class Mode(Enum):
    REFRESH = "refresh"
    CHECK = "check"


def _is_environment_noise(output: dict) -> bool:
    return output["output_type"] == "stream" and output["name"] == STANDARD_ERROR_STREAM


def _without_environment_noise(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    cleaned = copy.deepcopy(notebook)
    for cell in cleaned.cells:
        if cell.cell_type == "code":
            cell.outputs = [output for output in cell.outputs if not _is_environment_noise(output)]
    return cleaned


def _with_cells(notebook: nbformat.NotebookNode, cell_ids: tuple[str, ...]) -> nbformat.NotebookNode:
    selected = copy.deepcopy(notebook)
    selected.cells = [cell for cell in selected.cells if cell.id in cell_ids]
    return selected


def _executed(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    client = NotebookClient(
        copy.deepcopy(notebook),
        timeout=CELL_TIMEOUT_SECONDS,
        record_timing=False,
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )
    executed = _without_environment_noise(client.execute())
    executed.metadata = copy.deepcopy(notebook.metadata)
    return executed


def _output_text(output: dict) -> str:
    data = output.get("data", {})
    return "".join(data.get("text/html") or data.get("text/plain") or output.get("text", ""))


def _numbers(text: str) -> list[float]:
    return [float(number) for number in NUMBER_PATTERN.findall(text)]


def _numbers_are_within_tolerance(first_number: float, second_number: float) -> bool:
    tolerance = ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * max(abs(first_number), abs(second_number))
    return abs(first_number - second_number) <= tolerance


def _same_output(stored_text: str, executed_text: str) -> bool:
    stored_numbers, executed_numbers = _numbers(stored_text), _numbers(executed_text)
    return (
        NUMBER_PATTERN.sub(NUMBER_PLACEHOLDER, stored_text) == NUMBER_PATTERN.sub(NUMBER_PLACEHOLDER, executed_text)
        and len(stored_numbers) == len(executed_numbers)
        and all(map(_numbers_are_within_tolerance, stored_numbers, executed_numbers))
    )


def _outdated_cells(stored: nbformat.NotebookNode, executed: nbformat.NotebookNode) -> list[str]:
    executed_outputs = {cell.id: cell.outputs for cell in executed.cells if cell.cell_type == "code"}
    return [
        f"cell {cell.id}: the stored outputs differ from the executed ones"
        for cell in _without_environment_noise(stored).cells
        if cell.id in executed_outputs
        and not (
            len(cell.outputs) == len(executed_outputs[cell.id])
            and all(
                _same_output(_output_text(stored_output), _output_text(executed_output))
                for stored_output, executed_output in zip(cell.outputs, executed_outputs[cell.id])
            )
        )
    ]


def _code_cells(notebook: nbformat.NotebookNode) -> list[nbformat.NotebookNode]:
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


def _broken_cells(notebook: nbformat.NotebookNode) -> list[str]:
    def parse_error(source: str) -> str | None:
        try:
            ast.parse(source)
        except SyntaxError as error:
            return str(error)
        return None

    unparsable = [f"cell {cell.id}: {error}" for cell in _code_cells(notebook) if (error := parse_error(cell.source))]
    unknown_metrics = [
        f"cell {cell.id}: oceanbench.metrics.{metric}_compared_to_{reference} does not exist"
        for cell in _code_cells(notebook)
        for metric, reference in METRIC_CALL_PATTERN.findall(cell.source)
        if not hasattr(oceanbench.metrics, f"{metric}_compared_to_{reference}")
    ]
    return unparsable + unknown_metrics


def _table_values(html: str) -> list[float]:
    soup = BeautifulSoup(html, features="html.parser")
    headers = [header.get_text(strip=True) for header in soup.find("thead").find_all("th")][1:]
    value_columns = [index for index, header in enumerate(headers) if LEAD_DAY_HEADER_PATTERN.match(header)]
    cells = [[cell.get_text(strip=True) for cell in row.find_all("td")] for row in soup.find("tbody").find_all("tr")]
    return [float(row[index]) for row in cells for index in value_columns if NUMBER_PATTERN.fullmatch(row[index])]


def _sample_values(metric: str, reference: str) -> list[float]:
    rows = json.loads(SAMPLE_SCORES_PATH.read_text(encoding="utf-8"))["rows"]
    return [
        row["value"]
        for row in rows
        if row["metric"] == metric and row["reference"] == reference and row["value"] is not None
    ]


def _tables_differing_from_the_sample_scores(notebook: nbformat.NotebookNode) -> list[str]:
    differing = []
    for cell in _code_cells(notebook):
        for metric, reference in METRIC_CALL_PATTERN.findall(cell.source):
            stored_values = sorted(_table_values("".join(cell.outputs[0]["data"]["text/html"])))
            sample_values = sorted(_sample_values(metric, reference))
            if len(stored_values) != len(sample_values) or not all(
                map(_numbers_are_within_tolerance, stored_values, sample_values)
            ):
                differing.append(f"cell {cell.id}: the stored table differs from the sample scores file")
    return differing


def _refresh(stored: nbformat.NotebookNode, executed: nbformat.NotebookNode) -> int:
    nbformat.write(executed, NOTEBOOK_PATH)
    print(f"Refreshed the outputs of {NOTEBOOK_PATH}.")
    return 0


def _check(stored: nbformat.NotebookNode) -> int:
    problems = (
        _broken_cells(stored)
        + _outdated_cells(stored, _executed(_with_cells(stored, EXECUTED_CELL_IDS_WHEN_CHECKING)))
        + _tables_differing_from_the_sample_scores(stored)
    )
    print("\n".join(problems) if problems else "The notebook works and its stored outputs are up to date.")
    return 1 if problems else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", type=Mode, choices=list(Mode), metavar="{refresh,check}")
    mode = parser.parse_args().mode
    stored = nbformat.read(NOTEBOOK_PATH, as_version=4)
    match mode:
        case Mode.REFRESH:
            return _refresh(stored, _executed(stored))
        case Mode.CHECK:
            return _check(stored)


if __name__ == "__main__":
    sys.exit(main())
