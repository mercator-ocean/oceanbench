# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import nbformat
from importlib import resources
from oceanbench.core import templates
from urllib.request import urlopen


def _parse_challenger_python_code(
    challenger_python_code_uri_or_local_path: str,
) -> str:
    try:
        with open(challenger_python_code_uri_or_local_path, "r", encoding="utf8") as file:
            return file.read()
    except Exception:
        with urlopen(challenger_python_code_uri_or_local_path) as response:
            return response.read()


def generate_evaluation_notebook_file(
    challenger_python_code_uri_or_local_path: str,
    output_notebook_file_path: str,
):
    challenger_python_code = _parse_challenger_python_code(challenger_python_code_uri_or_local_path)
    notebook = _generate_template_notebook()
    new_notebook = _replace_code_to_open_challenger_datasets(challenger_python_code, notebook)
    nbformat.write(new_notebook, output_notebook_file_path)


def _generate_template_notebook() -> nbformat.NotebookNode:
    evaluation_template_file = resources.files(templates) / "evaluation_template.py"
    with evaluation_template_file.open("r", encoding="utf8") as file:
        evaluation_template_code = file.read()
        return _python_to_jupyter_notebook(evaluation_template_code)


def _new_cell(cell_content: str, cell_type: str):
    new_cell_content = cell_content.removesuffix("\n").removesuffix("\n")
    return (
        nbformat.v4.new_markdown_cell(new_cell_content)
        if cell_type == "markdown"
        else nbformat.v4.new_code_cell(new_cell_content)
    )


def _python_to_jupyter_notebook(python_code: str) -> nbformat.NotebookNode:
    cells = []
    current_cell_type = ""
    current_cell_content = ""

    for line in python_code.split("\n"):
        if line.strip().startswith("# "):
            if current_cell_content and current_cell_type != "markdown":
                cells.append(_new_cell(current_cell_content, current_cell_type))
                current_cell_content = ""
            current_cell_type = "markdown"
        elif line.strip() != "":
            if current_cell_content and current_cell_type != "code":
                cells.append(_new_cell(current_cell_content, current_cell_type))
                current_cell_content = ""
            current_cell_type = "code"
        new_line = line.removeprefix("# ") if current_cell_type == "markdown" else line
        current_cell_content += new_line + "\n"

    if current_cell_content:
        cells.append(_new_cell(current_cell_content, current_cell_type))

    notebook = nbformat.v4.new_notebook()
    notebook.cells = cells
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    return notebook


def _replace_code_to_open_challenger_datasets(
    python_code: str, notebook: nbformat.NotebookNode
) -> nbformat.NotebookNode:
    notebook["cells"][2]["source"] = python_code
    return notebook
