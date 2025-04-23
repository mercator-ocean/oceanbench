# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import nbformat


def generate_evaluation_notebook_file(
    python_file_path: str,
    output_notebook_file_path: str,
):
    notebook = _generate_template_notebook()
    with open(python_file_path, "r", encoding="utf8") as file:
        code = file.read()
        new_notebook = _replace_code_to_open_challenger_datasets(code, notebook)
        nbformat.write(new_notebook, output_notebook_file_path)


def _generate_template_notebook() -> nbformat.NotebookNode:
    with open("evaluation_template.py", "r", encoding="utf8") as file:
        code = file.read()
        return _python_to_jupyter_notebook(code)


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

    return notebook


def _replace_code_to_open_challenger_datasets(
    python_code: str, notebook: nbformat.NotebookNode
) -> nbformat.NotebookNode:
    notebook["cells"][2]["source"] = python_code
    return notebook
