# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from importlib import resources
from pathlib import Path
from urllib.request import urlopen

import nbformat

from oceanbench.core import templates
from oceanbench.core.regions import RegionLike, resolve_region, region_to_dict

CHALLENGER_DATASET_PLACEHOLDER = "challenger_dataset: xarray.Dataset = xarray.Dataset()"
EVALUATION_REGION_PLACEHOLDER = 'region = "global"'
CHALLENGER_NAME_PLACEHOLDER = 'challenger_name = "challenger"'
SCORE_FILE_PATH_PLACEHOLDER = 'score_file_path = "challenger.global.scores.json"'
WIDGET_ASSET_DIRECTORY_PLACEHOLDER = 'widget_asset_directory = "challenger.global.assets"'
WIDGET_ASSET_REFERENCE_PREFIX_PLACEHOLDER = 'widget_asset_reference_prefix = "challenger.global.assets/"'
REPORT_NOTEBOOK_SUFFIX = ".report.ipynb"


def _parse_challenger_python_code(
    challenger_python_code_uri_or_local_path: str,
) -> str:
    try:
        with open(challenger_python_code_uri_or_local_path, "r", encoding="utf8") as file:
            return file.read()
    except Exception:
        with urlopen(challenger_python_code_uri_or_local_path) as response:
            return response.read().decode("utf-8")


def generate_evaluation_notebook_file(
    challenger_python_code_uri_or_local_path: str,
    output_notebook_file_path: str,
    region: RegionLike = None,
):
    resolved_region = resolve_region(region)
    challenger_python_code = _parse_challenger_python_code(challenger_python_code_uri_or_local_path)
    notebook = _generate_template_notebook()
    notebook = _replace_code_to_open_challenger_datasets(challenger_python_code, notebook)
    notebook = _replace_evaluation_configuration_code(resolved_region, notebook)
    notebook = _replace_report_artifact_configuration_code(output_notebook_file_path, notebook)
    notebook.metadata.setdefault("oceanbench", {})["region"] = region_to_dict(resolved_region)
    nbformat.write(notebook, output_notebook_file_path)


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
    _replace_cell_source(notebook, CHALLENGER_DATASET_PLACEHOLDER, python_code)
    return notebook


def _replace_evaluation_configuration_code(
    region,
    notebook: nbformat.NotebookNode,
) -> nbformat.NotebookNode:
    _replace_cell_source(
        notebook,
        EVALUATION_REGION_PLACEHOLDER,
        "\n".join(
            [
                _generate_evaluation_configuration_code(region),
                CHALLENGER_NAME_PLACEHOLDER,
                SCORE_FILE_PATH_PLACEHOLDER,
                WIDGET_ASSET_DIRECTORY_PLACEHOLDER,
                WIDGET_ASSET_REFERENCE_PREFIX_PLACEHOLDER,
            ]
        ),
    )
    return notebook


def _report_stem(output_notebook_file_path: str) -> str:
    file_name = Path(output_notebook_file_path).name
    if file_name.endswith(REPORT_NOTEBOOK_SUFFIX):
        return file_name.removesuffix(REPORT_NOTEBOOK_SUFFIX)
    return Path(file_name).stem


def _challenger_name_from_report_stem(report_stem: str) -> str:
    return report_stem.rsplit(".", maxsplit=1)[0] if "." in report_stem else report_stem


def _replace_report_artifact_configuration_code(
    output_notebook_file_path: str,
    notebook: nbformat.NotebookNode,
) -> nbformat.NotebookNode:
    report_stem = _report_stem(output_notebook_file_path)
    _replace_cell_source_fragment(
        notebook,
        CHALLENGER_NAME_PLACEHOLDER,
        f"challenger_name = {_challenger_name_from_report_stem(report_stem)!r}",
    )
    _replace_cell_source_fragment(
        notebook,
        SCORE_FILE_PATH_PLACEHOLDER,
        f"score_file_path = {f'{report_stem}.scores.json'!r}",
    )
    _replace_cell_source_fragment(
        notebook,
        WIDGET_ASSET_DIRECTORY_PLACEHOLDER,
        f"widget_asset_directory = {f'{report_stem}.assets'!r}",
    )
    _replace_cell_source_fragment(
        notebook,
        WIDGET_ASSET_REFERENCE_PREFIX_PLACEHOLDER,
        f"widget_asset_reference_prefix = {f'{report_stem}.assets/'!r}",
    )
    return notebook


def _replace_cell_source(
    notebook: nbformat.NotebookNode,
    source_fragment: str,
    replacement_source: str,
) -> None:
    for cell in notebook["cells"]:
        if source_fragment in cell["source"]:
            cell["source"] = replacement_source
            return
    raise ValueError(f"Unable to find evaluation template cell containing {source_fragment!r}.")


def _replace_cell_source_fragment(
    notebook: nbformat.NotebookNode,
    source_fragment: str,
    replacement_source: str,
) -> None:
    for cell in notebook["cells"]:
        if source_fragment in cell["source"]:
            cell["source"] = cell["source"].replace(source_fragment, replacement_source)
            return
    raise ValueError(f"Unable to find evaluation template cell containing {source_fragment!r}.")


def _generate_evaluation_configuration_code(region) -> str:
    if region.official:
        return f"region = {region.id!r}"
    region_data = region_to_dict(region)
    bounds = region_data["bounds"]
    return (
        "region = oceanbench.regions.custom(\n"
        f"    identifier={region_data['id']!r},\n"
        f"    display_name={region_data['display_name']!r},\n"
        f"    minimum_latitude={bounds['minimum_latitude']!r},\n"
        f"    maximum_latitude={bounds['maximum_latitude']!r},\n"
        f"    minimum_longitude={bounds['minimum_longitude']!r},\n"
        f"    maximum_longitude={bounds['maximum_longitude']!r},\n"
        ")"
    )
