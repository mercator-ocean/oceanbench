# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from importlib import resources
from urllib.request import urlopen

import nbformat

from oceanbench.core import templates
from oceanbench.core.regions import RegionLike, resolve_region, region_to_dict

CHALLENGER_DATASET_PLACEHOLDER = "challenger_dataset: xarray.Dataset = xarray.Dataset()"
EVALUATION_REGION_PLACEHOLDER = 'region = "global"'

REPORT_PROFILE_DEFAULT = "default"
REPORT_PROFILE_SURFACE_ONLY = "surface_only"
_LIVE_EVALUATION_TEMPLATE_NAMES = {
    REPORT_PROFILE_DEFAULT: "live_evaluation_template.py",
    REPORT_PROFILE_SURFACE_ONLY: "live_evaluation_surface_template.py",
}


def _live_evaluation_template_file_name(report_profile: str | None) -> str:
    resolved_report_profile = report_profile or REPORT_PROFILE_DEFAULT
    try:
        return _LIVE_EVALUATION_TEMPLATE_NAMES[resolved_report_profile]
    except KeyError:
        raise ValueError(
            f"Unknown report profile {resolved_report_profile!r}. "
            f"Expected one of {sorted(_LIVE_EVALUATION_TEMPLATE_NAMES)}."
        )


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
    _generate_notebook_file(
        challenger_python_code_uri_or_local_path,
        output_notebook_file_path,
        region=region,
        template_file_name="evaluation_template.py",
        metadata_updates={},
    )


def generate_live_evaluation_notebook_file(
    challenger_python_code_uri_or_local_path: str,
    output_notebook_file_path: str,
    region: RegionLike = None,
    report_profile: str | None = REPORT_PROFILE_DEFAULT,
):
    resolved_report_profile = report_profile or REPORT_PROFILE_DEFAULT
    _generate_notebook_file(
        challenger_python_code_uri_or_local_path,
        output_notebook_file_path,
        region=region,
        template_file_name=_live_evaluation_template_file_name(resolved_report_profile),
        metadata_updates={"live_evaluation": True, "report_profile": resolved_report_profile},
    )


def _generate_notebook_file(
    challenger_python_code_uri_or_local_path: str,
    output_notebook_file_path: str,
    region: RegionLike = None,
    template_file_name: str = "evaluation_template.py",
    metadata_updates: dict | None = None,
):
    resolved_region = resolve_region(region)
    challenger_python_code = _parse_challenger_python_code(challenger_python_code_uri_or_local_path)
    notebook = _generate_template_notebook(template_file_name)
    notebook = _replace_code_to_open_challenger_datasets(challenger_python_code, notebook)
    notebook = _replace_evaluation_configuration_code(resolved_region, notebook)
    notebook.metadata.setdefault("oceanbench", {})["region"] = region_to_dict(resolved_region)
    notebook.metadata["oceanbench"].update(metadata_updates or {})
    nbformat.write(notebook, output_notebook_file_path)


def _generate_template_notebook(template_file_name: str = "evaluation_template.py") -> nbformat.NotebookNode:
    evaluation_template_file = resources.files(templates) / template_file_name
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
        _generate_evaluation_configuration_code(region),
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
