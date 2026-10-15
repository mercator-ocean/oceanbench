# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path

import nbformat

from oceanbench.core.evaluation_timings import evaluation_step_timings, format_evaluation_timings

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _executed_cell(cell: nbformat.NotebookNode, duration: float) -> nbformat.NotebookNode:
    cell.metadata["papermill"] = {"duration": duration}
    return cell


def _executed_notebook() -> nbformat.NotebookNode:
    notebook = nbformat.v4.new_notebook()
    notebook.cells = [
        _executed_cell(nbformat.v4.new_code_cell("import oceanbench\n\noceanbench.__version__"), 2.9),
        _executed_cell(nbformat.v4.new_markdown_cell("### Open challenger datasets\n\n> Insert here"), 0.001),
        _executed_cell(nbformat.v4.new_code_cell("# Open sample\nchallenger_dataset = open()"), 0.56),
        _executed_cell(nbformat.v4.new_markdown_cell("### Evaluation of challenger\n\n#### RMSD of MLD"), 0.002),
        _executed_cell(
            nbformat.v4.new_code_cell(
                "oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(\n"
                "    challenger_dataset,\n"
                "    region=region,\n"
                ")"
            ),
            22.26,
        ),
    ]
    notebook.metadata["papermill"] = {"duration": 30.0}
    return nbformat.from_dict(notebook)


def test_evaluation_step_timings_labels_code_cells():
    assert evaluation_step_timings(_executed_notebook()) == [
        ("import oceanbench", 2.9),
        ("Open challenger datasets", 0.56),
        ("rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis", 22.26),
    ]


def test_format_evaluation_timings_ends_with_total():
    assert format_evaluation_timings(_executed_notebook(), "sample.global.report.ipynb") == (
        "OceanBench step timings for sample.global.report.ipynb:\n"
        "       2.9 s  import oceanbench\n"
        "       0.6 s  Open challenger datasets\n"
        "      22.3 s  rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis\n"
        "      30.0 s  total"
    )


def test_evaluation_step_timings_reads_sample_report():
    notebook = nbformat.read(PROJECT_ROOT / "assets" / "glonet_1_degree_sample.global.report.ipynb", as_version=4)

    labels = [label for label, _ in evaluation_step_timings(notebook)]

    assert labels[:3] == ["import oceanbench", "Open challenger datasets", "Evaluation configuration"]
    assert labels[3:] == [
        "rmsd_of_variables_compared_to_glorys_reanalysis",
        "rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis",
        "rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis",
        "rmsd_of_variables_compared_to_observations",
        "deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis",
        "rmsd_of_variables_compared_to_glo12_analysis",
        "rmsd_of_mixed_layer_depth_compared_to_glo12_analysis",
        "rmsd_of_geostrophic_currents_compared_to_glo12_analysis",
        "deviation_of_lagrangian_trajectories_compared_to_glo12_analysis",
    ]
