# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import tempfile

import nbformat
from papermill import execute_notebook

from oceanbench.demo import glonet
from oceanbench.demo.build_glonet_demo_notebook import _environment_setup_code, _evaluation_cells


def _loader_code() -> str:
    return (glonet.project_root() / "assets" / "regional_evaluation.py").read_text(encoding="utf8")


def _reference_setup_code() -> str:
    return """from oceanbench.demo import glonet
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents
from oceanbench.core.derived_quantities import compute_mixed_layer_depth
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import observations

glorys_dataset = oceanbench.regions.subset_dataset_to_region(
    glorys_reanalysis_dataset(challenger_dataset),
    region_definition,
)
glo12_dataset = oceanbench.regions.subset_dataset_to_region(
    glo12_analysis_dataset(challenger_dataset),
    region_definition,
)
observation_dataset = observations(challenger_dataset)

challenger_eddy_dataset = oceanbench.regions.subset_dataset_to_region(
    glonet.load_local_eddy_challenger_dataset(),
    region_definition,
)
glorys_eddy_dataset = oceanbench.regions.subset_dataset_to_region(
    glonet.load_local_eddy_glorys_dataset(),
    region_definition,
)
glo12_eddy_dataset = oceanbench.regions.subset_dataset_to_region(
    glonet.load_local_eddy_glo12_dataset(),
    region_definition,
)

challenger_mld_dataset = compute_mixed_layer_depth(challenger_dataset)
glorys_mld_dataset = compute_mixed_layer_depth(glorys_dataset)
glo12_mld_dataset = compute_mixed_layer_depth(glo12_dataset)

challenger_geostrophic_dataset = compute_geostrophic_currents(challenger_dataset)
glorys_geostrophic_dataset = compute_geostrophic_currents(glorys_dataset)
glo12_geostrophic_dataset = compute_geostrophic_currents(glo12_dataset)

eddy_detection_parameters = oceanbench.eddies.default_eddy_detection_parameters()
"""


def build_notebook(output_path: Path | None = None) -> Path:
    glonet.ensure_local_demo_data_exists()
    glonet.ensure_local_eddy_demo_data_exists()

    project_root = glonet.project_root()
    notebook_output_path = output_path or project_root / "assets" / "regional_evaluation.report.ipynb"

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(_environment_setup_code()),
            nbformat.v4.new_code_cell("import oceanbench\n\noceanbench.__version__"),
            nbformat.v4.new_markdown_cell(
                "### Open challenger datasets\n\n"
                "Regional North Atlantic GLONET demo cached under `demo_data/glonet_sample/` "
                "and `demo_data/glonet_eddy_sample/`."
            ),
            nbformat.v4.new_code_cell(_loader_code()),
            nbformat.v4.new_markdown_cell(
                "### Region\n\n"
                "`North Atlantic`: latitude `0°` to `75°`, longitude `100°W` to `20°E`."
            ),
            nbformat.v4.new_code_cell(_reference_setup_code()),
            *_evaluation_cells(),
        ]
    )
    notebook.metadata.kernelspec = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    notebook.metadata.language_info = {
        "name": "python",
        "version": "3.11",
    }

    with tempfile.NamedTemporaryFile("w", suffix=".ipynb", delete=False, encoding="utf8") as tmp:
        nbformat.write(notebook, tmp)
        notebook_path = tmp.name

    execute_notebook(notebook_path, str(notebook_output_path), cwd=str(project_root))
    return notebook_output_path


def main() -> None:
    notebook_path = build_notebook()
    print(notebook_path)


if __name__ == "__main__":
    main()
