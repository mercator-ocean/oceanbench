# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from pathlib import Path
import tempfile

import nbformat
from papermill import execute_notebook

from oceanbench.demo import glonet


def _environment_setup_code() -> str:
    return """from pathlib import Path
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="IProgress not found.*")

candidate_roots = [Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]
PROJECT_ROOT = next(
    (path for path in candidate_roots if (path / "oceanbench").exists() and (path / "assets").exists()),
    Path.cwd(),
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

stale_oceanbench_modules = [
    module_name
    for module_name in list(sys.modules)
    if module_name == "oceanbench" or module_name.startswith("oceanbench.")
]
for module_name in sorted(stale_oceanbench_modules, reverse=True):
    sys.modules.pop(module_name, None)

CACHE_DIR = Path("/tmp/oceanbench-demo-cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MPL_CACHE_DIR = CACHE_DIR / "matplotlib"
MPL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ["XDG_CACHE_HOME"] = str(CACHE_DIR)
os.environ["MPLCONFIGDIR"] = str(MPL_CACHE_DIR)
"""


def _loader_code() -> str:
    return (glonet.project_root() / "assets" / "glonet_psd_demo.py").read_text(encoding="utf8")


def _reference_setup_code() -> str:
    return """from oceanbench.core.dataset_utils import Variable
from oceanbench.core.references.glorys import glorys_reanalysis_dataset

glorys_dataset = glorys_reanalysis_dataset(challenger_dataset)
"""


def _psd_cells() -> list[nbformat.NotebookNode]:
    return [
        nbformat.v4.new_markdown_cell(
            "### GLONET vs GLORYS PSD diagnostics"
        ),
        nbformat.v4.new_markdown_cell(
            "#### SSH zonal PSD"
        ),
        nbformat.v4.new_code_cell(
            "challenger_ssh_psd, glorys_ssh_psd = oceanbench.psd.zonal_longitude_psd_pair(\n"
            "    challenger_dataset,\n"
            "    glorys_dataset,\n"
            "    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,\n"
            ")\n"
            "challenger_ssh_psd"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Wavelength bands used for zonal PSD scores"
        ),
        nbformat.v4.new_code_cell(
            "zonal_wavelength_bands_km = oceanbench.psd.default_zonal_wavelength_bands_km(challenger_ssh_psd)\n"
            "zonal_wavelength_bands_km"
        ),
        nbformat.v4.new_markdown_cell(
            "#### SSH zonal PSD scores for all lead days"
        ),
        nbformat.v4.new_code_cell(
            "import pandas\n\n"
            "ssh_psd_scores = pandas.concat(\n"
            "    {\n"
            "        \"GLONET\": oceanbench.psd.zonal_longitude_psd_metrics_from_spectrum(\n"
            "            challenger_ssh_psd,\n"
            "            wavelength_bands_km=zonal_wavelength_bands_km,\n"
            "        ),\n"
            "        \"GLORYS\": oceanbench.psd.zonal_longitude_psd_metrics_from_spectrum(\n"
            "            glorys_ssh_psd,\n"
            "            wavelength_bands_km=zonal_wavelength_bands_km,\n"
            "        ),\n"
            "    }\n"
            ")\n"
            "ssh_psd_scores"
        ),
        nbformat.v4.new_markdown_cell(
            "#### SST zonal PSD"
        ),
        nbformat.v4.new_code_cell(
            "challenger_sst_psd, glorys_sst_psd = oceanbench.psd.zonal_longitude_psd_pair(\n"
            "    challenger_dataset,\n"
            "    glorys_dataset,\n"
            "    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,\n"
            ")\n"
            "challenger_sst_psd"
        ),
        nbformat.v4.new_markdown_cell(
            "#### SST zonal PSD scores for all lead days"
        ),
        nbformat.v4.new_code_cell(
            "sst_psd_scores = pandas.concat(\n"
            "    {\n"
            "        \"GLONET\": oceanbench.psd.zonal_longitude_psd_metrics_from_spectrum(\n"
            "            challenger_sst_psd,\n"
            "            wavelength_bands_km=zonal_wavelength_bands_km,\n"
            "        ),\n"
            "        \"GLORYS\": oceanbench.psd.zonal_longitude_psd_metrics_from_spectrum(\n"
            "            glorys_sst_psd,\n"
            "            wavelength_bands_km=zonal_wavelength_bands_km,\n"
            "        ),\n"
            "    }\n"
            ")\n"
            "sst_psd_scores"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Zonal PSD comparison for day 1 and day 10"
        ),
        nbformat.v4.new_code_cell(
            "glorys_zonal_psd_figure = oceanbench.visualization.plot_zonal_longitude_psd_comparison_gallery(\n"
            "    challenger_dataset,\n"
            "    glorys_dataset,\n"
            "    \"GLORYS reanalysis\",\n"
            "    lead_day_indices=[0, 9],\n"
            ")"
        ),
    ]


def build_notebook(output_path: Path | None = None) -> Path:
    glonet.ensure_local_demo_data_exists()

    project_root = glonet.project_root()
    notebook_output_path = output_path or project_root / "assets" / "glonet_psd_demo.report.ipynb"

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(_environment_setup_code()),
            nbformat.v4.new_code_cell("import oceanbench\n\noceanbench.__version__"),
            nbformat.v4.new_markdown_cell(
                "### Open challenger dataset\n\n"
                "Local GLONET demo sample cached under `demo_data/glonet_sample/`."
            ),
            nbformat.v4.new_code_cell(_loader_code()),
            nbformat.v4.new_code_cell(_reference_setup_code()),
            *_psd_cells(),
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
