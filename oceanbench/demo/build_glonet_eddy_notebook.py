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
    return (glonet.project_root() / "assets" / "glonet_eddy_demo.py").read_text(encoding="utf8")


def _reference_setup_code() -> str:
    return """glorys_dataset = glonet.load_local_eddy_glorys_dataset()"""


def _eddy_cells() -> list[nbformat.NotebookNode]:
    return [
        nbformat.v4.new_markdown_cell(
            "### GLONET vs GLORYS quarter-degree mesoscale eddy diagnostics"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Raw SSH fields"
        ),
        nbformat.v4.new_code_cell(
            "ssh_field_figure = oceanbench.visualization.plot_surface_field_comparison_gallery(\n"
            "    challenger_dataset,\n"
            "    glorys_dataset,\n"
            "    variable=\"sea_surface_height_above_geoid\",\n"
            "    reference_name=\"GLORYS reanalysis\",\n"
            "    challenger_name=\"GLONET\",\n"
            "    lead_day_indices=[0, 9],\n"
            "    cmap_override=\"jet\",\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Eddy detection parameters"
        ),
        nbformat.v4.new_code_cell(
            "eddy_detection_parameters = oceanbench.eddies.default_eddy_detection_parameters()\n"
            "eddy_detection_parameters"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Accepted eddy detections"
        ),
        nbformat.v4.new_code_cell(
            "raw_challenger_eddy_detections = oceanbench.eddies.detect_mesoscale_eddies(\n"
            "    challenger_dataset,\n"
            "    background_sigma_grid=eddy_detection_parameters[\"background_sigma_grid\"],\n"
            "    detection_sigma_grid=eddy_detection_parameters[\"detection_sigma_grid\"],\n"
            "    min_distance_grid=eddy_detection_parameters[\"min_distance_grid\"],\n"
            "    amplitude_threshold_meters=eddy_detection_parameters[\"amplitude_threshold_meters\"],\n"
            "    max_abs_latitude_degrees=eddy_detection_parameters[\"max_abs_latitude_degrees\"],\n"
            ")\n"
            "raw_glorys_eddy_detections = oceanbench.eddies.detect_mesoscale_eddies(\n"
            "    glorys_dataset,\n"
            "    background_sigma_grid=eddy_detection_parameters[\"background_sigma_grid\"],\n"
            "    detection_sigma_grid=eddy_detection_parameters[\"detection_sigma_grid\"],\n"
            "    min_distance_grid=eddy_detection_parameters[\"min_distance_grid\"],\n"
            "    amplitude_threshold_meters=eddy_detection_parameters[\"amplitude_threshold_meters\"],\n"
            "    max_abs_latitude_degrees=eddy_detection_parameters[\"max_abs_latitude_degrees\"],\n"
            ")\n"
            "challenger_eddy_contours = oceanbench.eddies.mesoscale_eddy_contours_from_detections(\n"
            "    raw_challenger_eddy_detections,\n"
            "    challenger_dataset,\n"
            "    background_sigma_grid=eddy_detection_parameters[\"background_sigma_grid\"],\n"
            "    detection_sigma_grid=eddy_detection_parameters[\"detection_sigma_grid\"],\n"
            "    amplitude_threshold_meters=eddy_detection_parameters[\"amplitude_threshold_meters\"],\n"
            "    max_abs_latitude_degrees=eddy_detection_parameters[\"max_abs_latitude_degrees\"],\n"
            "    contour_level_step_meters=eddy_detection_parameters[\"contour_level_step_meters\"],\n"
            "    min_contour_pixel_count=eddy_detection_parameters[\"min_contour_pixel_count\"],\n"
            "    max_contour_pixel_count=eddy_detection_parameters[\"max_contour_pixel_count\"],\n"
            "    min_contour_convexity=eddy_detection_parameters[\"min_contour_convexity\"],\n"
            ")\n"
            "glorys_eddy_contours = oceanbench.eddies.mesoscale_eddy_contours_from_detections(\n"
            "    raw_glorys_eddy_detections,\n"
            "    glorys_dataset,\n"
            "    background_sigma_grid=eddy_detection_parameters[\"background_sigma_grid\"],\n"
            "    detection_sigma_grid=eddy_detection_parameters[\"detection_sigma_grid\"],\n"
            "    amplitude_threshold_meters=eddy_detection_parameters[\"amplitude_threshold_meters\"],\n"
            "    max_abs_latitude_degrees=eddy_detection_parameters[\"max_abs_latitude_degrees\"],\n"
            "    contour_level_step_meters=eddy_detection_parameters[\"contour_level_step_meters\"],\n"
            "    min_contour_pixel_count=eddy_detection_parameters[\"min_contour_pixel_count\"],\n"
            "    max_contour_pixel_count=eddy_detection_parameters[\"max_contour_pixel_count\"],\n"
            "    min_contour_convexity=eddy_detection_parameters[\"min_contour_convexity\"],\n"
            ")\n"
            "challenger_eddy_detections = oceanbench.eddies.filter_mesoscale_eddy_detections_by_contours(\n"
            "    raw_challenger_eddy_detections,\n"
            "    challenger_eddy_contours,\n"
            ")\n"
            "glorys_eddy_detections = oceanbench.eddies.filter_mesoscale_eddy_detections_by_contours(\n"
            "    raw_glorys_eddy_detections,\n"
            "    glorys_eddy_contours,\n"
            ")\n"
            "eddy_matches = oceanbench.eddies.match_mesoscale_eddies(\n"
            "    challenger_eddy_detections,\n"
            "    glorys_eddy_detections,\n"
            "    max_match_distance_km=eddy_detection_parameters[\"max_match_distance_km\"],\n"
            ")\n"
            "challenger_eddy_detections.head()"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Number of cyclones and anticyclones, and hits and misses compared to GLORYS"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.eddies.mesoscale_eddy_summary_from_detections(\n"
            "    challenger_eddy_detections,\n"
            "    glorys_eddy_detections,\n"
            "    eddy_matches,\n"
            "    lead_day_count=challenger_dataset.sizes[\"lead_day_index\"],\n"
            "    challenger_name=\"GLONET\",\n"
            "    reference_name=\"GLORYS\",\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Mesoscale eddy detections over lead day 1 and lead day 10"
        ),
        nbformat.v4.new_code_cell(
            "mesoscale_eddy_overlay_figure = oceanbench.visualization.plot_mesoscale_eddy_overlay_gallery(\n"
            "    challenger_dataset,\n"
            "    glorys_dataset,\n"
            "    reference_name=\"GLORYS reanalysis\",\n"
            "    lead_day_indices=[0, 9],\n"
            "    challenger_detections=challenger_eddy_detections,\n"
            "    reference_detections=glorys_eddy_detections,\n"
            "    challenger_contours=challenger_eddy_contours,\n"
            "    reference_contours=glorys_eddy_contours,\n"
            "    **eddy_detection_parameters,\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Mesoscale eddy concentration over the 10-day forecast"
        ),
        nbformat.v4.new_code_cell(
            "mesoscale_eddy_concentration_figure = oceanbench.visualization.plot_mesoscale_eddy_concentration_gallery(\n"
            "    challenger_dataset,\n"
            "    glorys_dataset,\n"
            "    reference_name=\"GLORYS reanalysis\",\n"
            "    challenger_detections=challenger_eddy_detections,\n"
            "    reference_detections=glorys_eddy_detections,\n"
            "    challenger_contours=challenger_eddy_contours,\n"
            "    reference_contours=glorys_eddy_contours,\n"
            "    **eddy_detection_parameters,\n"
            ")"
        ),
    ]


def build_notebook(output_path: Path | None = None) -> Path:
    glonet.ensure_local_eddy_demo_data_exists()

    project_root = glonet.project_root()
    notebook_output_path = output_path or project_root / "assets" / "glonet_eddy_demo.report.ipynb"

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(_environment_setup_code()),
            nbformat.v4.new_code_cell("import oceanbench\nfrom oceanbench.demo import glonet\n\noceanbench.__version__"),
            nbformat.v4.new_markdown_cell(
                "### Open challenger dataset\n\n"
                "Local GLONET eddy demo sample cached under `demo_data/glonet_eddy_sample/`."
            ),
            nbformat.v4.new_code_cell(_loader_code()),
            nbformat.v4.new_code_cell(_reference_setup_code()),
            *_eddy_cells(),
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
