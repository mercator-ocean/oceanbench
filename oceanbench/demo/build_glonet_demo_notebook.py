# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
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
    return (glonet.project_root() / "assets" / "glonet_demo_sample.py").read_text(encoding="utf8")


def _reference_setup_code() -> str:
    return """from oceanbench.demo import glonet
from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents
from oceanbench.core.derived_quantities import compute_mixed_layer_depth
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import observations

glorys_dataset = glorys_reanalysis_dataset(challenger_dataset)
glo12_dataset = glo12_analysis_dataset(challenger_dataset)
observation_dataset = observations(challenger_dataset)

challenger_eddy_dataset = glonet.load_local_eddy_challenger_dataset()
glorys_eddy_dataset = glonet.load_local_eddy_glorys_dataset()
glo12_eddy_dataset = glonet.load_local_eddy_glo12_dataset()

challenger_mld_dataset = compute_mixed_layer_depth(challenger_dataset)
glorys_mld_dataset = compute_mixed_layer_depth(glorys_dataset)
glo12_mld_dataset = compute_mixed_layer_depth(glo12_dataset)

challenger_geostrophic_dataset = compute_geostrophic_currents(challenger_dataset)
glorys_geostrophic_dataset = compute_geostrophic_currents(glorys_dataset)
glo12_geostrophic_dataset = compute_geostrophic_currents(glo12_dataset)

eddy_detection_parameters = oceanbench.eddies.default_eddy_detection_parameters()
"""


def _evaluation_cells() -> list[nbformat.NotebookNode]:
    return [
        nbformat.v4.new_markdown_cell(
            "### Evaluation of challenger dataset using OceanBench"
        ),
        nbformat.v4.new_markdown_cell(
            "<h1 style=\"text-align: center;\">REANALYSIS Track</h1>\n\n"
            "#### Root Mean Square Deviation (RMSD) of variables compared to GLORYS reanalysis"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.metrics.rmsd_of_variables_compared_to_glorys_reanalysis(challenger_dataset)"
        ),
        nbformat.v4.new_code_cell(
            "glorys_variable_rmse_figure = oceanbench.visualization.plot_spatial_rmse_gallery(\n"
            "    challenger_dataset,\n"
            "    glorys_dataset,\n"
            "    \"GLORYS reanalysis\",\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS reanalysis"
        ),
        nbformat.v4.new_code_cell(
            "glorys_mld_field_figure = oceanbench.visualization.plot_surface_field_gallery(\n"
            "    challenger_mld_dataset,\n"
            "    variables=[Variable.MIXED_LAYER_DEPTH],\n"
            ")"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(challenger_dataset)"
        ),
        nbformat.v4.new_code_cell(
            "glorys_mld_rmse_figure = oceanbench.visualization.plot_spatial_rmse_gallery(\n"
            "    challenger_mld_dataset,\n"
            "    glorys_mld_dataset,\n"
            "    \"GLORYS reanalysis\",\n"
            "    variables=[Variable.MIXED_LAYER_DEPTH],\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS reanalysis"
        ),
        nbformat.v4.new_code_cell(
            "glorys_geostrophic_field_figure = oceanbench.visualization.plot_surface_field_gallery(\n"
            "    challenger_geostrophic_dataset,\n"
            "    variables=[\n"
            "        Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,\n"
            "        Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,\n"
            "    ],\n"
            ")"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(challenger_dataset)"
        ),
        nbformat.v4.new_code_cell(
            "glorys_geostrophic_rmse_figure = oceanbench.visualization.plot_spatial_rmse_gallery(\n"
            "    challenger_geostrophic_dataset,\n"
            "    glorys_geostrophic_dataset,\n"
            "    \"GLORYS reanalysis\",\n"
            "    variables=[\n"
            "        Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,\n"
            "        Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,\n"
            "    ],\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Deviation of Lagrangian trajectories compared to GLORYS reanalysis"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(challenger_dataset)"
        ),
        nbformat.v4.new_code_cell(
            "glorys_lagrangian_figure = oceanbench.visualization.plot_lagrangian_trajectory_comparison(\n"
            "    challenger_dataset,\n"
            "    glorys_dataset,\n"
            "    \"GLORYS reanalysis\",\n"
            "    particle_percentage=10.0,\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Accepted mesoscale eddies compared to GLORYS reanalysis"
        ),
        nbformat.v4.new_code_cell(
            "raw_challenger_eddy_detections = oceanbench.eddies.detect_mesoscale_eddies(\n"
            "    challenger_eddy_dataset,\n"
            "    background_sigma_grid=eddy_detection_parameters[\"background_sigma_grid\"],\n"
            "    detection_sigma_grid=eddy_detection_parameters[\"detection_sigma_grid\"],\n"
            "    min_distance_grid=eddy_detection_parameters[\"min_distance_grid\"],\n"
            "    amplitude_threshold_meters=eddy_detection_parameters[\"amplitude_threshold_meters\"],\n"
            "    max_abs_latitude_degrees=eddy_detection_parameters[\"max_abs_latitude_degrees\"],\n"
            ")\n"
            "challenger_eddy_contours = oceanbench.eddies.mesoscale_eddy_contours_from_detections(\n"
            "    raw_challenger_eddy_detections,\n"
            "    challenger_eddy_dataset,\n"
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
            "raw_glorys_eddy_detections = oceanbench.eddies.detect_mesoscale_eddies(\n"
            "    glorys_eddy_dataset,\n"
            "    background_sigma_grid=eddy_detection_parameters[\"background_sigma_grid\"],\n"
            "    detection_sigma_grid=eddy_detection_parameters[\"detection_sigma_grid\"],\n"
            "    min_distance_grid=eddy_detection_parameters[\"min_distance_grid\"],\n"
            "    amplitude_threshold_meters=eddy_detection_parameters[\"amplitude_threshold_meters\"],\n"
            "    max_abs_latitude_degrees=eddy_detection_parameters[\"max_abs_latitude_degrees\"],\n"
            ")\n"
            "glorys_eddy_contours = oceanbench.eddies.mesoscale_eddy_contours_from_detections(\n"
            "    raw_glorys_eddy_detections,\n"
            "    glorys_eddy_dataset,\n"
            "    background_sigma_grid=eddy_detection_parameters[\"background_sigma_grid\"],\n"
            "    detection_sigma_grid=eddy_detection_parameters[\"detection_sigma_grid\"],\n"
            "    amplitude_threshold_meters=eddy_detection_parameters[\"amplitude_threshold_meters\"],\n"
            "    max_abs_latitude_degrees=eddy_detection_parameters[\"max_abs_latitude_degrees\"],\n"
            "    contour_level_step_meters=eddy_detection_parameters[\"contour_level_step_meters\"],\n"
            "    min_contour_pixel_count=eddy_detection_parameters[\"min_contour_pixel_count\"],\n"
            "    max_contour_pixel_count=eddy_detection_parameters[\"max_contour_pixel_count\"],\n"
            "    min_contour_convexity=eddy_detection_parameters[\"min_contour_convexity\"],\n"
            ")\n"
            "glorys_eddy_detections = oceanbench.eddies.filter_mesoscale_eddy_detections_by_contours(\n"
            "    raw_glorys_eddy_detections,\n"
            "    glorys_eddy_contours,\n"
            ")\n"
            "glorys_eddy_matches = oceanbench.eddies.match_mesoscale_eddies(\n"
            "    challenger_eddy_detections,\n"
            "    glorys_eddy_detections,\n"
            "    max_match_distance_km=eddy_detection_parameters[\"max_match_distance_km\"],\n"
            ")\n"
            "challenger_eddy_detections.head()"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.eddies.mesoscale_eddy_summary_from_detections(\n"
            "    challenger_eddy_detections,\n"
            "    glorys_eddy_detections,\n"
            "    glorys_eddy_matches,\n"
            "    lead_day_count=challenger_eddy_dataset.sizes[\"lead_day_index\"],\n"
            "    challenger_name=\"GLONET\",\n"
            "    reference_name=\"GLORYS\",\n"
            ")"
        ),
        nbformat.v4.new_code_cell(
            "glorys_eddy_overlay_figure = oceanbench.visualization.plot_mesoscale_eddy_overlay_gallery(\n"
            "    challenger_eddy_dataset,\n"
            "    glorys_eddy_dataset,\n"
            "    reference_name=\"GLORYS reanalysis\",\n"
            "    lead_day_indices=[0, 9],\n"
            "    challenger_detections=challenger_eddy_detections,\n"
            "    reference_detections=glorys_eddy_detections,\n"
            "    challenger_contours=challenger_eddy_contours,\n"
            "    reference_contours=glorys_eddy_contours,\n"
            "    **eddy_detection_parameters,\n"
            ")"
        ),
        nbformat.v4.new_code_cell(
            "glorys_eddy_concentration_figure = oceanbench.visualization.plot_mesoscale_eddy_concentration_gallery(\n"
            "    challenger_eddy_dataset,\n"
            "    glorys_eddy_dataset,\n"
            "    reference_name=\"GLORYS reanalysis\",\n"
            "    challenger_detections=challenger_eddy_detections,\n"
            "    reference_detections=glorys_eddy_detections,\n"
            "    challenger_contours=challenger_eddy_contours,\n"
            "    reference_contours=glorys_eddy_contours,\n"
            "    **eddy_detection_parameters,\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "<h1 style=\"text-align: center;\">ANALYSIS Track</h1>\n\n"
            "#### Root Mean Square Deviation (RMSD) of variables compared to GLO12 analysis"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.metrics.rmsd_of_variables_compared_to_glo12_analysis(challenger_dataset)"
        ),
        nbformat.v4.new_code_cell(
            "glo12_variable_rmse_figure = oceanbench.visualization.plot_spatial_rmse_gallery(\n"
            "    challenger_dataset,\n"
            "    glo12_dataset,\n"
            "    \"GLO12 analysis\",\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO12 analysis"
        ),
        nbformat.v4.new_code_cell(
            "glo12_mld_field_figure = oceanbench.visualization.plot_surface_field_gallery(\n"
            "    challenger_mld_dataset,\n"
            "    variables=[Variable.MIXED_LAYER_DEPTH],\n"
            ")"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(challenger_dataset)"
        ),
        nbformat.v4.new_code_cell(
            "glo12_mld_rmse_figure = oceanbench.visualization.plot_spatial_rmse_gallery(\n"
            "    challenger_mld_dataset,\n"
            "    glo12_mld_dataset,\n"
            "    \"GLO12 analysis\",\n"
            "    variables=[Variable.MIXED_LAYER_DEPTH],\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLO12 analysis"
        ),
        nbformat.v4.new_code_cell(
            "glo12_geostrophic_field_figure = oceanbench.visualization.plot_surface_field_gallery(\n"
            "    challenger_geostrophic_dataset,\n"
            "    variables=[\n"
            "        Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,\n"
            "        Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,\n"
            "    ],\n"
            ")"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glo12_analysis(challenger_dataset)"
        ),
        nbformat.v4.new_code_cell(
            "glo12_geostrophic_rmse_figure = oceanbench.visualization.plot_spatial_rmse_gallery(\n"
            "    challenger_geostrophic_dataset,\n"
            "    glo12_geostrophic_dataset,\n"
            "    \"GLO12 analysis\",\n"
            "    variables=[\n"
            "        Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,\n"
            "        Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,\n"
            "    ],\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Deviation of Lagrangian trajectories compared to GLO12 analysis"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(challenger_dataset)"
        ),
        nbformat.v4.new_code_cell(
            "glo12_lagrangian_figure = oceanbench.visualization.plot_lagrangian_trajectory_comparison(\n"
            "    challenger_dataset,\n"
            "    glo12_dataset,\n"
            "    \"GLO12 analysis\",\n"
            "    particle_percentage=10.0,\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Accepted mesoscale eddies compared to GLO12 analysis"
        ),
        nbformat.v4.new_code_cell(
            "raw_glo12_eddy_detections = oceanbench.eddies.detect_mesoscale_eddies(\n"
            "    glo12_eddy_dataset,\n"
            "    background_sigma_grid=eddy_detection_parameters[\"background_sigma_grid\"],\n"
            "    detection_sigma_grid=eddy_detection_parameters[\"detection_sigma_grid\"],\n"
            "    min_distance_grid=eddy_detection_parameters[\"min_distance_grid\"],\n"
            "    amplitude_threshold_meters=eddy_detection_parameters[\"amplitude_threshold_meters\"],\n"
            "    max_abs_latitude_degrees=eddy_detection_parameters[\"max_abs_latitude_degrees\"],\n"
            ")\n"
            "glo12_eddy_contours = oceanbench.eddies.mesoscale_eddy_contours_from_detections(\n"
            "    raw_glo12_eddy_detections,\n"
            "    glo12_eddy_dataset,\n"
            "    background_sigma_grid=eddy_detection_parameters[\"background_sigma_grid\"],\n"
            "    detection_sigma_grid=eddy_detection_parameters[\"detection_sigma_grid\"],\n"
            "    amplitude_threshold_meters=eddy_detection_parameters[\"amplitude_threshold_meters\"],\n"
            "    max_abs_latitude_degrees=eddy_detection_parameters[\"max_abs_latitude_degrees\"],\n"
            "    contour_level_step_meters=eddy_detection_parameters[\"contour_level_step_meters\"],\n"
            "    min_contour_pixel_count=eddy_detection_parameters[\"min_contour_pixel_count\"],\n"
            "    max_contour_pixel_count=eddy_detection_parameters[\"max_contour_pixel_count\"],\n"
            "    min_contour_convexity=eddy_detection_parameters[\"min_contour_convexity\"],\n"
            ")\n"
            "glo12_eddy_detections = oceanbench.eddies.filter_mesoscale_eddy_detections_by_contours(\n"
            "    raw_glo12_eddy_detections,\n"
            "    glo12_eddy_contours,\n"
            ")\n"
            "glo12_eddy_matches = oceanbench.eddies.match_mesoscale_eddies(\n"
            "    challenger_eddy_detections,\n"
            "    glo12_eddy_detections,\n"
            "    max_match_distance_km=eddy_detection_parameters[\"max_match_distance_km\"],\n"
            ")\n"
            "glo12_eddy_detections.head()"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.eddies.mesoscale_eddy_summary_from_detections(\n"
            "    challenger_eddy_detections,\n"
            "    glo12_eddy_detections,\n"
            "    glo12_eddy_matches,\n"
            "    lead_day_count=challenger_eddy_dataset.sizes[\"lead_day_index\"],\n"
            "    challenger_name=\"GLONET\",\n"
            "    reference_name=\"GLO12\",\n"
            ")"
        ),
        nbformat.v4.new_code_cell(
            "glo12_eddy_overlay_figure = oceanbench.visualization.plot_mesoscale_eddy_overlay_gallery(\n"
            "    challenger_eddy_dataset,\n"
            "    glo12_eddy_dataset,\n"
            "    reference_name=\"GLO12 analysis\",\n"
            "    lead_day_indices=[0, 9],\n"
            "    challenger_detections=challenger_eddy_detections,\n"
            "    reference_detections=glo12_eddy_detections,\n"
            "    challenger_contours=challenger_eddy_contours,\n"
            "    reference_contours=glo12_eddy_contours,\n"
            "    **eddy_detection_parameters,\n"
            ")"
        ),
        nbformat.v4.new_code_cell(
            "glo12_eddy_concentration_figure = oceanbench.visualization.plot_mesoscale_eddy_concentration_gallery(\n"
            "    challenger_eddy_dataset,\n"
            "    glo12_eddy_dataset,\n"
            "    reference_name=\"GLO12 analysis\",\n"
            "    challenger_detections=challenger_eddy_detections,\n"
            "    reference_detections=glo12_eddy_detections,\n"
            "    challenger_contours=challenger_eddy_contours,\n"
            "    reference_contours=glo12_eddy_contours,\n"
            "    **eddy_detection_parameters,\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "<h1 style=\"text-align: center;\">CLASS-4 Track</h1>\n\n"
            "#### Root Mean Square Deviation (RMSD) of variables compared to observations"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.metrics.rmsd_of_variables_compared_to_observations(challenger_dataset)"
        ),
        nbformat.v4.new_code_cell(
            "class4_scatter_figure = oceanbench.visualization.plot_class4_scatter_gallery(\n"
            "    challenger_dataset,\n"
            "    observation_dataset,\n"
            "    render_mode=\"grid\",\n"## or scatter,grid
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Deviation of Lagrangian trajectories compared to Class-4 drifter observations"
        ),
        nbformat.v4.new_code_cell(
            "oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_class4_observations(challenger_dataset)"
        ),
        nbformat.v4.new_code_cell(
            "class4_drifter_lagrangian_figure = oceanbench.visualization.plot_class4_drifter_trajectory_comparison(\n"
            "    challenger_dataset,\n"
            "    observation_dataset,\n"
            "    particle_percentage=20.0,\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "<h1 style=\"text-align: center;\">PSD Analysis</h1>\n\n"
            "#### Power Spectral Density (PSD) of SSH and SST compared to GLORYS reanalysis"
        ),
        nbformat.v4.new_code_cell(
            "from oceanbench.core.dataset_utils import Variable\n"
            "import pandas\n\n"
            "challenger_ssh_psd, glorys_ssh_psd = oceanbench.psd.zonal_longitude_psd_pair(\n"
            "    challenger_dataset,\n"
            "    glorys_dataset,\n"
            "    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,\n"
            ")\n"
            "challenger_sst_psd, glorys_sst_psd = oceanbench.psd.zonal_longitude_psd_pair(\n"
            "    challenger_dataset,\n"
            "    glorys_dataset,\n"
            "    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,\n"
            ")\n"
            "zonal_wavelength_bands_km = oceanbench.psd.default_zonal_wavelength_bands_km(challenger_ssh_psd)\n"
            "zonal_wavelength_bands_km"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Zonal PSD comparison for day 1 and day 10 against GLORYS reanalysis"
        ),
        nbformat.v4.new_code_cell(
            "glorys_zonal_psd_figure = oceanbench.visualization.plot_zonal_longitude_psd_comparison_gallery(\n"
            "    challenger_dataset,\n"
            "    glorys_dataset,\n"
            "    \"GLORYS reanalysis\",\n"
            "    lead_day_indices=[0, 9],\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### SSH zonal PSD scores for all lead days compared to GLORYS reanalysis"
        ),
        nbformat.v4.new_code_cell(
            "glorys_ssh_psd_scores = pandas.concat(\n"
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
            "glorys_ssh_psd_scores"
        ),
        nbformat.v4.new_markdown_cell(
            "#### SST zonal PSD scores for all lead days compared to GLORYS reanalysis"
        ),
        nbformat.v4.new_code_cell(
            "glorys_sst_psd_scores = pandas.concat(\n"
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
            "glorys_sst_psd_scores"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Power Spectral Density (PSD) of SSH and SST compared to GLO12 analysis"
        ),
        nbformat.v4.new_code_cell(
            "challenger_ssh_psd, glo12_ssh_psd = oceanbench.psd.zonal_longitude_psd_pair(\n"
            "    challenger_dataset,\n"
            "    glo12_dataset,\n"
            "    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,\n"
            ")\n"
            "challenger_sst_psd, glo12_sst_psd = oceanbench.psd.zonal_longitude_psd_pair(\n"
            "    challenger_dataset,\n"
            "    glo12_dataset,\n"
            "    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### Zonal PSD comparison for day 1 and day 10 against GLO12 analysis"
        ),
        nbformat.v4.new_code_cell(
            "glo12_zonal_psd_figure = oceanbench.visualization.plot_zonal_longitude_psd_comparison_gallery(\n"
            "    challenger_dataset,\n"
            "    glo12_dataset,\n"
            "    \"GLO12 analysis\",\n"
            "    lead_day_indices=[0, 9],\n"
            ")"
        ),
        nbformat.v4.new_markdown_cell(
            "#### SSH zonal PSD scores for all lead days compared to GLO12 analysis"
        ),
        nbformat.v4.new_code_cell(
            "glo12_ssh_psd_scores = pandas.concat(\n"
            "    {\n"
            "        \"GLONET\": oceanbench.psd.zonal_longitude_psd_metrics_from_spectrum(\n"
            "            challenger_ssh_psd,\n"
            "            wavelength_bands_km=zonal_wavelength_bands_km,\n"
            "        ),\n"
            "        \"GLO12\": oceanbench.psd.zonal_longitude_psd_metrics_from_spectrum(\n"
            "            glo12_ssh_psd,\n"
            "            wavelength_bands_km=zonal_wavelength_bands_km,\n"
            "        ),\n"
            "    }\n"
            ")\n"
            "glo12_ssh_psd_scores"
        ),
        nbformat.v4.new_markdown_cell(
            "#### SST zonal PSD scores for all lead days compared to GLO12 analysis"
        ),
        nbformat.v4.new_code_cell(
            "glo12_sst_psd_scores = pandas.concat(\n"
            "    {\n"
            "        \"GLONET\": oceanbench.psd.zonal_longitude_psd_metrics_from_spectrum(\n"
            "            challenger_sst_psd,\n"
            "            wavelength_bands_km=zonal_wavelength_bands_km,\n"
            "        ),\n"
            "        \"GLO12\": oceanbench.psd.zonal_longitude_psd_metrics_from_spectrum(\n"
            "            glo12_sst_psd,\n"
            "            wavelength_bands_km=zonal_wavelength_bands_km,\n"
            "        ),\n"
            "    }\n"
            ")\n"
            "glo12_sst_psd_scores"
        ),
    ]


def build_notebook(output_path: Path | None = None) -> Path:
    glonet.ensure_local_demo_data_exists()
    glonet.ensure_local_eddy_demo_data_exists()

    project_root = glonet.project_root()
    notebook_output_path = output_path or project_root / "assets" / "glonet_demo_sample.report.ipynb"

    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_code_cell(_environment_setup_code()),
            nbformat.v4.new_code_cell("import oceanbench\n\noceanbench.__version__"),
            nbformat.v4.new_markdown_cell(
                "### Open challenger datasets\n\n"
                "Local GLONET demo sample cached under `demo_data/glonet_sample/`."
            ),
            nbformat.v4.new_code_cell(_loader_code()),
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
