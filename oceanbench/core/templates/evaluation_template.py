import oceanbench

oceanbench.__version__

# ### Open challenger datasets

# > Insert here the code that opens the challenger dataset as `challenger_dataset: xarray.Dataset`

import xarray

challenger_dataset: xarray.Dataset = xarray.Dataset()

# ### Evaluation configuration

region = "global"

# ### Evaluation setup

import oceanbench.visualization

import warnings
from dask.array.core import PerformanceWarning

from oceanbench.core.dataset_utils import Variable
from oceanbench.core.derived_quantities import compute_geostrophic_currents, compute_mixed_layer_depth
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.rmsd import rmsd

regional_challenger_dataset = oceanbench.regions.subset(challenger_dataset, region)
forecast_comparison_variables = [
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    Variable.SEA_WATER_SALINITY,
    Variable.EASTWARD_SEA_WATER_VELOCITY,
    Variable.NORTHWARD_SEA_WATER_VELOCITY,
]
dynamic_diagnostic_variables = [
    Variable.MIXED_LAYER_DEPTH,
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
]
geostrophic_current_variables = [
    Variable.GEOSTROPHIC_NORTHWARD_SEA_WATER_VELOCITY,
    Variable.GEOSTROPHIC_EASTWARD_SEA_WATER_VELOCITY,
]

# ### Evaluation of challenger dataset using OceanBench

# #### Open GLORYS reanalysis reference dataset

glorys_dataset = oceanbench.regions.subset(glorys_reanalysis_dataset(regional_challenger_dataset), region)

# #### Root Mean Square Deviation (RMSD) of variables compared to GLORYS reanalysis

rmsd(
    challenger_dataset=regional_challenger_dataset,
    reference_dataset=glorys_dataset,
    variables=forecast_comparison_variables,
)

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS reanalysis

challenger_mld_dataset = compute_mixed_layer_depth(regional_challenger_dataset)
glorys_mld_dataset = compute_mixed_layer_depth(glorys_dataset)

rmsd(
    challenger_dataset=challenger_mld_dataset,
    reference_dataset=glorys_mld_dataset,
    variables=[Variable.MIXED_LAYER_DEPTH],
)

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS reanalysis

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=PerformanceWarning, message="Increasing number of chunks.*")
    challenger_geostrophic_dataset = compute_geostrophic_currents(regional_challenger_dataset)
    glorys_geostrophic_dataset = compute_geostrophic_currents(glorys_dataset)

rmsd(
    challenger_dataset=challenger_geostrophic_dataset,
    reference_dataset=glorys_geostrophic_dataset,
    variables=geostrophic_current_variables,
)

# #### Root Mean Square Deviation (RMSD) of variables compared to observations

oceanbench.metrics.rmsd_of_variables_compared_to_observations(
    challenger_dataset,
    region=region,
)

# #### Deviation of Lagrangian trajectories compared to GLORYS reanalysis

oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
    challenger_dataset,
    region=region,
)

# #### Open GLO12 analysis reference dataset

glo12_dataset = oceanbench.regions.subset(glo12_analysis_dataset(regional_challenger_dataset), region)

# #### Root Mean Square Deviation (RMSD) of variables compared to GLO12 analysis

rmsd(
    challenger_dataset=regional_challenger_dataset,
    reference_dataset=glo12_dataset,
    variables=forecast_comparison_variables,
)

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO12 analysis

glo12_mld_dataset = compute_mixed_layer_depth(glo12_dataset)

rmsd(
    challenger_dataset=challenger_mld_dataset,
    reference_dataset=glo12_mld_dataset,
    variables=[Variable.MIXED_LAYER_DEPTH],
)

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLO12 analysis

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=PerformanceWarning, message="Increasing number of chunks.*")
    glo12_geostrophic_dataset = compute_geostrophic_currents(glo12_dataset)

rmsd(
    challenger_dataset=challenger_geostrophic_dataset,
    reference_dataset=glo12_geostrophic_dataset,
    variables=geostrophic_current_variables,
)

# #### Deviation of Lagrangian trajectories compared to GLO12 analysis

oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
    challenger_dataset,
    region=region,
)

# ### Lagrangian trajectory divergence

# > Animated trajectories use a sampled subset of particles.
# > Motion is smooth visual interpolation between true daily positions.

lagrangian_trajectory_explorer = oceanbench.visualization.plot_multi_reference_lagrangian_trajectory_explorer(
    regional_challenger_dataset,
    {
        "GLORYS reanalysis": glorys_dataset,
        "GLO12 analysis": glo12_dataset,
    },
    particle_count=300,
)
lagrangian_trajectory_explorer

# ### Forecast comparison maps

forecast_comparison_explorer = oceanbench.visualization.plot_multi_reference_surface_comparison_explorer(
    regional_challenger_dataset,
    {
        "GLORYS reanalysis": glorys_dataset,
        "GLO12 analysis": glo12_dataset,
    },
    variables=forecast_comparison_variables,
    title="Forecast comparison maps",
)

# > Interactive map loading... The report can pause here while the browser parses the embedded map images.

forecast_comparison_explorer

# ### Dynamic diagnostic maps

# > Geostrophic currents are masked near the equator where the Coriolis parameter is too small.

challenger_dynamic_dataset = xarray.merge([challenger_mld_dataset, challenger_geostrophic_dataset])
glorys_dynamic_dataset = xarray.merge([glorys_mld_dataset, glorys_geostrophic_dataset])
glo12_dynamic_dataset = xarray.merge([glo12_mld_dataset, glo12_geostrophic_dataset])

dynamic_diagnostic_explorer = oceanbench.visualization.plot_multi_reference_surface_comparison_explorer(
    challenger_dynamic_dataset,
    {
        "GLORYS reanalysis": glorys_dynamic_dataset,
        "GLO12 analysis": glo12_dynamic_dataset,
    },
    variables=dynamic_diagnostic_variables,
    title="Dynamic diagnostic maps",
)

# > Interactive map loading... The report can pause here while the browser parses the embedded map images.

dynamic_diagnostic_explorer

# ### Spectral diagnostic

zonal_psd_figure = oceanbench.visualization.plot_multi_reference_zonal_psd_comparison(
    regional_challenger_dataset,
    {
        "GLORYS reanalysis": glorys_dataset,
        "GLO12 analysis": glo12_dataset,
    },
    variables=[
        Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
        Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    ],
)
None
