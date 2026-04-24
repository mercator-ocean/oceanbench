import oceanbench

oceanbench.__version__

# ### Open challenger datasets

# > Insert here the code that opens the challenger dataset as `challenger_dataset: xarray.Dataset`

import xarray

challenger_dataset: xarray.Dataset = xarray.Dataset()

# ### Evaluation configuration

region = "global"

# ### Surface comparison maps

import oceanbench.visualization

from oceanbench.core.dataset_utils import Variable
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset

regional_challenger_dataset = oceanbench.regions.subset(challenger_dataset, region)
glorys_dataset = oceanbench.regions.subset(glorys_reanalysis_dataset(challenger_dataset), region)
glo12_dataset = oceanbench.regions.subset(glo12_analysis_dataset(challenger_dataset), region)
surface_comparison_variables = [
    Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID,
    Variable.SEA_WATER_POTENTIAL_TEMPERATURE,
    Variable.SEA_WATER_SALINITY,
    Variable.EASTWARD_SEA_WATER_VELOCITY,
    Variable.NORTHWARD_SEA_WATER_VELOCITY,
]

surface_comparison_explorer = oceanbench.visualization.plot_multi_reference_surface_comparison_explorer(
    regional_challenger_dataset,
    {
        "GLORYS reanalysis": glorys_dataset,
        "GLO12 analysis": glo12_dataset,
    },
    variables=surface_comparison_variables,
)
surface_comparison_explorer

# ### Evaluation of challenger dataset using OceanBench

# #### Root Mean Square Deviation (RMSD) of variables compared to GLORYS reanalysis

oceanbench.metrics.rmsd_of_variables_compared_to_glorys_reanalysis(
    challenger_dataset,
    region=region,
)

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS reanalysis

oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(
    challenger_dataset,
    region=region,
)

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS reanalysis

oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
    challenger_dataset,
    region=region,
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

# #### Root Mean Square Deviation (RMSD) of variables compared to GLO12 analysis

oceanbench.metrics.rmsd_of_variables_compared_to_glo12_analysis(
    challenger_dataset,
    region=region,
)

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO12 analysis

oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(
    challenger_dataset,
    region=region,
)

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLO12 analysis

oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
    challenger_dataset,
    region=region,
)

# #### Deviation of Lagrangian trajectories compared to GLO12 analysis

oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
    challenger_dataset,
    region=region,
)
