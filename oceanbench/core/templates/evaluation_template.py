import oceanbench

oceanbench.__version__

# ### Open challenger datasets

# > Insert here the code that opens the challenger dataset as `challenger_dataset: xarray.Dataset`

import xarray

challenger_dataset: xarray.Dataset = xarray.Dataset()

# ### Evaluation of challenger dataset using OceanBench

# #### Root Mean Square Deviation (RMSD) of variables compared to GLORYS reanalysis

oceanbench.metrics.rmsd_of_variables_compared_to_glorys_reanalysis(challenger_dataset)

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS reanalysis

oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(challenger_dataset)

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS reanalysis

oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(challenger_dataset)

# #### Deviation of Lagrangian trajectories compared to GLORYS reanalysis

oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(challenger_dataset)

# #### Root Mean Square Deviation (RMSD) of variables compared to GLO12 analysis

oceanbench.metrics.rmsd_of_variables_compared_to_glo12_analysis(challenger_dataset)

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO12 analysis

oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(challenger_dataset)

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLOR12 analysis

oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glo12_analysis(challenger_dataset)

# #### Deviation of Lagrangian trajectories compared to GLO12 analysis

oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(challenger_dataset)
