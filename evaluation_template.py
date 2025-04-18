import oceanbench

oceanbench.__version__

# ### Open challenger datasets

# > Insert here the code that opens the challenger datasets as `challenger_datasets: List[xarray.Dataset]`

import xarray
from typing import List

challenger_datasets: List[xarray.Dataset] = ...

# ### Evaluation of challenger datasets using OceanBench

# #### Root Mean Square Deviation (RMSD) of variables compared to GLORYS

oceanbench.metrics.rmsd_of_variables_compared_to_glorys(challenger_datasets)

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS

oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glorys(challenger_datasets)

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS

oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glorys(challenger_datasets)

# #### Deviation of Lagrangian trajectories compared to GLORYS

oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glorys(challenger_datasets)

# #### Density analysis

oceanbench.derived_quantities.density(challenger_datasets)

# #### Energy cascading analysis

oceanbench.metrics.energy_cascade(challenger_datasets)

# #### Kinetic energy analysis

oceanbench.derived_quantities.kinetic_energy(challenger_datasets)

# #### Vorticity analysis
oceanbench.derived_quantities.vorticity(challenger_datasets)

# #### Mass conservation analysis

oceanbench.derived_quantities.mass_conservation(challenger_datasets)
