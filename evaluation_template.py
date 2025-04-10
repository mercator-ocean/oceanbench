import oceanbench

oceanbench.__version__

# ### Open candidate datasets

# > Insert here the code that opens the candidate datasets as `candidate_datasets: List[xarray.Dataset]`

import xarray
from typing import List

candidate_datasets: List[xarray.Dataset] = ...

# ### Evaluation of candidate datasets using OceanBench

# #### Root Mean Square Error (RMSE) compared to GLORYS

oceanbench.metrics.rmse_to_glorys(candidate_datasets)

# #### Mixed Layer Depth (MLD) analysis

oceanbench.derived_quantities.mld(candidate_datasets)

# #### Geostrophic current analysis

oceanbench.derived_quantities.geostrophic_currents(candidate_datasets)

# #### Density analysis

oceanbench.derived_quantities.density(candidate_datasets)

# #### Euclidean distance to GLORYS reference

oceanbench.metrics.euclidean_distance_to_glorys(candidate_datasets)

# #### Energy cascading analysis

oceanbench.metrics.energy_cascade(candidate_datasets)

# #### Kinetic energy analysis

oceanbench.derived_quantities.kinetic_energy(candidate_datasets)

# #### Vorticity analysis
oceanbench.derived_quantities.vorticity(candidate_datasets)

# #### Mass conservation analysis

oceanbench.derived_quantities.mass_conservation(candidate_datasets)
