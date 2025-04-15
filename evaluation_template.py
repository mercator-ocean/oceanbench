import oceanbench

oceanbench.__version__

# ### Open challenger datasets

# > Insert here the code that opens the challenger datasets as `challenger_datasets: List[xarray.Dataset]`

import xarray
from typing import List

challenger_datasets: List[xarray.Dataset] = ...

# ### Evaluation of challenger datasets using OceanBench

# #### Root Mean Square Error (RMSE) compared to GLORYS variables

oceanbench.metrics.rmse_compared_to_glorys_variables(challenger_datasets)

# #### Root Mean Square Error (RMSE) compared to GLORYS Mixed Layer Depth (MLD)

oceanbench.metrics.rmse_compared_to_glorys_mixed_layer_depth(challenger_datasets)

# #### Geostrophic current analysis

oceanbench.derived_quantities.geostrophic_currents(challenger_datasets)

# #### Density analysis

oceanbench.derived_quantities.density(challenger_datasets)

# #### Euclidean distance to GLORYS reference

oceanbench.metrics.euclidean_distance_to_glorys(challenger_datasets)

# #### Energy cascading analysis

oceanbench.metrics.energy_cascade(challenger_datasets)

# #### Kinetic energy analysis

oceanbench.derived_quantities.kinetic_energy(challenger_datasets)

# #### Vorticity analysis
oceanbench.derived_quantities.vorticity(challenger_datasets)

# #### Mass conservation analysis

oceanbench.derived_quantities.mass_conservation(challenger_datasets)
