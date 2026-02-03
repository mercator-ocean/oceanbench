import oceanbench

oceanbench.__version__

# ### Open challenger datasets

# > Insert here the code that opens the challenger dataset as `challenger_dataset: xarray.Dataset`

import xarray

challenger_dataset: xarray.Dataset = xarray.Dataset()

# ### Evaluation of challenger dataset using OceanBench

# #### Root Mean Square Deviation (RMSD) of variables compared to observations

oceanbench.metrics.rmsd_of_variables_compared_to_observations(challenger_dataset)
