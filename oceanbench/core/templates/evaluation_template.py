import oceanbench
from oceanbench.core.memory_diagnostics import (
    default_memory_tracker,
    describe_dataset,
    enable_memory_diagnostics,
)

enable_memory_diagnostics()
memory_tracker = default_memory_tracker("notebook")
memory_tracker.checkpoint("Notebook started")
oceanbench.__version__

# ### Open challenger datasets

# > Insert here the code that opens the challenger dataset as `challenger_dataset: xarray.Dataset`

import xarray

challenger_dataset: xarray.Dataset = xarray.Dataset()

# #### RAM diagnostics

# > RAM diagnostics are enabled by default in this notebook template.
# > Optional: call `enable_memory_diagnostics(log_path="/tmp/oceanbench_mem.log")` to also write logs to a file.

memory_tracker.checkpoint("Challenger dataset loaded")
describe_dataset(challenger_dataset, "challenger_dataset", memory_tracker)

# ### Evaluation of challenger dataset using OceanBench

# #### Root Mean Square Deviation (RMSD) of variables compared to GLORYS reanalysis

with memory_tracker.step("rmsd_of_variables_compared_to_glorys_reanalysis"):
    _rmsd_glorys_variables = oceanbench.metrics.rmsd_of_variables_compared_to_glorys_reanalysis(challenger_dataset)
_rmsd_glorys_variables

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS reanalysis

with memory_tracker.step("rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis"):
    _rmsd_glorys_mld = oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glorys_reanalysis(challenger_dataset)
_rmsd_glorys_mld

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS reanalysis

with memory_tracker.step("rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis"):
    _rmsd_glorys_geostrophic = oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glorys_reanalysis(
        challenger_dataset
    )
_rmsd_glorys_geostrophic

# #### Deviation of Lagrangian trajectories compared to GLORYS reanalysis

with memory_tracker.step("deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis"):
    _lagrangian_glorys = oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glorys_reanalysis(
        challenger_dataset
    )
_lagrangian_glorys

# #### Root Mean Square Deviation (RMSD) of variables compared to GLO12 analysis

with memory_tracker.step("rmsd_of_variables_compared_to_glo12_analysis"):
    _rmsd_glo12_variables = oceanbench.metrics.rmsd_of_variables_compared_to_glo12_analysis(challenger_dataset)
_rmsd_glo12_variables

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO12 analysis

with memory_tracker.step("rmsd_of_mixed_layer_depth_compared_to_glo12_analysis"):
    _rmsd_glo12_mld = oceanbench.metrics.rmsd_of_mixed_layer_depth_compared_to_glo12_analysis(challenger_dataset)
_rmsd_glo12_mld

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLO12 analysis

with memory_tracker.step("rmsd_of_geostrophic_currents_compared_to_glo12_analysis"):
    _rmsd_glo12_geostrophic = oceanbench.metrics.rmsd_of_geostrophic_currents_compared_to_glo12_analysis(
        challenger_dataset
    )
_rmsd_glo12_geostrophic

# #### Deviation of Lagrangian trajectories compared to GLO12 analysis

with memory_tracker.step("deviation_of_lagrangian_trajectories_compared_to_glo12_analysis"):
    _lagrangian_glo12 = oceanbench.metrics.deviation_of_lagrangian_trajectories_compared_to_glo12_analysis(
        challenger_dataset
    )
_lagrangian_glo12

# #### Root Mean Square Deviation (RMSD) of variables compared to observations

with memory_tracker.step("rmsd_of_variables_compared_to_observations"):
    _rmsd_observations = oceanbench.metrics.rmsd_of_variables_compared_to_observations(challenger_dataset)

memory_tracker.checkpoint("Notebook completed")
_rmsd_observations
