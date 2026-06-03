# ### Report guide

# Score tables provide the quantitative OceanBench evaluation.

# Interactive figures help diagnose the scores by showing spatial errors, lead-time evolution,
# dynamic diagnostics, Lagrangian drift, eddy matching, and Class IV observation errors.

# Some figure layers are downsampled or compressed for responsive notebook and website rendering;
# metric scores are computed from the underlying datasets.

import oceanbench

oceanbench.__version__

# ### Open challenger datasets

# > Insert here the code that opens the challenger dataset as `challenger_dataset: xarray.Dataset`

import xarray

challenger_dataset: xarray.Dataset = xarray.Dataset()

# ### Evaluation configuration

region = "global"

# ### Evaluation setup

from oceanbench.core.evaluation_report import prepare_evaluation_report

evaluation_report = prepare_evaluation_report(challenger_dataset, region=region)

# ### Evaluation of challenger dataset using OceanBench

# #### Root Mean Square Deviation (RMSD) of variables compared to GLORYS reanalysis

evaluation_report.glorys_variable_rmsd

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLORYS reanalysis

evaluation_report.glorys_mixed_layer_depth_rmsd

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLORYS reanalysis

evaluation_report.glorys_geostrophic_current_rmsd

# #### Root Mean Square Deviation (RMSD) of variables compared to observations

evaluation_report.class4_observation.rmsd

# #### Deviation of Lagrangian trajectories compared to Class IV drifter observations

evaluation_report.class4_drifter_trajectory_deviation

# #### Class IV observation error explorer

evaluation_report.class4_observation_error_explorer

# #### Class IV drifter trajectory explorer

evaluation_report.class4_drifter_trajectory_explorer

# #### Deviation of Lagrangian trajectories compared to GLORYS reanalysis

evaluation_report.glorys_lagrangian_trajectory_deviation

# #### Root Mean Square Deviation (RMSD) of variables compared to GLO12 analysis

evaluation_report.glo12_variable_rmsd

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO12 analysis

evaluation_report.glo12_mixed_layer_depth_rmsd

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLO12 analysis

evaluation_report.glo12_geostrophic_current_rmsd

# #### Deviation of Lagrangian trajectories compared to GLO12 analysis

evaluation_report.glo12_lagrangian_trajectory_deviation

# ### Lagrangian trajectory divergence

# > Animated trajectories use a sampled subset of particles.
# > Motion is smooth visual interpolation between true daily positions.

evaluation_report.lagrangian_trajectory_explorer

# ### Mesoscale eddy matching

# > Eddy centers and contours are detected from sea surface height anomalies.
# > Frames are discrete lead days; contours are not interpolated.

evaluation_report.eddy_matching_explorer

# ### Forecast comparison maps

# > Interactive map loading... The report can pause here while the browser parses the embedded map images.

evaluation_report.forecast_comparison_explorer

# ### Dynamic diagnostic maps

# > Geostrophic currents are masked near the equator where the Coriolis parameter is too small.

# > Interactive map loading... The report can pause here while the browser parses the embedded map images.

evaluation_report.dynamic_diagnostic_explorer

# ### Spectral diagnostic

# > Zonal spectra use metric longitude coordinates, linear detrending, Tukey windowing,
# > and wavelength-band scale separation.

evaluation_report.zonal_psd_explorer
