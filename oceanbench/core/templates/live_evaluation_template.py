# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# ### Live evaluation guide

# Live evaluation reports monitor one recent operational forecast against recent Class IV observations
# and GLO12 operational analysis. These diagnostics are for scientific validation and operational
# monitoring, not for annual benchmark ranking.

# Observation-based scores use the configured recent Class IV bucket. The default development bucket
# is the 2026 OceanBench observation prefix, capped before the recent current-observation gap.

from pathlib import Path
import sys

for repository_root in [Path.cwd(), *Path.cwd().parents]:
    if (repository_root / "oceanbench").is_dir():
        sys.path.insert(0, str(repository_root))
        break

import oceanbench

oceanbench.__version__

# ### Open challenger datasets

# > Insert here the code that opens the challenger dataset as `challenger_dataset: xarray.Dataset`

import xarray

challenger_dataset: xarray.Dataset = xarray.Dataset()

# ### Evaluation configuration

region = "global"

# ### Live evaluation setup

from oceanbench.core.evaluation_report import prepare_live_evaluation_report
from oceanbench.core.live_datasets import (
    live_class4_observation_last_day,
    live_class4_observation_zarr_template,
    live_glo12_analysis_zarr_template,
)

evaluation_report = prepare_live_evaluation_report(
    challenger_dataset,
    region=region,
    observation_zarr_template=live_class4_observation_zarr_template(),
    observation_last_available_day=live_class4_observation_last_day(),
    glo12_zarr_template=live_glo12_analysis_zarr_template(),
)

# ### Live evaluation of challenger dataset using OceanBench

# #### Root Mean Square Deviation (RMSD) of variables compared to recent Class IV observations

evaluation_report.class4_observation.rmsd

# #### Class IV observation error explorer

evaluation_report.class4_observation_error_explorer

# #### Root Mean Square Deviation (RMSD) of variables compared to GLO12 analysis

evaluation_report.glo12_variable_rmsd

# #### Root Mean Square Deviation (RMSD) of Mixed Layer Depth (MLD) compared to GLO12 analysis

evaluation_report.glo12_mixed_layer_depth_rmsd

# #### Root Mean Square Deviation (RMSD) of geostrophic currents compared to GLO12 analysis

evaluation_report.glo12_geostrophic_current_rmsd

# ### Forecast consistency maps

# > GLO12 analysis is used as an operational consistency reference, not as independent truth.

evaluation_report.forecast_comparison_explorer

# ### Dynamic diagnostic maps

# > Geostrophic currents are masked near the equator where the Coriolis parameter is too small.

evaluation_report.dynamic_diagnostic_explorer

# ### Spectral diagnostic

# > Zonal spectra use metric longitude coordinates, linear detrending, Tukey windowing,
# > and wavelength-band scale separation.

evaluation_report.zonal_psd_explorer
