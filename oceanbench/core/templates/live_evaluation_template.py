# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# ### Near-real-time forecast validation guide

# Near-real-time forecast validation compares one recent operational forecast with recent Class IV
# observations. These diagnostics are for scientific validation and operational monitoring, not for annual
# benchmark ranking.

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

# ### Forecast validation setup

from oceanbench.core.evaluation_report import prepare_live_evaluation_report
from oceanbench.core.live_datasets import (
    live_class4_observation_last_day,
    live_class4_observation_zarr_template,
)

evaluation_report = prepare_live_evaluation_report(
    challenger_dataset,
    region=region,
    observation_zarr_template=live_class4_observation_zarr_template(),
    observation_last_available_day=live_class4_observation_last_day(),
)

# ### Forecast validation of challenger dataset using OceanBench

# #### Root Mean Square Deviation (RMSD) of variables compared to recent Class IV observations

evaluation_report.class4_observation.rmsd

# #### Deviation of Lagrangian trajectories compared to recent Class IV drifter observations

evaluation_report.class4_drifter_trajectory_deviation

# #### Class IV observation error explorer

evaluation_report.class4_observation_error_explorer

# #### Class IV drifter trajectory explorer

evaluation_report.class4_drifter_trajectory_explorer
