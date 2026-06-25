# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# ### Near-real-time surface forecast evaluation guide

# Near-real-time surface forecast evaluation compares one recent surface-only operational forecast with recent
# Class IV drifter observations. These diagnostics are for scientific evaluation and operational monitoring, not
# for annual benchmark ranking.

# This surface-only profile reports Lagrangian drifter trajectory diagnostics only. Surface-currents-only systems
# do not provide the 15 m Class IV currents needed for the gridded Class IV observation scores, so those cells are
# omitted.

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

# ### Forecast evaluation setup

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

# ### Forecast evaluation of challenger dataset using OceanBench

# #### Deviation of Lagrangian trajectories compared to recent Class IV drifter observations

evaluation_report.class4_drifter_trajectory_deviation

# #### Class IV drifter trajectory explorer

evaluation_report.class4_drifter_trajectory_explorer
