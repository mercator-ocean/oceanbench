# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Fill the local stage directory with the data an evaluation reads, without computing any score.

Every asset is opened through the same function the metrics call, so the stage builders write exactly
the entries a later ``oceanbench evaluate --stage-dir`` run finds.
"""

import os
import sys

from oceanbench.core.evaluate import resolve_evaluation_region
from oceanbench.core.ocean_mask import ocean_mask
from oceanbench.core.python2jupyter import _parse_challenger_python_code
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import (
    ObservationDataUnavailableError,
    load_mean_dynamic_topography,
    observations,
)
from oceanbench.core.regions import RegionLike, subset_dataset_to_region
from oceanbench.core.resolution import get_dataset_resolution
from oceanbench.core.runtime_configuration import RuntimeConfiguration, set_runtime_configuration


def _add_current_directory_to_python_path() -> None:
    # Same rule as IPython's InteractiveShellApp.init_path, so the challenger file imports what it imports in evaluate.
    if "" in sys.path:
        return
    for index, path in enumerate(sys.path):
        if os.path.basename(path) in {"site-packages", "dist-packages"}:
            break
    else:
        index = 0
    sys.path.insert(index, "")


def stage_challenger_assets(
    challenger_python_code_uri_or_local_path: str,
    runtime_configuration: RuntimeConfiguration,
    region: RegionLike = None,
) -> None:
    set_runtime_configuration(runtime_configuration)
    _add_current_directory_to_python_path()
    challenger_namespace: dict = {"__name__": "__main__"}
    exec(_parse_challenger_python_code(challenger_python_code_uri_or_local_path), challenger_namespace)
    challenger_dataset = subset_dataset_to_region(
        challenger_namespace["challenger_dataset"],
        resolve_evaluation_region(region),
    )

    glorys_reanalysis_dataset(challenger_dataset)
    glo12_analysis_dataset(challenger_dataset)
    try:
        observations(challenger_dataset)
    except ObservationDataUnavailableError:
        pass
    load_mean_dynamic_topography(get_dataset_resolution(challenger_dataset))
    ocean_mask()
