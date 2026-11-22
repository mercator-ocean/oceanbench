# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Fill the local stage directory with the data an evaluation reads, without computing any score.

Every asset is opened through the same function the metrics call, so the stage builders write exactly
the entries a later ``oceanbench evaluate --stage-dir`` run finds.
"""

from oceanbench.core.evaluate import read_challenger_python_code, open_challenger_dataset, resolve_evaluation_region
from oceanbench.core.ocean_mask import ocean_mask
from oceanbench.core.references.glo12 import glo12_analysis_dataset
from oceanbench.core.references.glorys import glorys_reanalysis_dataset
from oceanbench.core.references.observations import (
    ObservationDataUnavailableError,
    has_increasing_start_dates,
    load_mean_dynamic_topography,
    observations,
    observations_by_start,
)
from oceanbench.core.regions import RegionLike, subset_dataset_to_region
from oceanbench.core.resolution import get_dataset_resolution
from oceanbench.core.runtime_configuration import RuntimeConfiguration, set_runtime_configuration


def stage_challenger_assets(
    challenger_python_code_uri_or_local_path: str,
    runtime_configuration: RuntimeConfiguration,
    region: RegionLike = None,
) -> None:
    set_runtime_configuration(runtime_configuration)
    challenger_dataset = subset_dataset_to_region(
        open_challenger_dataset(read_challenger_python_code(challenger_python_code_uri_or_local_path)),
        resolve_evaluation_region(region),
    )

    glorys_reanalysis_dataset(challenger_dataset)
    glo12_analysis_dataset(challenger_dataset)
    try:
        _stage_observations(challenger_dataset)
    except ObservationDataUnavailableError:
        pass
    load_mean_dynamic_topography(get_dataset_resolution(challenger_dataset))
    ocean_mask()


def _stage_observations(challenger_dataset) -> None:
    if has_increasing_start_dates(challenger_dataset):
        for _ in observations_by_start(challenger_dataset):
            pass
    else:
        observations(challenger_dataset)
