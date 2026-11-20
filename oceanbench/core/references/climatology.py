# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import copernicusmarine
from xarray import Dataset

from oceanbench.core.climate_forecast_standard_names import StandardVariable

GLORYS_MONTHLY_CLIMATOLOGY_DATASET_IDENTIFIER = "cmems_mod_glo_phy_my_0.083deg-climatology_P1M-m"
CLIMATOLOGY_MAXIMUM_DEPTH_METERS = 700.0

_GLORYS_MONTHLY_CLIMATOLOGY_DATASET_CACHE: dict[str, Dataset] = {}


def glorys_monthly_climatology_dataset() -> Dataset:
    cached_dataset = _GLORYS_MONTHLY_CLIMATOLOGY_DATASET_CACHE.get(GLORYS_MONTHLY_CLIMATOLOGY_DATASET_IDENTIFIER)
    if cached_dataset is not None:
        return cached_dataset
    dataset = copernicusmarine.open_dataset(
        dataset_id=GLORYS_MONTHLY_CLIMATOLOGY_DATASET_IDENTIFIER,
        variables=[
            StandardVariable.SEA_WATER_POTENTIAL_TEMPERATURE.value,
            StandardVariable.SEA_WATER_SALINITY.value,
            StandardVariable.EASTWARD_SEA_WATER_VELOCITY.value,
            StandardVariable.NORTHWARD_SEA_WATER_VELOCITY.value,
            StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.value,
        ],
        maximum_depth=CLIMATOLOGY_MAXIMUM_DEPTH_METERS,
    )
    _GLORYS_MONTHLY_CLIMATOLOGY_DATASET_CACHE[GLORYS_MONTHLY_CLIMATOLOGY_DATASET_IDENTIFIER] = dataset
    return dataset
