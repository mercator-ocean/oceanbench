# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.ocean_mask import OCEAN_MASK_DEPTHS

FIRST_DAYS = numpy.array(["2024-01-03", "2024-01-10"], dtype="datetime64[ns]")
LEAD_DAYS_COUNT = 3
LATITUDES = numpy.arange(0.0, 4.01, 0.25)
LONGITUDES = numpy.arange(10.0, 14.01, 0.25)
DEPTHS = numpy.array([0.5, 10.0, 50.0, 150.0, 400.0])
VARIABLES = [Variable.SEA_WATER_POTENTIAL_TEMPERATURE, Variable.SEA_WATER_SALINITY]


def challenger_dataset(seed: int = 1) -> xarray.Dataset:
    generator = numpy.random.default_rng(seed)
    shape = (len(FIRST_DAYS), LEAD_DAYS_COUNT, len(DEPTHS), len(LATITUDES), len(LONGITUDES))
    dimensions = [
        Dimension.FIRST_DAY_DATETIME.key(),
        Dimension.LEAD_DAY_INDEX.key(),
        Dimension.DEPTH.key(),
        Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(),
    ]
    temperature = 15.0 + generator.normal(0.0, 1.0, shape)
    salinity = 35.0 + generator.normal(0.0, 0.1, shape)
    # A missing column, so some observations have no challenger value.
    temperature[:, :, :, 8, 8] = numpy.nan
    salinity[:, :, :, 8, 8] = numpy.nan
    return xarray.Dataset(
        {
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (dimensions, temperature),
            Variable.SEA_WATER_SALINITY.key(): (dimensions, salinity),
        },
        coords={
            Dimension.FIRST_DAY_DATETIME.key(): FIRST_DAYS,
            Dimension.LEAD_DAY_INDEX.key(): numpy.arange(LEAD_DAYS_COUNT),
            Dimension.DEPTH.key(): DEPTHS,
            Dimension.LATITUDE.key(): LATITUDES,
            Dimension.LONGITUDE.key(): LONGITUDES,
        },
    )


def observations_dataset(observation_count: int = 600, seed: int = 2) -> xarray.Dataset:
    generator = numpy.random.default_rng(seed)
    first_days = generator.choice(FIRST_DAYS, observation_count)
    lead_days = generator.integers(0, LEAD_DAYS_COUNT, observation_count)
    latitudes = generator.uniform(0.5, 3.5, observation_count)
    longitudes = generator.uniform(10.5, 13.5, observation_count)
    # One observation exactly at the missing column, so it is scored as missing.
    latitudes[0], longitudes[0] = LATITUDES[8], LONGITUDES[8]
    observation_dimension = "observation"
    return xarray.Dataset(
        {
            Dimension.TIME.key(): (
                observation_dimension,
                first_days + lead_days.astype("timedelta64[D]").astype("timedelta64[ns]"),
            ),
            Dimension.LATITUDE.key(): (observation_dimension, latitudes),
            Dimension.LONGITUDE.key(): (observation_dimension, longitudes),
            Dimension.FIRST_DAY_DATETIME.key(): (observation_dimension, first_days),
            Dimension.DEPTH.key(): (observation_dimension, generator.uniform(0.0, 550.0, observation_count)),
            Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(): (
                observation_dimension,
                15.0 + generator.normal(0.0, 1.0, observation_count),
            ),
            Variable.SEA_WATER_SALINITY.key(): (
                observation_dimension,
                35.0 + generator.normal(0.0, 0.1, observation_count),
            ),
        }
    )


def ocean_mask() -> xarray.DataArray:
    latitudes = numpy.arange(0.0, 4.0 + 1e-9, 1.0 / 12.0)
    longitudes = numpy.arange(10.0, 14.0 + 1e-9, 1.0 / 12.0)
    is_wet = numpy.full((len(OCEAN_MASK_DEPTHS), len(latitudes), len(longitudes)), True)
    # A dry patch, so the shared population filter drops some observations.
    is_wet[:, 30:36, 30:36] = False
    return xarray.DataArray(
        is_wet,
        dims=[Dimension.DEPTH.key(), Dimension.LATITUDE.key(), Dimension.LONGITUDE.key()],
        coords={
            Dimension.DEPTH.key(): OCEAN_MASK_DEPTHS,
            Dimension.LATITUDE.key(): latitudes,
            Dimension.LONGITUDE.key(): longitudes,
        },
    )
