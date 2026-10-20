# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import pandas
from xarray import Dataset, merge
import copernicusmarine
from oceanbench.core.climate_forecast_standard_names import StandardVariable


def ibi36_reference(start_date: str, end_date: str) -> Dataset:
    first_day = pandas.Timestamp(start_date).to_pydatetime()
    last_day = pandas.Timestamp(end_date).to_pydatetime()

    start_datetime = first_day.strftime("%Y-%m-%dT00:00:00")
    end_datetime = last_day.strftime("%Y-%m-%dT00:00:00")

    dataset_temperature = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_ibi_phy-temp_my_0.027deg_P1D-m",
        variables=[StandardVariable.SEA_WATER_POTENTIAL_TEMPERATURE.value],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    dataset_salinity = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_ibi_phy-sal_my_0.027deg_P1D-m",
        variables=[StandardVariable.SEA_WATER_SALINITY.value],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    dataset_current = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_ibi_phy-cur_my_0.027deg_P1D-m",
        variables=[
            StandardVariable.EASTWARD_SEA_WATER_VELOCITY.value,
            StandardVariable.NORTHWARD_SEA_WATER_VELOCITY.value,
        ],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    dataset_sea_surface_height = copernicusmarine.open_dataset(
        dataset_id="cmems_mod_ibi_phy-ssh_my_0.027deg_P1D-m",
        variables=[StandardVariable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.value],
        start_datetime=start_datetime,
        end_datetime=end_datetime,
    )

    dataset = merge([dataset_temperature, dataset_salinity, dataset_current, dataset_sea_surface_height])

    return dataset
