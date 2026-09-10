# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Open GLO36V1 forecasts with oceanbench, for a single first day
from datetime import datetime
import xarray
import oceanbench

challenger_dataset: xarray.Dataset = oceanbench.datasets.challenger.glo36v1([datetime.fromisoformat("2023-01-04")])

challenger_dataset["zos"].attrs["standard_name"] = "sea_surface_height_above_geoid"
challenger_dataset["thetao"].attrs["standard_name"] = "sea_water_potential_temperature"
challenger_dataset["so"].attrs["standard_name"] = "sea_water_salinity"
challenger_dataset["uo"].attrs["standard_name"] = "eastward_sea_water_velocity"
challenger_dataset["vo"].attrs["standard_name"] = "northward_sea_water_velocity"
challenger_dataset["latitude"].attrs["standard_name"] = "latitude"
challenger_dataset["longitude"].attrs["standard_name"] = "longitude"

challenger_dataset
