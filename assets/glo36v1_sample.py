# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Open GLONET forecast sample with xarray
from datetime import datetime
import xarray

challenger_dataset: xarray.Dataset = (
    xarray.open_mfdataset(
        [
            "https://minio.dive.edito.eu/project-moi-glo36-oceanbench/public/20230104.zarr",
            "https://minio.dive.edito.eu/project-moi-glo36-oceanbench/public/20230111.zarr",
        ],
        engine="zarr",
        combine="nested",
        concat_dim="first_day_datetime",
        parallel=True,
    )
    .unify_chunks()
    .assign({"first_day_datetime": [datetime.fromisoformat("2023-01-04"), datetime.fromisoformat("2023-01-11")]})
)


challenger_dataset["zos"].attrs["standard_name"] = "sea_surface_height_above_geoid"
challenger_dataset["thetao"].attrs["standard_name"] = "sea_water_potential_temperature"
challenger_dataset["so"].attrs["standard_name"] = "sea_water_salinity"
challenger_dataset["uo"].attrs["standard_name"] = "eastward_sea_water_velocity"
challenger_dataset["vo"].attrs["standard_name"] = "northward_sea_water_velocity"
challenger_dataset["lat"].attrs["standard_name"] = "latitude"
challenger_dataset["lon"].attrs["standard_name"] = "longitude"

challenger_dataset

print(challenger_dataset)
