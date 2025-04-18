# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Open XIHE forecast sample with xarray
from datetime import datetime
import xarray

challenger_dataset: xarray.Dataset = xarray.open_mfdataset(
    [
        "https://minio.dive.edito.eu/project-oceanbench/public/XIHE/20240103.zarr",
    ],
    engine="zarr",
    preprocess=lambda dataset: dataset.rename({"time": "lead_day_index"}).assign({"lead_day_index": range(10)}),
    combine="nested",
    concat_dim="first_day_datetime",
    parallel=True,
).assign(
    {
        "first_day_datetime": [
            datetime.fromisoformat("2024-01-03"),
        ]
    }
)
challenger_dataset

print(challenger_dataset)
