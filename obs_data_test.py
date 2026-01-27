# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import importlib
import xarray

importlib.reload(xarray)

import xarray as xr

ZARR_URL = "https://minio.dive.edito.eu/project-ml-compression/public/observations_valid_only.zarr"
dataset = xr.open_zarr(ZARR_URL, consolidated=True)
print(dataset)
