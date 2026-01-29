# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import importlib
import xarray

importlib.reload(xarray)

ZARR_URL = "https://minio.dive.edito.eu/project-ml-compression/public/observations_by_day/20240104.zarr"

# dataset = xarray.open_zarr(ZARR_URL, consolidated=True)
# print(dataset)

try:
    ds = xarray.open_dataset(
        "https://minio.dive.edito.eu/project-ml-compression/public/observations_by_day/20240113.zarr", engine="zarr"
    )
    print(f"✅ 20240113.zarr exists with {len(ds.obs)} observations")
except Exception:
    print("❌ 20240113.zarr not found or empty")
