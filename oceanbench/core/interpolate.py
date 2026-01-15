# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import xarray as xr
import numpy as np


def interpolate_1deg(data: xr.Dataset) -> xr.Dataset:
    # create target lat/lon values at 1 deg
    new_lat = np.arange(np.floor(data.lat.min()), np.ceil(data.lat.max()), 1)
    new_lon = np.arange(np.floor(data.lon.min()), np.ceil(data.lon.max()), 1)
    print("interpolating")
    # interpolate
    if hasattr(data, "time") and hasattr(data, "depth"):
        data = data.chunk({"time": 1, "lat": -1, "lon": -1, "depth": 1})
    else:
        data = data.chunk({"lat": -1, "lon": -1})

    data_interp = data.interp(lat=new_lat, lon=new_lon)
    return data_interp
