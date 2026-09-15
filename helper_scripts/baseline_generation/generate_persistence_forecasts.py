#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Generate persistence baseline forecasts for OceanBench.

Persistence = the GLO12 nowcast bulletin issued on the start day (a state valid
the day before), held constant over the 10 lead days. This is the initial
condition the ML challengers start from. For each weekly start date we read the
nowcast bulletin's single timestep and broadcast it across the lead-day axis,
writing one zarr per start date in the standard challenger schema
(time, depth, latitude, longitude).
"""

import os
import sys
import time
from datetime import timedelta

import aiohttp
import pandas
import xarray

GLO12_NOWCASTS_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/GLO12_NOWCAST"
VARIABLES = ["so", "thetao", "uo", "vo", "zos"]
LEAD_DAYS = 10
CHUNKS = {"time": 1, "depth": 1, "latitude": 640, "longitude": 1280}
FILL_VALUE = 9.969209968386869e36

_STORAGE_OPTIONS = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=900, sock_connect=60, sock_read=120)}}


def start_dates():
    return list(pandas.date_range("2024-01-03", "2024-12-25", freq="7D").to_pydatetime())


def glo12_nowcast_url(start):
    return f"{GLO12_NOWCASTS_URL}/{start.strftime('%Y%m%d')}.zarr"


def read_nowcast(url, attempts=8):
    for attempt in range(attempts):
        try:
            dataset = xarray.open_zarr(url, consolidated=True, storage_options=_STORAGE_OPTIONS)
            nowcast = dataset[VARIABLES].isel(time=0, drop=True).load()
            for variable in nowcast.variables:
                nowcast[variable].encoding.clear()
            return nowcast
        except Exception as error:
            if attempt == attempts - 1:
                raise
            print(f"  read failed ({type(error).__name__}); retry {attempt + 1}/{attempts}", flush=True)
            time.sleep(5 * (attempt + 1))


def build_persistence(start):
    nowcast = read_nowcast(glo12_nowcast_url(start))
    valid_times = pandas.to_datetime([start + timedelta(days=lead) for lead in range(LEAD_DAYS)])
    persistence = nowcast.expand_dims({"time": LEAD_DAYS}).assign_coords(time=("time", valid_times))
    return persistence.chunk({dim: size for dim, size in CHUNKS.items() if dim in persistence.dims})


def main():
    output_root = sys.argv[1] if len(sys.argv) > 1 else "persistence_forecasts"
    os.makedirs(output_root, exist_ok=True)
    dates = start_dates()
    for position, start in enumerate(dates, start=1):
        output_path = os.path.join(output_root, f"{start.strftime('%Y%m%d')}.zarr")
        if os.path.exists(output_path):
            print(f"skip {start:%Y-%m-%d} (exists)", flush=True)
            continue
        print(f"[{position}/{len(dates)}] {start:%Y-%m-%d} -> {output_path}", flush=True)
        encoding = {variable: {"_FillValue": FILL_VALUE} for variable in VARIABLES}
        build_persistence(start).to_zarr(output_path, mode="w", consolidated=True, encoding=encoding)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
