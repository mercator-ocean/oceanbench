#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Generate persistence baseline forecasts for OceanBench.

Persistence = the GLO12 nowcast (the initial condition the ML challengers start
from) held constant across all 10 lead days. For each weekly start date we read
the GLO12 forecast's first timestep (lead 0 == nowcast) and broadcast it across
the lead-day axis, writing one zarr per start date in the standard challenger
schema (time, depth, latitude, longitude).
"""

import os
import time
from datetime import timedelta

import aiohttp
import pandas
import xarray

GLO12_FORECASTS_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/dev/additionnal-data/GLO12"
VARIABLES = ["so", "thetao", "uo", "vo", "zos"]
LEAD_DAYS = 10
OUTPUT_ROOT = "/mnt/shared/jseillade/persistence_forecasts"

_STORAGE_OPTIONS = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=900, sock_connect=60, sock_read=120)}}


def start_dates():
    return list(pandas.date_range("2024-01-03", "2024-12-25", freq="7D").to_pydatetime())


def glo12_forecast_url(start):
    run_date = (start + timedelta(days=1)).strftime("%Y%m%d")
    return f"{GLO12_FORECASTS_URL}/glo12_rg_1d-m_fcst_R{run_date}.zarr"


def read_nowcast_variable(url, variable, attempts=8):
    for attempt in range(attempts):
        try:
            dataset = xarray.open_zarr(url, group=variable, consolidated=True, storage_options=_STORAGE_OPTIONS)
            return dataset[[variable]].isel(time=0, drop=True).load()
        except Exception as error:
            if attempt == attempts - 1:
                raise
            print(f"  read {variable} failed ({type(error).__name__}); retry {attempt + 1}/{attempts}", flush=True)
            time.sleep(5 * (attempt + 1))


def build_persistence(start):
    url = glo12_forecast_url(start)
    nowcast = xarray.merge([read_nowcast_variable(url, variable) for variable in VARIABLES])
    valid_times = pandas.to_datetime([start + timedelta(days=lead) for lead in range(LEAD_DAYS)])
    persistence = nowcast.expand_dims({"time": LEAD_DAYS}).assign_coords(time=("time", valid_times))
    chunks = {"time": LEAD_DAYS, "depth": 1, "latitude": 640, "longitude": 1280}
    return persistence.chunk({dim: size for dim, size in chunks.items() if dim in persistence.dims})


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    dates = start_dates()
    for position, start in enumerate(dates, start=1):
        output_path = os.path.join(OUTPUT_ROOT, f"{start.strftime('%Y%m%d')}.zarr")
        if os.path.exists(output_path):
            print(f"skip {start:%Y-%m-%d} (exists)", flush=True)
            continue
        print(f"[{position}/{len(dates)}] {start:%Y-%m-%d} -> {output_path}", flush=True)
        build_persistence(start).to_zarr(output_path, mode="w", consolidated=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
