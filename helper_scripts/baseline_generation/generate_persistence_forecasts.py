#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Generate persistence baseline forecasts for OceanBench.

Persistence = the GLO12 nowcast (the initial condition the ML challengers use)
held constant across all 10 lead days. For each weekly start date we read the
GLO12 forecast's first timestep (lead 0 == nowcast) and broadcast it across the
lead-day axis, writing one zarr per start date in the standard challenger
schema (time, depth, latitude, longitude). The lead axis is stored as a single
chunk so the ten identical planes compress away to ~one nowcast on disk.
"""

import argparse
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


def start_dates():
    return list(pandas.date_range("2024-01-03", "2024-12-25", freq="7D").to_pydatetime())


def glo12_forecast_url(start):
    run_date = (start + timedelta(days=1)).strftime("%Y%m%d")
    return f"{GLO12_FORECASTS_URL}/glo12_rg_1d-m_fcst_R{run_date}.zarr"


_HTTP_STORAGE_OPTIONS = {"client_kwargs": {"timeout": aiohttp.ClientTimeout(total=900, sock_connect=60, sock_read=120)}}


def _open_and_load(url, variable, label, attempts=8):
    for attempt in range(attempts):
        try:
            field = xarray.open_zarr(url, group=variable, consolidated=True, storage_options=_HTTP_STORAGE_OPTIONS)[
                [variable]
            ].isel(time=0, drop=True)
            return field.load()
        except Exception as error:
            if attempt == attempts - 1:
                raise
            print(f"  read {label} failed ({type(error).__name__}); retry {attempt + 1}/{attempts}", flush=True)
            time.sleep(5 * (attempt + 1))


def read_nowcast(start):
    url = glo12_forecast_url(start)
    loaded = [_open_and_load(url, variable, f"{variable} {start:%Y-%m-%d}") for variable in VARIABLES]
    return xarray.merge(loaded)


def build_persistence(start):
    nowcast = read_nowcast(start)
    valid_times = pandas.to_datetime([start + timedelta(days=lead) for lead in range(LEAD_DAYS)])
    persistence = nowcast.expand_dims({"time": LEAD_DAYS}).assign_coords(time=("time", valid_times))
    chunk_sizes = {"time": LEAD_DAYS, "latitude": 640, "longitude": 1280}
    if "depth" in persistence.dims:
        chunk_sizes["depth"] = 1
    return persistence.chunk(chunk_sizes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-first", action="store_true", help="generate just the first start date (test)")
    arguments = parser.parse_args()

    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    dates = start_dates()[:1] if arguments.only_first else start_dates()
    for index, start in enumerate(dates):
        output_path = os.path.join(OUTPUT_ROOT, f"{start.strftime('%Y%m%d')}.zarr")
        if os.path.exists(output_path):
            print(f"skip {start:%Y-%m-%d} (exists)", flush=True)
            continue
        print(f"[{index + 1}/{len(dates)}] {start:%Y-%m-%d} -> {output_path}", flush=True)
        build_persistence(start).to_zarr(output_path, mode="w", consolidated=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
