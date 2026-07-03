#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Assemble the GLORYS day-of-year climatology master into OceanBench weekly
challenger forecasts.

The master is a smoothed (+/-15 day) day-of-year climatology (1993-2019),
stored per (variable, depth) as zarrs with dims (dayofyear 1..366, latitude,
longitude). For each weekly start date and each of the 10 lead days, the
forecast field is the climatology for that lead's *valid* calendar date
(start + lead). Output matches the standard challenger schema:
(time, depth, latitude, longitude) with variables so/thetao/uo/vo/zos.

thetao/so are full-depth (50 levels). uo/vo exist only at the 6 scored levels
and are reindexed onto the 50-level grid (NaN elsewhere) so every variable
shares one depth axis -- no metric uses deep currents, and the scored levels
are present.
"""

import argparse
import os
from datetime import timedelta

import pandas
import xarray

MASTER = "/mnt/shared/jseillade/glorys_climatology"
OUTPUT = "/mnt/shared/jseillade/climatology_forecasts"
LEAD_DAYS = 10
SCORED_LEVEL_TAGS = [f"zs{level}" for level in range(8)]  # +zs6(13.47m),zs7(15.81m) for the 15m current obs metric


def start_dates():
    return list(pandas.date_range("2024-01-03", "2024-12-25", freq="7D"))


def _open_full_depth(variable, n_depths=50):
    slabs = [xarray.open_zarr(f"{MASTER}/{variable}/z{index:02d}.zarr")[variable] for index in range(n_depths)]
    return xarray.concat(slabs, dim="depth")


def _open_scored_depth(variable):
    slabs = [xarray.open_zarr(f"{MASTER}/{variable}/{tag}.zarr")[variable] for tag in SCORED_LEVEL_TAGS]
    return xarray.concat(slabs, dim="depth").sortby("depth")  # appended 15m levels are out of order


def load_master():
    thetao = _open_full_depth("thetao")
    salinity = _open_full_depth("so")
    eastward = _open_scored_depth("uo")
    northward = _open_scored_depth("vo")
    sea_surface_height = xarray.open_zarr(f"{MASTER}/zos/zsurface.zarr")["zos"]
    full_depth = thetao["depth"]
    salinity = salinity.reindex(depth=full_depth, method="nearest", tolerance=1e-2)
    eastward = eastward.reindex(depth=full_depth, method="nearest", tolerance=1e-2)
    northward = northward.reindex(depth=full_depth, method="nearest", tolerance=1e-2)
    return {"so": salinity, "thetao": thetao, "uo": eastward, "vo": northward, "zos": sea_surface_height}


def assemble_week(master, start):
    valid_times = pandas.to_datetime([start + timedelta(days=lead) for lead in range(LEAD_DAYS)])
    day_of_years = [int(valid_time.dayofyear) for valid_time in valid_times]
    fields = {}
    for name, climatology in master.items():
        selected = climatology.sel(dayofyear=day_of_years)
        fields[name] = selected.rename({"dayofyear": "time"}).assign_coords(time=("time", valid_times))
    week = xarray.Dataset(fields)
    dimension_order = [dim for dim in ("time", "depth", "latitude", "longitude") if dim in week.dims]
    week = week.transpose(*dimension_order)
    chunk_sizes = {"time": 1, "latitude": 640, "longitude": 1280}
    if "depth" in week.dims:
        chunk_sizes["depth"] = 1
    week = week.chunk(chunk_sizes)
    for variable in week.variables.values():
        variable.encoding.clear()
    return week


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-first", action="store_true", help="assemble just the first start date (test)")
    arguments = parser.parse_args()

    os.makedirs(OUTPUT, exist_ok=True)
    master = load_master()
    dates = start_dates()[:1] if arguments.only_first else start_dates()
    for index, start in enumerate(dates):
        output_path = f"{OUTPUT}/{start.strftime('%Y%m%d')}.zarr"
        if os.path.exists(output_path):
            print(f"skip {start:%Y-%m-%d} (exists)", flush=True)
            continue
        print(f"[{index + 1}/{len(dates)}] {start:%Y-%m-%d} -> {output_path}", flush=True)
        assemble_week(master, start).to_zarr(output_path, mode="w", consolidated=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
