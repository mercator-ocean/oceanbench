#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Assemble the GLORYS day-of-year climatology into OceanBench weekly forecasts.

The climatology master holds one zarr per (variable, model level) with dims
(dayofyear 1..366, latitude, longitude). For each weekly start date and each of
the 10 lead days, the forecast is the climatology for that lead's *valid*
calendar date (start + lead). Output uses the standard challenger schema
(time, depth, latitude, longitude) with variables so/thetao/uo/vo/zos.

thetao and so exist at every model level. uo/vo exist only at the scored levels
(see compute_glorys_climatology.py) and are placed back onto the full depth axis
-- NaN elsewhere -- so every variable shares one axis; no metric reads currents
off the scored levels.
"""

import glob
import os
from datetime import timedelta

import pandas
import xarray

MASTER = "/mnt/shared/jseillade/glorys_climatology"
OUTPUT = "/mnt/shared/jseillade/climatology_forecasts"
LEAD_DAYS = 10


def start_dates():
    return list(pandas.date_range("2024-01-03", "2024-12-25", freq="7D"))


def open_variable(variable):
    """Concatenate a variable's per-level slabs along depth, ascending by level."""
    paths = sorted(glob.glob(f"{MASTER}/{variable}/z*.zarr"))  # z00 < z09 < z10 keeps depth order
    slabs = [xarray.open_zarr(path)[variable] for path in paths]
    return xarray.concat(slabs, dim="depth")


def load_master():
    thetao = open_variable("thetao")
    salinity = open_variable("so")
    eastward = open_variable("uo")
    northward = open_variable("vo")
    sea_surface_height = xarray.open_zarr(f"{MASTER}/zos/surface.zarr")["zos"]
    depth_axis = thetao["depth"]
    # currents live on a sparse subset of levels -> restore the full axis (NaN elsewhere)
    eastward = eastward.reindex(depth=depth_axis, method="nearest", tolerance=1e-2)
    northward = northward.reindex(depth=depth_axis, method="nearest", tolerance=1e-2)
    return {"so": salinity, "thetao": thetao, "uo": eastward, "vo": northward, "zos": sea_surface_height}


def assemble_week(master, start):
    valid_times = pandas.to_datetime([start + timedelta(days=lead) for lead in range(LEAD_DAYS)])
    day_of_years = [int(valid_time.dayofyear) for valid_time in valid_times]
    fields = {}
    for name, climatology in master.items():
        selected = climatology.sel(dayofyear=day_of_years)
        fields[name] = selected.rename({"dayofyear": "time"}).assign_coords(time=("time", valid_times))
    week = xarray.Dataset(fields)
    order = [dim for dim in ("time", "depth", "latitude", "longitude") if dim in week.dims]
    week = week.transpose(*order)
    chunks = {"time": 1, "depth": 1, "latitude": 640, "longitude": 1280}
    week = week.chunk({dim: size for dim, size in chunks.items() if dim in week.dims})
    for variable in week.variables.values():
        variable.encoding.clear()
    return week


def main():
    os.makedirs(OUTPUT, exist_ok=True)
    master = load_master()
    dates = start_dates()
    for position, start in enumerate(dates, start=1):
        output_path = f"{OUTPUT}/{start.strftime('%Y%m%d')}.zarr"
        if os.path.exists(output_path):
            print(f"skip {start:%Y-%m-%d} (exists)", flush=True)
            continue
        print(f"[{position}/{len(dates)}] {start:%Y-%m-%d} -> {output_path}", flush=True)
        assemble_week(master, start).to_zarr(output_path, mode="w", consolidated=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
