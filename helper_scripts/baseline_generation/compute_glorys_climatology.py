#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Compute the native 1/12 degree GLORYS day-of-year climatology (1993-2019).

Streams the GLORYS12V1 reanalysis from CMEMS one (variable, depth) slab at a
time, reduces each to a +/-15 day windowed day-of-year climatology, and writes
one zarr per slab. Re-runnable: a slab whose output zarr already exists is
skipped.

Temperature and salinity are computed at every model level: their in-situ
observations occur at arbitrary depths through 600 m, so scoring needs a full
profile. Currents are only ever scored at a fixed handful of depths -- the six
gridded-RMSD levels plus the 15 m drifter bracket -- so uo/vo are computed only
there and the rest of the depth axis is left empty.

Credentials come from the environment
(COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD).
"""

import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import xarray
import copernicusmarine

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
YEARS = range(1993, 2020)
WINDOW_DAYS = 15
DOWNLOAD_WORKERS = 8

# Output/scratch locations -- adjust for your host.
OUTPUT_ROOT = "/mnt/shared/jseillade/glorys_climatology"
SCRATCH_ROOT = "/mnt/shared/jseillade/glorys_clim_scratch"

FULL_DEPTH_VARIABLES = ["thetao", "so"]
CURRENT_VARIABLES = ["uo", "vo"]

# Depths (metres) at which currents are scored: the six gridded-RMSD levels
# (surface, 50, 100, 200, 300, 500 m) plus the 13.467 & 15.810 m levels that
# bracket the 15 m drifter target. These are actual GLORYS grid depths, so each
# lands on its own model level.
SCORED_CURRENT_DEPTHS = [
    0.494025,
    13.46714,
    15.81007,
    47.37369,
    92.32607,
    222.4752,
    318.1274,
    541.0889,
]


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def model_depths():
    dataset = copernicusmarine.open_dataset(dataset_id=DATASET_ID)
    return [float(value) for value in dataset.depth.values]


def nearest_index(depths, target):
    return min(range(len(depths)), key=lambda index: abs(depths[index] - target))


def build_slabs(depths):
    """(variable, depth_index, depth_value) per slab; index/value are None for zos."""
    slabs = []
    for variable in FULL_DEPTH_VARIABLES:
        for index, depth in enumerate(depths):
            slabs.append((variable, index, depth))
    for variable in CURRENT_VARIABLES:
        for target in SCORED_CURRENT_DEPTHS:
            index = nearest_index(depths, target)
            slabs.append((variable, index, depths[index]))
    slabs.append(("zos", None, None))
    return slabs


def slab_path(variable, depth_index):
    tag = "surface" if depth_index is None else f"z{depth_index:02d}"
    return os.path.join(OUTPUT_ROOT, variable, f"{tag}.zarr")


def download_year(variable, depth, year, directory):
    output_path = os.path.join(directory, f"{variable}_{year}.nc")
    request = dict(
        dataset_id=DATASET_ID,
        variables=[variable],
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T00:00:00",
        output_directory=directory,
        output_filename=f"{variable}_{year}.nc",
        coordinates_selection_method="nearest",
        overwrite=True,
        disable_progress_bar=True,
    )
    if depth is not None:
        request.update(minimum_depth=depth, maximum_depth=depth)
    attempts = 8
    for attempt in range(attempts):
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
            copernicusmarine.subset(**request)
            return output_path
        except Exception as error:
            if attempt == attempts - 1:
                raise
            log(
                f"  download {variable} {year} z={depth} failed "
                f"({type(error).__name__}); retry {attempt + 1}/{attempts}"
            )
            time.sleep(10 * (attempt + 1))


def windowed_day_of_year(climatology):
    """+/-WINDOW_DAYS rolling mean over day-of-year, wrapped at the year boundary."""
    length = climatology.sizes["dayofyear"]
    padded = xarray.concat(
        [
            climatology.isel(dayofyear=slice(length - WINDOW_DAYS, length)),
            climatology,
            climatology.isel(dayofyear=slice(0, WINDOW_DAYS)),
        ],
        dim="dayofyear",
    )
    smoothed = padded.rolling(dayofyear=2 * WINDOW_DAYS + 1, center=True).mean()
    return smoothed.isel(dayofyear=slice(WINDOW_DAYS, WINDOW_DAYS + length))


def process_slab(variable, depth_index, depth):
    output_path = slab_path(variable, depth_index)
    if os.path.exists(output_path):
        log(f"skip {variable}/{os.path.basename(output_path)} (already computed)")
        return
    scratch = os.path.join(SCRATCH_ROOT, f"{variable}_{depth_index}")
    os.makedirs(scratch, exist_ok=True)

    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for future in as_completed([pool.submit(download_year, variable, depth, year, scratch) for year in YEARS]):
            future.result()

    yearly = xarray.open_mfdataset(os.path.join(scratch, "*.nc"), combine="by_coords", chunks={"time": 120})
    field = yearly[variable]
    if "depth" in field.dims:
        field = field.isel(depth=0)
    climatology = field.groupby("time.dayofyear").mean("time")
    climatology = windowed_day_of_year(climatology).astype("float32")

    result = climatology.rename(variable).to_dataset()
    if depth is not None:
        result = result.assign_coords(depth=float(depth))
    chunks = {"dayofyear": 61, "latitude": 512, "longitude": 1080}
    result = result.chunk({dim: size for dim, size in chunks.items() if dim in result.dims})

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_zarr(output_path, mode="w", consolidated=True)
    yearly.close()
    shutil.rmtree(scratch, ignore_errors=True)
    log(f"done {variable}/{os.path.basename(output_path)}")


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(SCRATCH_ROOT, exist_ok=True)
    depths = model_depths()
    log(f"{len(depths)} model levels (surface={depths[0]:.3f} m, deepest={depths[-1]:.0f} m)")
    slabs = build_slabs(depths)
    log(f"{len(slabs)} slabs over {YEARS.start}-{YEARS.stop - 1}")

    failed = []
    for position, (variable, depth_index, depth) in enumerate(slabs, start=1):
        log(f"[{position}/{len(slabs)}] {variable} index={depth_index} depth={depth}")
        try:
            process_slab(variable, depth_index, depth)
        except Exception as error:
            log(f"  SLAB FAILED {variable} z{depth_index} ({type(error).__name__}: {str(error)[:160]})")
            failed.append(f"{variable} z{depth_index}")

    log(f"DONE with failures: {failed}" if failed else "ALL SLABS DONE")


if __name__ == "__main__":
    main()
