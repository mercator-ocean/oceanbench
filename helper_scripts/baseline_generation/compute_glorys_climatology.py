#!/usr/bin/env python
# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Compute a native 1/12 degree GLORYS day-of-year climatology (1993-2019).

Streams the GLORYS12V1 reanalysis from CMEMS one (variable, depth) slab at a
time so disk stays bounded, reduces each slab to a +/-15 day windowed
day-of-year climatology, and writes one zarr per slab. Resumable: a slab whose
output zarr already exists is skipped.

Credentials are read from the environment
(COPERNICUSMARINE_SERVICE_USERNAME / COPERNICUSMARINE_SERVICE_PASSWORD).
"""

import argparse
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import xarray
import copernicusmarine

DATASET_ID = "cmems_mod_glo_phy_my_0.083deg_P1D-m"
WINDOW_DAYS = 15
OUTPUT_ROOT = "/mnt/shared/jseillade/glorys_climatology"
SCRATCH_ROOT = "/mnt/shared/jseillade/glorys_clim_scratch"

# The six depth levels scored by OceanBench (metres). Used for uo/vo.
SCORED_DEPTHS = [
    0.494025,
    47.37369,
    92.32607,
    222.4752,
    318.1274,
    541.0889,
    13.46714,
    15.81007,
]  # + 15m obs bracket (zs6,zs7)
FULL_DEPTH_VARIABLES = ["thetao", "so"]
SCORED_DEPTH_VARIABLES = ["uo", "vo"]


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def discover_depths():
    lazy_dataset = copernicusmarine.open_dataset(dataset_id=DATASET_ID)
    return [float(value) for value in lazy_dataset.depth.values]


def nearest(values, target):
    return min(values, key=lambda value: abs(value - target))


def build_slabs(full_depths, validate):
    if validate:
        return [("thetao", full_depths[0], "00")]
    slabs = []
    for variable in FULL_DEPTH_VARIABLES:
        for depth_index, depth in enumerate(full_depths):
            slabs.append((variable, depth, f"{depth_index:02d}"))
    for variable in SCORED_DEPTH_VARIABLES:
        for level_index, target in enumerate(SCORED_DEPTHS):
            slabs.append((variable, nearest(full_depths, target), f"s{level_index}"))
    slabs.append(("zos", None, "surface"))
    return slabs


def slab_output_path(variable, slab_tag):
    return os.path.join(OUTPUT_ROOT, variable, f"z{slab_tag}.zarr")


def download_year(variable, depth, year, output_directory, validate):
    output_filename = f"{variable}_{year}.nc"
    request = dict(
        dataset_id=DATASET_ID,
        variables=[variable],
        start_datetime=f"{year}-01-01T00:00:00",
        end_datetime=f"{year}-12-31T00:00:00",
        output_directory=output_directory,
        output_filename=output_filename,
        coordinates_selection_method="nearest",
        overwrite=True,
        disable_progress_bar=True,
    )
    if depth is not None:
        request.update(minimum_depth=depth, maximum_depth=depth)
    if validate:
        request.update(
            minimum_longitude=-30.0,
            maximum_longitude=-10.0,
            minimum_latitude=30.0,
            maximum_latitude=50.0,
        )
    output_path = os.path.join(output_directory, output_filename)
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


def process_slab(variable, depth, slab_tag, years, download_workers, validate):
    output_path = slab_output_path(variable, slab_tag)
    if os.path.exists(output_path):
        log(f"skip {variable} z{slab_tag} (already computed)")
        return
    scratch_directory = os.path.join(SCRATCH_ROOT, f"{variable}_z{slab_tag}")
    os.makedirs(scratch_directory, exist_ok=True)

    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        futures = {
            pool.submit(download_year, variable, depth, year, scratch_directory, validate): year for year in years
        }
        for future in as_completed(futures):
            future.result()

    yearly_dataset = xarray.open_mfdataset(
        os.path.join(scratch_directory, "*.nc"),
        combine="by_coords",
        chunks={"time": 120},
    )
    field = yearly_dataset[variable]
    if "depth" in field.dims:
        field = field.isel(depth=0)
    climatology = field.groupby("time.dayofyear").mean("time")
    climatology = windowed_day_of_year(climatology).astype("float32")

    result = climatology.rename(variable).to_dataset()
    if depth is not None:
        result = result.assign_coords(depth=float(depth))

    chunk_sizes = {"dayofyear": 61}
    if "latitude" in result.dims:
        chunk_sizes["latitude"] = 512
    if "longitude" in result.dims:
        chunk_sizes["longitude"] = 1080
    result = result.chunk(chunk_sizes)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_zarr(output_path, mode="w", consolidated=True)
    yearly_dataset.close()
    shutil.rmtree(scratch_directory, ignore_errors=True)
    log(f"done {variable} z{slab_tag} -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true", help="fast 2-year, 1-slab, small-region run")
    parser.add_argument("--download-workers", type=int, default=8)
    arguments = parser.parse_args()

    years = [2018, 2019] if arguments.validate else list(range(1993, 2020))
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    os.makedirs(SCRATCH_ROOT, exist_ok=True)

    full_depths = discover_depths()
    log(f"{len(full_depths)} depth levels (surface={full_depths[0]:.3f} m, deepest={full_depths[-1]:.0f} m)")
    slabs = build_slabs(full_depths, arguments.validate)
    log(f"{len(slabs)} slabs to compute over years {years[0]}-{years[-1]}")

    failed_slabs = []
    for slab_index, (variable, depth, slab_tag) in enumerate(slabs):
        log(f"[{slab_index + 1}/{len(slabs)}] {variable} z{slab_tag} depth={depth}")
        try:
            process_slab(variable, depth, slab_tag, years, arguments.download_workers, arguments.validate)
        except Exception as error:
            log(f"  SLAB FAILED {variable} z{slab_tag} ({type(error).__name__}: {str(error)[:160]}) — continuing")
            failed_slabs.append(f"{variable} z{slab_tag}")

    if failed_slabs:
        log(f"DONE WITH {len(failed_slabs)} FAILED SLABS (re-run to retry): {failed_slabs}")
    else:
        log("ALL SLABS DONE")


if __name__ == "__main__":
    main()
