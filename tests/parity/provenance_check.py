#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2
"""Empirical #298 provenance check for the parity golden.

Question: were the published v0.2.1 reports (tests/parity/golden_scores.parquet)
produced with area-weighted (cos-lat) gridded RMSD (#298, current main tip) or
with unweighted RMSD (pre-#298)?

Cheapest decisive data path: surface sea-surface-height RMSD of the
``glonet_1_degree`` challenger (public CloudFerro HTTP) against the 1-degree
GLORYS reference (public EDITO MinIO), over the full 52 weekly starts of 2024,
computed BOTH ways and compared to the golden values for
``rmsd_variables_glorys / sea_surface_height_above_geoid / surface``.

The per-lead reduction mirrors ``oceanbench.core.rmsd._rmsd`` exactly:
``sqrt( area_weighted_mean_over_latlon( (challenger-reference)**2 ) )`` averaged
over the 52 starts, with the weighting toggled off for the unweighted variant.

Run: ``python tests/parity/provenance_check.py [--starts N]``. Requires network
access to CloudFerro and EDITO MinIO. This is a confirmatory check: Julien
confirmed (authoritative) the published 0.2.1 golden is pre-#298 (unweighted).
"""
import argparse
import os
import sys
from datetime import datetime, timedelta

import numpy
import pandas
import xarray

HERE = os.path.dirname(os.path.abspath(__file__))

_GLONET_URL = "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/ml-forecast-outputs/glonet"
_GLORYS_1DEGREE_URL = "https://minio.dive.edito.eu/project-oceanbench/public/glorys_1degree_2024_V2"
_SSH = "sea_surface_height_above_geoid"
_LEAD_DAYS = 10


def _weekly_starts(count: int) -> list[datetime]:
    first, last, step = datetime(2024, 1, 3), datetime(2024, 12, 25), 7
    dates = []
    current = first
    while current <= last:
        dates.append(current)
        current = current + timedelta(days=step)
    return dates[:count]


def _one_degree_grid(latitudes: numpy.ndarray, longitudes: numpy.ndarray):
    latitude_start = numpy.ceil(latitudes.min() - 0.5) + 0.5
    latitude_end = numpy.floor(latitudes.max() + 0.5) - 0.5
    longitude_start = numpy.ceil(longitudes.min() - 0.5) + 0.5
    longitude_end = numpy.floor(longitudes.max() + 0.5) - 0.5
    return (
        numpy.arange(latitude_start, latitude_end + 1, 1.0),
        numpy.arange(longitude_start, longitude_end + 1, 1.0),
    )


def _challenger_surface_height_one_degree(start: datetime) -> xarray.DataArray:
    dataset = xarray.open_zarr(f"{_GLONET_URL}/{start.strftime('%Y%m%d')}.zarr")
    surface_height = dataset["zos"].isel(time=slice(0, _LEAD_DAYS)).load()
    new_latitude, new_longitude = _one_degree_grid(surface_height.latitude.values, surface_height.longitude.values)
    interpolated = surface_height.interp(latitude=new_latitude, longitude=new_longitude)
    return interpolated.rename({"time": "lead_day_index"}).assign_coords(
        lead_day_index=range(surface_height.sizes["time"])
    )


def _reference_surface_height(start: datetime) -> xarray.DataArray:
    dataset = xarray.open_zarr(f"{_GLORYS_1DEGREE_URL}/{start.strftime('%Y%m%d')}.zarr")
    surface_height = dataset[_SSH].isel(time=slice(0, _LEAD_DAYS)).load()
    return surface_height.rename({"time": "lead_day_index"}).assign_coords(
        lead_day_index=range(surface_height.sizes["time"])
    )


def _per_start_rmsd(challenger: xarray.DataArray, reference: xarray.DataArray):
    squared_error = (challenger - reference) ** 2
    weights = numpy.cos(numpy.deg2rad(squared_error.latitude))
    weighted = numpy.sqrt(squared_error.weighted(weights).mean(dim=["latitude", "longitude"]))
    unweighted = numpy.sqrt(squared_error.mean(dim=["latitude", "longitude"]))
    return weighted.values, unweighted.values


def _golden_surface_height_rmsd() -> numpy.ndarray:
    golden = pandas.read_parquet(os.path.join(HERE, "golden_scores.parquet"))
    rows = golden[
        (golden.challenger == "glonet_1_degree")
        & (golden.region == "global")
        & (golden.metric_key == "rmsd_variables_glorys")
        & (golden.variable_standard_name == _SSH)
        & (golden.depth_label == "Surface")
    ].sort_values("lead_day")
    return rows["value"].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--starts", type=int, default=52, help="number of weekly starts to include")
    arguments = parser.parse_args()

    starts = _weekly_starts(arguments.starts)
    weighted_per_start = []
    unweighted_per_start = []
    for index, start in enumerate(starts, start=1):
        challenger = _challenger_surface_height_one_degree(start)
        reference = _reference_surface_height(start)
        weighted, unweighted = _per_start_rmsd(challenger, reference)
        weighted_per_start.append(weighted)
        unweighted_per_start.append(unweighted)
        print(f"  [{index:2d}/{len(starts)}] {start:%Y-%m-%d} done", flush=True)

    weighted_mean = numpy.mean(numpy.vstack(weighted_per_start), axis=0)
    unweighted_mean = numpy.mean(numpy.vstack(unweighted_per_start), axis=0)
    golden = _golden_surface_height_rmsd()

    print("\nlead  golden      weighted    unweighted   |w-golden|   |u-golden|")
    for lead in range(len(golden)):
        weighted_absolute = abs(weighted_mean[lead] - golden[lead])
        unweighted_absolute = abs(unweighted_mean[lead] - golden[lead])
        print(
            f"{lead + 1:4d}  {golden[lead]:.6f}    {weighted_mean[lead]:.6f}    "
            f"{unweighted_mean[lead]:.6f}    {weighted_absolute:.2e}     {unweighted_absolute:.2e}"
        )

    weighted_max = float(numpy.max(numpy.abs(weighted_mean - golden)))
    unweighted_max = float(numpy.max(numpy.abs(unweighted_mean - golden)))
    print(f"\nmax |weighted   - golden| = {weighted_max:.2e}")
    print(f"max |unweighted - golden| = {unweighted_max:.2e}")
    verdict = "UNWEIGHTED (pre-#298)" if unweighted_max < weighted_max else "AREA-WEIGHTED (#298)"
    print(f"golden matches: {verdict}")
    if len(starts) < 52:
        print("NOTE: partial run (< 52 starts) — not directly comparable to the 52-start golden mean.")


if __name__ == "__main__":
    sys.exit(main())
