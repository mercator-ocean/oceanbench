# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Rung STRAT: the LWS basis with CURRENT_TEST carried per row, plus the 011 stratum.

Currents are FILTR minus wind slippage, exactly the LWS recipe of materialize_ws.py,
on every row the default policy keeps. In addition the confirmed undrogued rows
(CURRENT_TEST 011) whose only policy failure was the drogue rule are restored, so
that the contaminated stratum is scored on the same basis as the kept ones. Their
FILTR values were blanked by the builder, so they are read back from the source
GL_TS_DC_*_FILTR.nc archive and joined on obs_id. The join is validated on the kept
rows, where the archive value must equal the stored value bit for bit.

The published store is never touched: this is a read-time view.
"""
import os
import shutil
import sys
import time

import numpy
import pandas
import xarray
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, "/scratch/jseillade/obs-rebuild")
import build_observations as builder

SRC = "/scratch/jseillade/obs-rebuild/store-v2"
ARCHIVE = "/scratch/jseillade/obs-rebuild/raw-archive"
OUT = "/scratch/jseillade/obs-rebuild/views2/STRAT"

SSH = "sea_surface_height_above_geoid"
TEMP = "sea_water_potential_temperature"
PSAL = "sea_water_salinity"
UO = "eastward_sea_water_velocity"
VO = "northward_sea_water_velocity"
NEEDED = (
    "depth", "latitude", "longitude", "time", SSH, TEMP, PSAL, UO, VO,
    "uo_ws", "vo_ws", "obs_type", "current_test", "qc_keep", "qc_reason",
    "obs_id", "platform_code", "time_ns",
)
UNDROGUED_CODE = 11


def archive_filtr(day):
    """obs_id -> (EWCT_FILTR, NSCT_FILTR) for the source file of that day."""
    path = os.path.join(ARCHIVE, day, f"GL_TS_DC_{day}_FILTR.nc")
    dataset = xarray.open_dataset(path)
    try:
        n = dataset.sizes["TIME"]
        frame = pandas.DataFrame({
            "obs_type": numpy.full(n, builder.OBS_TYPE_DRIFTER_CURRENT),
            "platform_code": builder.char_to_str(dataset["PLATFORM_CODE"].values, n),
            "depth": numpy.asarray(dataset["DEPH"].values)[:, 0].astype(numpy.float64),
        })
        frame["time_ns"] = builder.to_datetime_ns(dataset["TIME"].values)
        ids = builder.build_obs_ids(frame).to_numpy()
        uo = numpy.asarray(dataset["EWCT_FILTR"].values)[:, 0].astype(numpy.float64)
        vo = numpy.asarray(dataset["NSCT_FILTR"].values)[:, 0].astype(numpy.float64)
    finally:
        dataset.close()
    return {key: index for index, key in enumerate(ids)}, uo, vo


def build_day(day):
    target = os.path.join(OUT, f"{day}.zarr")
    if os.path.exists(os.path.join(target, ".zmetadata")):
        return day, "skip", {}
    source = os.path.join(SRC, f"{day}.zarr")
    dataset = xarray.open_dataset(source, engine="zarr", decode_cf=False, consolidated=True)
    data = {name: dataset[name].values for name in NEEDED}
    dataset.close()

    size = data["obs_type"].shape[0]
    currents = data["obs_type"] == 3
    index_map, arc_uo, arc_vo = archive_filtr(day)

    positions = numpy.full(size, -1, dtype=numpy.int64)
    current_indices = numpy.flatnonzero(currents)
    for position in current_indices:
        positions[position] = index_map.get(data["obs_id"][position], -1)
    matched = positions >= 0

    filtr_uo = numpy.full(size, numpy.nan)
    filtr_vo = numpy.full(size, numpy.nan)
    filtr_uo[matched] = arc_uo[positions[matched]]
    filtr_vo[matched] = arc_vo[positions[matched]]

    kept = currents & (data["qc_keep"] == 1) & numpy.isfinite(data[UO]) & numpy.isfinite(data[VO])
    check = kept & matched
    join_error_u = float(numpy.abs(filtr_uo[check] - data[UO][check]).max()) if check.any() else 0.0
    join_error_v = float(numpy.abs(filtr_vo[check] - data[VO][check]).max()) if check.any() else 0.0

    rescued = (
        currents
        & (data["current_test"] == UNDROGUED_CODE)
        & (data["qc_reason"] == "undrogued")
        & matched
        & numpy.isfinite(filtr_uo)
        & numpy.isfinite(filtr_vo)
    )

    uo = numpy.full(size, numpy.nan)
    vo = numpy.full(size, numpy.nan)
    uo[kept] = data[UO][kept]
    vo[kept] = data[VO][kept]
    uo[rescued] = filtr_uo[rescued]
    vo[rescued] = filtr_vo[rescued]

    active = kept | rescued
    correction_u = numpy.where(numpy.isfinite(data["uo_ws"]), data["uo_ws"], 0.0)
    correction_v = numpy.where(numpy.isfinite(data["vo_ws"]), data["vo_ws"], 0.0)
    uo[active] = uo[active] - correction_u[active]
    vo[active] = vo[active] - correction_v[active]

    longitude = data["longitude"].copy()
    longitude[longitude >= 180.0] -= 360.0

    code = numpy.where(currents, data["current_test"], -1).astype(numpy.int32)
    ws_magnitude = numpy.where(
        numpy.isfinite(data["uo_ws"]) & numpy.isfinite(data["vo_ws"]),
        numpy.sqrt(data["uo_ws"] ** 2 + data["vo_ws"] ** 2),
        numpy.nan,
    )

    view = xarray.Dataset({
        "depth": ("obs", data["depth"]),
        "latitude": ("obs", data["latitude"]),
        "longitude": ("obs", longitude),
        "time": ("obs", data["time"]),
        SSH: ("obs", data[SSH]),
        TEMP: ("obs", data[TEMP]),
        PSAL: ("obs", data[PSAL]),
        UO: ("obs", uo),
        VO: ("obs", vo),
        "current_test": ("obs", code),
        "ws_magnitude": ("obs", ws_magnitude),
    })
    temporary = target + ".tmp"
    if os.path.exists(temporary):
        shutil.rmtree(temporary)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    view.to_zarr(temporary, mode="w", consolidated=True)
    os.replace(temporary, target)

    counts = {
        "currents": int(currents.sum()),
        "kept": int(kept.sum()),
        "rescued": int(rescued.sum()),
        "undrogued_total": int((currents & (data["current_test"] == UNDROGUED_CODE)).sum()),
        "unmatched_currents": int((currents & ~matched).sum()),
    }
    return day, "ok", {**counts, "join_error": max(join_error_u, join_error_v)}


def main():
    days = sorted(n[:-5] for n in os.listdir(SRC) if n.endswith(".zarr") and not n.endswith(".tmp"))
    print(f"{len(days)} source days", flush=True)
    started = time.time()
    totals = {}
    worst_join = 0.0
    failures = 0
    with ProcessPoolExecutor(max_workers=int(os.environ.get("VIEW_WORKERS", "12"))) as pool:
        for day, status, counts in pool.map(build_day, days):
            if status not in ("ok", "skip"):
                failures += 1
                print(f"{day} {status}", flush=True)
            worst_join = max(worst_join, counts.pop("join_error", 0.0))
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
    print(f"DONE days={len(days)} failures={failures} totals={totals} "
          f"worst_join_error={worst_join:.3e} in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
