# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Rung L_WS: default policy view with wind slippage subtracted from currents.

LWS  : uo = uo_default - uo_ws where uo_ws is finite, uo_default otherwise.
LWSp : same with the opposite sign, kept only to disambiguate the sign
       convention empirically. Not a candidate policy.

Everything else (row set, temperature, salinity, sla, longitude normalisation)
is byte-for-byte the L3 recipe of materialize_views2.py. The published store is
never touched: this is a read-time view.
"""
import os, shutil, time
import numpy, xarray
from concurrent.futures import ProcessPoolExecutor

SRC = "/scratch/jseillade/obs-rebuild/store-v2"
VIEWS = "/scratch/jseillade/obs-rebuild/views2"
RUNGS = ("LWS", "LWSp")

SSH = "sea_surface_height_above_geoid"
TEMP = "sea_water_potential_temperature"
PSAL = "sea_water_salinity"
UO = "eastward_sea_water_velocity"
VO = "northward_sea_water_velocity"
NEEDED = ("depth", "latitude", "longitude", "time", SSH, TEMP, PSAL, UO, VO, "uo_ws", "vo_ws")


def build_day(day):
    src = os.path.join(SRC, f"{day}.zarr")
    if not os.path.exists(os.path.join(src, ".zmetadata")):
        return (day, "MISSING_SOURCE", {})
    targets = {r: os.path.join(VIEWS, r, f"{day}.zarr") for r in RUNGS}
    if all(os.path.exists(os.path.join(p, ".zmetadata")) for p in targets.values()):
        return (day, "skip", {})
    ds = xarray.open_dataset(src, engine="zarr", decode_cf=False, consolidated=True)
    data = {k: ds[k].values for k in NEEDED}
    ds.close()

    longitude = data["longitude"].copy()
    wrapped = longitude >= 180.0
    longitude[wrapped] -= 360.0

    uo, vo = data[UO], data[VO]
    correction_u = numpy.where(numpy.isfinite(data["uo_ws"]), data["uo_ws"], 0.0)
    correction_v = numpy.where(numpy.isfinite(data["vo_ws"]), data["vo_ws"], 0.0)
    variants = {
        "LWS": (uo - correction_u, vo - correction_v),
        "LWSp": (uo + correction_u, vo + correction_v),
    }
    kept = numpy.isfinite(uo) & numpy.isfinite(vo)
    counts = {
        "kept": int(kept.sum()),
        "kept_ws": int((kept & numpy.isfinite(data["uo_ws"]) & numpy.isfinite(data["vo_ws"])).sum()),
    }
    for rung, (new_uo, new_vo) in variants.items():
        out = targets[rung]
        if os.path.exists(os.path.join(out, ".zmetadata")):
            continue
        view = xarray.Dataset(
            {
                "depth": ("obs", data["depth"]),
                "latitude": ("obs", data["latitude"]),
                "longitude": ("obs", longitude),
                "time": ("obs", data["time"]),
                SSH: ("obs", data[SSH]),
                TEMP: ("obs", data[TEMP]),
                PSAL: ("obs", data[PSAL]),
                UO: ("obs", new_uo),
                VO: ("obs", new_vo),
            }
        )
        tmp = out + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        view.to_zarr(tmp, mode="w", consolidated=True)
        os.replace(tmp, out)
    return (day, "ok", counts)


def main():
    days = sorted(n[:-5] for n in os.listdir(SRC) if n.endswith(".zarr") and not n.endswith(".tmp"))
    print(f"{len(days)} source days", flush=True)
    t0 = time.time()
    bad = 0
    totals = {}
    with ProcessPoolExecutor(max_workers=int(os.environ.get("VIEW_WORKERS", "12"))) as ex:
        for day, status, counts in ex.map(build_day, days):
            if status not in ("ok", "skip"):
                bad += 1
                print(f"{day} {status}", flush=True)
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
    print(f"DONE days={len(days)} failures={bad} totals={totals} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
