# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Materialize legacy-schema (9 column) daily zarr views from the observations2024-v2 store.

Views:
  L1  legacy-equivalent   currents = uo_raw/vo_raw for every row (drogued or not),
                          temperature = temp_raw, salinity = psal_raw,
                          sla = the legacy column as written (see note)
  L2  L1 + drogue         currents = uo_raw/vo_raw restricted to drogued == 1
  L2b L3 row set, raw     currents = uo_raw/vo_raw restricted to the rows the full
                          policy keeps, so L3 minus L2b is the pure inertial-filter effect
  L3  full default policy currents/temp/psal/sla exactly as the legacy columns are written

Note on sla: the store does not retain the filtered SLA value for rows that fail the
policy, so the L1 sla column is necessarily the same as L3. Only sla_unfiltered exists
for failing rows and that is a different DUACS product variable, not a policy relaxation.
"""
import os, sys, shutil, time
import numpy, xarray
from concurrent.futures import ProcessPoolExecutor

SRC = "/scratch/jseillade/obs-rebuild/store-v2"
VIEWS = "/scratch/jseillade/obs-rebuild/views"
RUNGS = ("L1", "L2", "L2b", "L3")

SSH = "sea_surface_height_above_geoid"
TEMP = "sea_water_potential_temperature"
PSAL = "sea_water_salinity"
UO = "eastward_sea_water_velocity"
VO = "northward_sea_water_velocity"
COORDS = ("depth", "latitude", "longitude", "time")
NEEDED = COORDS + (SSH, TEMP, PSAL, UO, VO, "uo_raw", "vo_raw", "temp_raw", "psal_raw", "drogued")


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

    drogued = data["drogued"]
    legacy_uo_kept = numpy.isfinite(data[UO])
    variants = {
        "L1": (data["uo_raw"], data["vo_raw"], data["temp_raw"], data["psal_raw"]),
        "L2": (
            numpy.where(drogued == 1, data["uo_raw"], numpy.nan),
            numpy.where(drogued == 1, data["vo_raw"], numpy.nan),
            data["temp_raw"], data["psal_raw"],
        ),
        "L2b": (
            numpy.where(legacy_uo_kept, data["uo_raw"], numpy.nan),
            numpy.where(legacy_uo_kept, data["vo_raw"], numpy.nan),
            data[TEMP], data[PSAL],
        ),
        "L3": (data[UO], data[VO], data[TEMP], data[PSAL]),
    }
    counts = {}
    for rung, (uo, vo, temp, psal) in variants.items():
        out = targets[rung]
        if os.path.exists(os.path.join(out, ".zmetadata")):
            continue
        view = xarray.Dataset(
            {
                "depth": ("obs", data["depth"]),
                "latitude": ("obs", data["latitude"]),
                "longitude": ("obs", data["longitude"]),
                "time": ("obs", data["time"]),
                SSH: ("obs", data[SSH]),
                TEMP: ("obs", temp),
                PSAL: ("obs", psal),
                UO: ("obs", uo),
                VO: ("obs", vo),
            }
        )
        counts[rung] = int(numpy.isfinite(uo).sum())
        tmp = out + ".tmp"
        if os.path.exists(tmp):
            shutil.rmtree(tmp)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        view.to_zarr(tmp, mode="w", consolidated=True)
        os.replace(tmp, out)
    return (day, "ok", counts)


def main():
    days = sorted(
        name[:-5] for name in os.listdir(SRC)
        if name.endswith(".zarr") and not name.endswith(".tmp")
    )
    print(f"{len(days)} source days", flush=True)
    t0 = time.time()
    bad = 0
    with ProcessPoolExecutor(max_workers=int(os.environ.get("VIEW_WORKERS", "12"))) as ex:
        for day, status, counts in ex.map(build_day, days):
            if status not in ("ok", "skip"):
                bad += 1
            print(f"{day} {status} {counts} t={time.time()-t0:.0f}s", flush=True)
    print(f"DONE days={len(days)} failures={bad} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
