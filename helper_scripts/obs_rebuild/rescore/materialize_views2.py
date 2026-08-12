# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Second-pass views: same as materialize_views.py plus two corrections.

1. Longitude is normalised to [-180, 180). The v2 store writes SLA longitude on
   0..360 while every in-situ stream is on -180..180, so half the SLA population
   falls outside the model grid and scores NaN. Legacy wrote SLA on -180..180.
2. New rung L1q: the legacy-equivalent temperature and salinity. Legacy applied
   measurement QC flag 1, so raw temp_raw/psal_raw (which hold fill values up to
   234 psu) are not a legacy equivalent. L1q masks them with temp_qc/psal_qc == 1
   and leaves position, depth, time QC, day alignment and dedup off.
"""
import os, shutil, time
import numpy, xarray
from concurrent.futures import ProcessPoolExecutor

SRC = "/scratch/jseillade/obs-rebuild/store-v2"
VIEWS = "/scratch/jseillade/obs-rebuild/views2"
RUNGS = ("L1", "L1q", "L2", "L2b", "L3")

SSH = "sea_surface_height_above_geoid"
TEMP = "sea_water_potential_temperature"
PSAL = "sea_water_salinity"
UO = "eastward_sea_water_velocity"
VO = "northward_sea_water_velocity"
NEEDED = (
    "depth", "latitude", "longitude", "time", SSH, TEMP, PSAL, UO, VO,
    "uo_raw", "vo_raw", "temp_raw", "psal_raw", "temp_qc", "psal_qc", "drogued",
)


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

    drogued = data["drogued"]
    legacy_uo_kept = numpy.isfinite(data[UO])
    raw_uo, raw_vo = data["uo_raw"], data["vo_raw"]
    qc_temp = numpy.where(data["temp_qc"] == 1, data["temp_raw"], numpy.nan)
    qc_psal = numpy.where(data["psal_qc"] == 1, data["psal_raw"], numpy.nan)

    variants = {
        "L1": (raw_uo, raw_vo, data["temp_raw"], data["psal_raw"]),
        "L1q": (raw_uo, raw_vo, qc_temp, qc_psal),
        "L2": (
            numpy.where(drogued == 1, raw_uo, numpy.nan),
            numpy.where(drogued == 1, raw_vo, numpy.nan),
            qc_temp, qc_psal,
        ),
        "L2b": (
            numpy.where(legacy_uo_kept, raw_uo, numpy.nan),
            numpy.where(legacy_uo_kept, raw_vo, numpy.nan),
            data[TEMP], data[PSAL],
        ),
        "L3": (data[UO], data[VO], data[TEMP], data[PSAL]),
    }
    counts = {"wrapped": int(wrapped.sum())}
    for rung, (uo, vo, temp, psal) in variants.items():
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
                TEMP: ("obs", temp),
                PSAL: ("obs", psal),
                UO: ("obs", uo),
                VO: ("obs", vo),
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
    with ProcessPoolExecutor(max_workers=int(os.environ.get("VIEW_WORKERS", "12"))) as ex:
        for day, status, counts in ex.map(build_day, days):
            if status not in ("ok", "skip"):
                bad += 1
                print(f"{day} {status}", flush=True)
    print(f"DONE days={len(days)} failures={bad} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
