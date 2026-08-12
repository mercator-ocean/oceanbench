# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Lean scoring view of the rewritten 2024-v2.1.0 store.

The published store carries about forty columns including wide string columns,
which the class-4 scorer neither needs nor should pay to read. This writes the
nine legacy columns straight through, with no arithmetic of any kind: the
default current columns already carry the adopted policy after the in-place
rewrite. Longitude is already on [-180, 180) since 2024-v2.0.1, so it is copied
verbatim and asserted rather than re-wrapped.

Same shape and recipe as materialize_ws.py so the ladder scripts read it without
changes.
"""
import os
import shutil
import time

import numpy
import xarray
from concurrent.futures import ProcessPoolExecutor

SRC = "/scratch/jseillade/obs-rebuild/store-v2"
OUT = "/scratch/jseillade/obs-rebuild/views2/V21"
EXPECTED_VERSION = "2024-v2.1.0"

SSH = "sea_surface_height_above_geoid"
TEMP = "sea_water_potential_temperature"
PSAL = "sea_water_salinity"
UO = "eastward_sea_water_velocity"
VO = "northward_sea_water_velocity"
COLUMNS = ("depth", "latitude", "longitude", "time", SSH, TEMP, PSAL, UO, VO)


def build_day(day):
    source = os.path.join(SRC, f"{day}.zarr")
    target = os.path.join(OUT, f"{day}.zarr")
    if os.path.exists(os.path.join(target, ".zmetadata")):
        return day, "skip", {}
    dataset = xarray.open_dataset(source, engine="zarr", decode_cf=False, consolidated=True)
    version = dataset.attrs.get("obs_basis_version")
    if version != EXPECTED_VERSION:
        dataset.close()
        return day, f"WRONG_VERSION:{version}", {}
    data = {name: dataset[name].values for name in COLUMNS}
    dataset.close()

    longitude = data["longitude"]
    finite = numpy.isfinite(longitude)
    if not bool(numpy.all((longitude[finite] >= -180.0) & (longitude[finite] < 180.0))):
        return day, "LONGITUDE_OUT_OF_RANGE", {}

    view = xarray.Dataset({name: ("obs", data[name]) for name in COLUMNS})
    temporary = target + ".tmp"
    if os.path.exists(temporary):
        shutil.rmtree(temporary)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    view.to_zarr(temporary, mode="w", consolidated=True)
    os.replace(temporary, target)

    counts = {
        "rows": int(longitude.size),
        "currents_finite": int((numpy.isfinite(data[UO]) & numpy.isfinite(data[VO])).sum()),
        "temp_finite": int(numpy.isfinite(data[TEMP]).sum()),
        "psal_finite": int(numpy.isfinite(data[PSAL]).sum()),
        "sla_finite": int(numpy.isfinite(data[SSH]).sum()),
    }
    return day, "ok", counts


def main():
    days = sorted(n[:-5] for n in os.listdir(SRC) if n.endswith(".zarr"))
    shard = int(os.environ.get("VIEW_SHARD", "0"))
    shard_count = int(os.environ.get("VIEW_SHARD_COUNT", "1"))
    days = [d for index, d in enumerate(days) if index % shard_count == shard]
    print(f"{len(days)} source days (shard {shard}/{shard_count})", flush=True)
    started = time.time()
    failures = 0
    totals = {}
    with ProcessPoolExecutor(max_workers=int(os.environ.get("VIEW_WORKERS", "12"))) as pool:
        for day, status, counts in pool.map(build_day, days):
            if status not in ("ok", "skip"):
                failures += 1
                print(f"{day} {status}", flush=True)
            for key, value in counts.items():
                totals[key] = totals.get(key, 0) + value
    print(f"DONE days={len(days)} failures={failures} totals={totals} "
          f"in {time.time() - started:.0f}s", flush=True)
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
