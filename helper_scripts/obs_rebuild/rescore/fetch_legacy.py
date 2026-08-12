# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Copy the legacy public observations2024 daily zarrs to local scratch, as-is."""
import os, sys, time
import pandas, xarray
from concurrent.futures import ThreadPoolExecutor

DST = "/scratch/jseillade/obs-rebuild/store-legacy"
BASE = "https://minio.dive.edito.eu/project-oceanbench/public/observations2024"
days = pandas.date_range("2024-01-01", "2025-01-04", freq="D")

def one(day):
    ds_name = day.strftime("%Y%m%d")
    out = os.path.join(DST, ds_name + ".zarr")
    if os.path.exists(os.path.join(out, ".zmetadata")):
        return (ds_name, "skip", 0)
    tmp = out + ".tmp"
    try:
        ds = xarray.open_dataset(f"{BASE}/{ds_name}.zarr", engine="zarr", decode_cf=False, consolidated=True)
        ds = ds.load()
        if os.path.exists(tmp):
            import shutil; shutil.rmtree(tmp)
        ds.to_zarr(tmp, mode="w", consolidated=True)
        ds.close()
        os.replace(tmp, out)
        return (ds_name, "ok", ds.sizes["obs"])
    except Exception as exc:
        return (ds_name, f"FAIL {type(exc).__name__}: {exc}", 0)

t0 = time.time()
bad = 0
with ThreadPoolExecutor(max_workers=8) as ex:
    for name, status, n in ex.map(one, days):
        if status.startswith("FAIL"):
            bad += 1
        print(f"{name} {status} rows={n} t={time.time()-t0:.0f}s", flush=True)
print(f"DONE days={len(days)} failures={bad} in {time.time()-t0:.0f}s", flush=True)
