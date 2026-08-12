# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""CURRENT_TEST code distribution over currents rows of the v2 store, full year.
Also the spatial structure of the wind slippage correction."""
import os, collections
import numpy, xarray, pandas
from concurrent.futures import ProcessPoolExecutor

SRC = "/scratch/jseillade/obs-rebuild/store-v2"
UO = "eastward_sea_water_velocity"
VO = "northward_sea_water_velocity"

def day(d):
    ds = xarray.open_dataset(os.path.join(SRC, d + ".zarr"), engine="zarr", decode_cf=False, consolidated=True)
    a = {n: ds[n].values for n in ("obs_type", "current_test", "qc_keep", "uo_ws", "vo_ws", "latitude", "longitude", UO, VO)}
    ds.close()
    cur = a["obs_type"] == 3
    kept = cur & (a["qc_keep"] == 1) & numpy.isfinite(a[UO]) & numpy.isfinite(a[VO])
    all_counts = collections.Counter(a["current_test"][cur].tolist())
    kept_counts = collections.Counter(a["current_test"][kept].tolist())
    lat = a["latitude"]
    band = numpy.floor(numpy.clip(lat, -89.9, 89.9) / 20.0).astype(int) * 20
    sel = kept & numpy.isfinite(a["uo_ws"]) & numpy.isfinite(a["vo_ws"])
    bands = pandas.DataFrame({
        "band": band[sel], "du": -a["uo_ws"][sel], "dv": -a["vo_ws"][sel],
        "sq": a["uo_ws"][sel] ** 2 + a["vo_ws"][sel] ** 2,
    }).groupby("band").agg(n=("du", "size"), du=("du", "sum"), dv=("dv", "sum"), sq=("sq", "sum"))
    return all_counts, kept_counts, bands

days = sorted(n[:-5] for n in os.listdir(SRC) if n.endswith(".zarr"))
total_all, total_kept = collections.Counter(), collections.Counter()
band_frames = []
with ProcessPoolExecutor(max_workers=12) as ex:
    for a, k, b in ex.map(day, days):
        total_all.update(a); total_kept.update(k); band_frames.append(b)

n_all = sum(total_all.values()); n_kept = sum(total_kept.values())
print(f"days {len(days)}  currents rows in store {n_all}  rows kept by default policy {n_kept}")
print()
print("| code | rows in store | pct of store | rows kept by default | pct of kept |")
print("| --- | --- | --- | --- | --- |")
for code, c in sorted(total_all.items(), key=lambda kv: -kv[1]):
    label = "untested/fill" if code < 0 else f"{code:03d}"
    print(f"| {label} | {c} | {100.0*c/n_all:.2f}% | {total_kept.get(code,0)} | {100.0*total_kept.get(code,0)/n_kept:.2f}% |")
only313 = total_kept.get(313, 0)
print()
print(f"keep-only-313 on the default-policy row set: {only313} rows, {100.0*only313/n_kept:.2f}% of the {n_kept} kept, a drop of {n_kept-only313} rows")
print(f"keep-only-313 against every currents row in the store: {100.0*total_all.get(313,0)/n_all:.2f}%")
print()
bands = pandas.concat(band_frames).groupby("band").sum()
bands["mean_du"] = bands["du"] / bands["n"]
bands["mean_dv"] = bands["dv"] / bands["n"]
bands["rms_ws"] = numpy.sqrt(bands["sq"] / bands["n"])
print("signed obs delta by 20 deg latitude band (delta = -ws)")
print(bands[["n", "mean_du", "mean_dv", "rms_ws"]].to_string())
tot_n = bands["n"].sum()
print()
print("global mean_du %.6f mean_dv %.6f rms_ws %.6f  ratio |mean|/rms %.3f" % (
    bands["du"].sum()/tot_n, bands["dv"].sum()/tot_n, numpy.sqrt(bands["sq"].sum()/tot_n),
    numpy.hypot(bands["du"].sum()/tot_n, bands["dv"].sum()/tot_n) / numpy.sqrt(bands["sq"].sum()/tot_n)))
