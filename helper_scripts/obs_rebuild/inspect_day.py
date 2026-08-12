#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Inspect one built observation day. Read only, prints a report."""
import json
import sys
import numpy as np
import pandas as pd
import xarray as xr

store = sys.argv[1] if len(sys.argv) > 1 else "/scratch/jseillade/obs-rebuild/test-store/20240615.zarr"
ds = xr.open_zarr(store, consolidated=True)
n = ds.sizes["obs"]
print("total rows:", n)

ot = ds["obs_type"].values
keep = ds["qc_keep"].values
names = {1: "argo_profile", 2: "drifter_sst", 3: "drifter_current", 4: "sla"}
print("\n-- rows per obs_type / qc_keep pass rate")
for code, label in names.items():
    m = ot == code
    tot = int(m.sum())
    kept = int((m & (keep == 1)).sum())
    rate = 100.0 * kept / tot if tot else 0.0
    print(f"  {code} {label:16s} total={tot:8d} kept={kept:8d} pass={rate:6.2f}%")
print(f"  {'ALL':18s} total={n:8d} kept={int((keep==1).sum()):8d} "
      f"pass={100.0*(keep==1).sum()/n:6.2f}%")

print("\n-- qc_reason counts (failing rows)")
reason = pd.Series(ds["qc_reason"].values)
print(reason[keep == 0].value_counts().to_string())

ct = ds["current_test"].values
dg = ds["drogued"].values
cur = ot == 3
print("\n-- drogue")
print("  current_test==11 (undrogued):", int((ct[cur] == 11).sum()))
print("  unknown drogue (drogued==-1):", int((dg[cur] == -1).sum()))
print("  drogued==1:", int((dg[cur] == 1).sum()), " drogued==0:", int((dg[cur] == 0).sum()))
print("  current_test value counts:")
print(pd.Series(ct[cur]).value_counts().head(10).to_string())

print("\n-- day alignment")
print("  day_misaligned rows:", int((reason == "day_misaligned").sum()))
tns = pd.to_datetime(pd.Series(ds["time_ns"].values))
print("  time_ns min:", tns.min(), " max:", tns.max())

print("\n-- string widths (max seen vs limit)")
limits = {"obs_id": 96, "platform_code": 32, "platform_source": 32,
          "sla_mission": 8, "qc_reason": 48, "data_mode": 1, "time": 19}
for name, lim in limits.items():
    vals = ds[name].values
    mx = max((len(str(v)) for v in vals), default=0)
    print(f"  {name:16s} dtype={str(ds[name].dtype):8s} max_len={mx:3d} limit={lim}")

print("\n-- legacy 9 columns, dtype check")
legacy = ["depth", "latitude", "longitude", "time",
          "sea_surface_height_above_geoid", "sea_water_potential_temperature",
          "sea_water_salinity", "eastward_sea_water_velocity",
          "northward_sea_water_velocity"]
for name in legacy:
    present = name in ds.variables
    print(f"  {name:34s} present={present} dtype={ds[name].dtype if present else '-'}")
print("  time sample:", ds["time"].values[:3])

print("\n-- undrogued rows: legacy uo/vo NaN, raw retained")
und = cur & (ct == 11)
if und.any():
    uo = ds["eastward_sea_water_velocity"].values[und]
    uor = ds["uo_raw"].values[und]
    print("  n undrogued:", int(und.sum()))
    print("  legacy uo finite count:", int(np.isfinite(uo).sum()), "(want 0)")
    print("  uo_raw finite count:", int(np.isfinite(uor).sum()), "of", int(und.sum()))

print("\n-- attrs present")
for key in ["policy", "builder_script_sha256", "package_versions", "source_files",
            "row_counts_before_policy", "row_counts_after_policy",
            "n_duplicates_removed", "obs_basis_version", "sla_satellites_found"]:
    val = ds.attrs.get(key)
    ok = val is not None
    show = (str(val)[:70] + "...") if ok and len(str(val)) > 70 else val
    print(f"  {key:28s} {'OK ' if ok else 'MISSING'} {show}")
print("  n_duplicates_removed =", ds.attrs.get("n_duplicates_removed"))
ds.close()
