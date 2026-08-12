#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Deeper verification of the built day: uniqueness, FILTR basis, manifest."""
import json
import numpy as np
import pandas as pd
import xarray as xr

ROOT = "/scratch/jseillade/obs-rebuild"
STORE = f"{ROOT}/test-store/20240615.zarr"
ARCHIVE = f"{ROOT}/raw-archive/20240615"

ds = xr.open_zarr(STORE, consolidated=True)
oid = ds["obs_id"].values
print("obs_id: n =", len(oid), " unique =", len(np.unique(oid)),
      " UNIQUE" if len(np.unique(oid)) == len(oid) else " *** NOT UNIQUE ***")

print("\nplatform_code samples per obs_type:")
ot = ds["obs_type"].values
for code in [1, 2, 3, 4]:
    vals = ds["platform_code"].values[ot == code][:3]
    src = ds["platform_source"].values[ot == code][:3]
    print(f"  ot={code} platform_code={list(vals)} platform_source={list(src)}")
print("obs_id samples:", list(oid[:2]), list(oid[ot == 3][:1]))

print("\nargo_cycle (CYCLE_NUMBER absent in CO_PR_PF):")
print("  value counts:", pd.Series(ds["argo_cycle"].values[ot == 1]).value_counts().head(3).to_dict())
print("data_mode counts ot=1:", pd.Series(ds["data_mode"].values[ot == 1]).value_counts().to_dict())

# ---- FILTR basis check against the source file
print("\n-- legacy uo/vo vs source EWCT_FILTR (drifter currents)")
src = xr.open_dataset(f"{ARCHIVE}/GL_TS_DC_20240615_FILTR.nc")
ew_f = np.asarray(src["EWCT_FILTR"].values)[:, 0]
ns_f = np.asarray(src["NSCT_FILTR"].values)[:, 0]
ew_r = np.asarray(src["EWCT"].values)[:, 0]
ct = np.asarray(src["CURRENT_TEST"].values).ravel()

cur = ot == 3
uo = ds["eastward_sea_water_velocity"].values[cur]
vo = ds["northward_sea_water_velocity"].values[cur]
uo_raw = ds["uo_raw"].values[cur]
keep = ds["qc_keep"].values[cur] == 1

# rows are in source order for this stream
ok = keep & np.isfinite(uo)
print("  compared rows:", int(ok.sum()))
print("  max |legacy_uo - EWCT_FILTR| =", float(np.nanmax(np.abs(uo[ok] - ew_f[ok]))))
print("  max |legacy_vo - NSCT_FILTR| =", float(np.nanmax(np.abs(vo[ok] - ns_f[ok]))))
print("  max |legacy_uo - EWCT(raw)|  =", float(np.nanmax(np.abs(uo[ok] - ew_r[ok]))),
      "(want clearly nonzero: legacy is NOT the raw basis)")
print("  max |uo_raw - EWCT(raw)|     =", float(np.nanmax(np.abs(uo_raw[ok] - ew_r[ok]))))
print("  sample legacy_uo :", np.round(uo[ok][:5], 6).tolist())
print("  sample EWCT_FILTR:", np.round(ew_f[ok][:5], 6).tolist())
print("  sample EWCT raw  :", np.round(ew_r[ok][:5], 6).tolist())

und = ct == 11
print("\n  undrogued (CURRENT_TEST==11) in source:", int(und.sum()),
      " in store:", int((ds['current_test'].values[cur] == 11).sum()))
print("  their legacy uo all NaN:", bool(np.all(~np.isfinite(uo[und]))))
print("  their uo_raw matches source EWCT:",
      bool(np.allclose(uo_raw[und], ew_r[und], equal_nan=True)))
src.close()

# ---- manifest
print("\n-- manifest")
man = json.load(open(f"{ROOT}/test-store/20240615.manifest.json"))
print("  valid JSON, keys:", sorted(man.keys()))
nfiles = 0
for stream, recs in man["source_files"].items():
    for r in recs:
        nfiles += 1
        has_ck = bool(r.get("etag"))
        print(f"    {stream:12s} {r['key'].split('/')[-1]:52s} size={r['size']:>10d} "
              f"rows={r.get('n_rows'):>7} checksum={'yes' if has_ck else 'NO'}")
print("  total source files listed:", nfiles)
print("  n_obs_total:", man["n_obs_total"], " n_obs_kept:", man["n_obs_kept"],
      " dups removed:", man["n_duplicates_removed"])
print("  policy accepted_depth_qc_flags:", man["policy"].get("accepted_depth_qc_flags"))
ds.close()
