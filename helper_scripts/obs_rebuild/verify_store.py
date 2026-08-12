# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import json, sys, collections
import concurrent.futures as cf
import numpy as np
sys.path.insert(0, "/scratch/jseillade/obs-rebuild")
import build_observations as B
import s3fs, xarray as xr

ROOT = "oceanbench-bucket/dev/observations2024-v2"
so = B.target_storage_options()
fs = s3fs.S3FileSystem(**so)
fs.invalidate_cache()
entries = fs.ls(ROOT)
zarrs = sorted(e for e in entries if e.endswith(".zarr"))
mans = sorted(e for e in entries if e.endswith(".manifest.json"))
tmps = [e for e in entries if e.endswith(".tmp")]
print("=== INVENTORY")
print("zarr_dirs", len(zarrs), "manifests", len(mans), "leftover_tmp", len(tmps))
have = {z.split("/")[-1][:8] for z in zarrs}
import datetime as dt
expect = []
d = dt.date(2024, 1, 1)
while d <= dt.date(2025, 1, 4):
    expect.append(d.strftime("%Y%m%d")); d += dt.timedelta(days=1)
print("expected_days", len(expect), "missing", sorted(set(expect) - have))

def load(m):
    with fs.open(m, "rb") as h:
        return json.loads(h.read().decode())

with cf.ThreadPoolExecutor(max_workers=16) as pool:
    manifests = list(pool.map(load, mans))

print("=== AGGREGATE FROM MANIFESTS")
tot_rows = sum(m["n_obs_total"] for m in manifests)
tot_kept = sum(m["n_obs_kept"] for m in manifests)
print("total_rows", tot_rows, "total_kept", tot_kept)
per_stream_before = collections.Counter()
per_stream_after = collections.Counter()
per_month = collections.Counter()
per_month_sla = collections.Counter()
no_h2b, few_sats, dups = [], [], 0
for m in manifests:
    for k, v in m["row_counts_before_policy"].items():
        per_stream_before[k] += v
    for k, v in m["row_counts_after_policy"].items():
        per_stream_after[k] += v
    per_month[m["date"][:7]] += 1
    per_month_sla[m["date"][:7]] += m["row_counts_after_policy"]["sla"]
    sats = m["sla_satellites_found"]
    if "h2b" not in sats:
        no_h2b.append(m["date"])
    if len(sats) < 6:
        few_sats.append((m["date"], sats))
    dups += m["n_duplicates_removed"]
print("rows_before_policy_per_stream", dict(per_stream_before))
print("rows_kept_per_stream", dict(per_stream_after))
print("duplicates_removed_total", dups)
print("days_per_month", dict(sorted(per_month.items())))
print("mean_sla_kept_per_day_by_month",
      {k: round(per_month_sla[k] / per_month[k]) for k in sorted(per_month)})
print("h2b_absent_days", len(no_h2b))
print("h2b_absent_list", no_h2b)
print("days_with_fewer_than_6_satellites", few_sats)
sizes = fs.du(ROOT)
print("=== SIZE")
print("store_bytes", sizes, "store_GiB", round(sizes / 2**30, 2))

print("=== SPOT DAYS")
for day in ["20240105", "20240615", "20241120"]:
    ds = xr.open_zarr(f"s3://{ROOT}/{day}.zarr", storage_options=so, consolidated=True)
    ot = ds.obs_type.values
    keep = ds.qc_keep.values
    oid = ds.obs_id.values
    print("---", day, "n_obs", ds.sizes["obs"])
    for name, code in [("argo", 1), ("drifter_sst", 2), ("drifter_cur", 3), ("sla", 4)]:
        sel = ot == code
        n = int(sel.sum())
        k = int((keep[sel] == 1).sum())
        print(f"  {name:12s} rows={n:8d} kept={k:8d} keep_rate={0 if n==0 else round(100*k/n,2)}%")
    print("  unique_obs_id", len(np.unique(oid)) == len(oid), len(np.unique(oid)), "of", len(oid))
    cur = ot == 3
    dr = ds.drogued.values[cur]
    print("  undrogued_frac_%", round(100 * float((dr == 0).mean()), 2),
          "unknown_%", round(100 * float((dr < 0).mean()), 2))
    a = ds.attrs
    print("  attrs_present", all(x in a for x in
          ["obs_basis_version", "builder_script_sha256", "policy", "source_files",
           "sla_satellites_found", "package_versions", "row_counts_before_policy"]))
    print("  basis", a.get("obs_basis_version"), "sha", a.get("builder_script_sha256", "")[:12],
          "sats", a.get("sla_satellites_found"))
    print("  n_dup_removed", a.get("n_duplicates_removed"),
          "before", a.get("row_counts_before_policy"), "after", a.get("row_counts_after_policy"))
    ds.close()

print("=== 20240615 VS SMOKE TEST")
import glob, os
cands = sorted(glob.glob("/scratch/jseillade/obs-v2*/20240615.zarr") +
               glob.glob("/scratch/jseillade/obs-rebuild/*test*/20240615.zarr"))
print("local smoke stores found:", cands)
new = xr.open_zarr(f"s3://{ROOT}/20240615.zarr", storage_options=so, consolidated=True)
print("new_total_rows", new.sizes["obs"])
for name, code in [("argo", 1), ("drifter_sst", 2), ("drifter_cur", 3), ("sla", 4)]:
    sel = new.obs_type.values == code
    print(f"  {name:12s} rows={int(sel.sum()):8d} qc_keep1={int((new.qc_keep.values[sel]==1).sum()):8d}")
for c in cands:
    old = xr.open_zarr(c, consolidated=True)
    print("smoke", c, "rows", old.sizes["obs"], "identical_total", old.sizes["obs"] == new.sizes["obs"])
    for name, code in [("argo", 1), ("drifter_sst", 2), ("drifter_cur", 3), ("sla", 4)]:
        s2 = old.obs_type.values == code
        print(f"  smoke {name:12s} rows={int(s2.sum()):8d} qc_keep1={int((old.qc_keep.values[s2]==1).sum()):8d}")
    old.close()
new.close()
