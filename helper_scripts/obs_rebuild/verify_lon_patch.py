# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Verify the SLA longitude patch on given days, remote store and local mirror.

Usage: verify_lon_patch.py YYYYMMDD [YYYYMMDD ...]
"""
import hashlib
import json
import os
import sys

import numpy as np
import s3fs
import zarr

MIRROR = "/scratch/jseillade/obs-rebuild/store-v2"
BUCKET_PREFIX = "oceanbench-bucket/dev/observations2024-v2"
ENDPOINT = "https://s3.waw3-1.cloudferro.com"
STATE_PATH = "/scratch/jseillade/obs-rebuild/patch-lon-state.json"
NEW_VERSION = "2024-v2.0.1"
OBS_TYPE_SLA = 4


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def main(argv):
    days = argv[1:]
    fs = s3fs.S3FileSystem(
        key=os.environ["CF_KEY"],
        secret=os.environ["CF_SECRET"],
        client_kwargs={"endpoint_url": ENDPOINT},
        config_kwargs={"s3": {"addressing_style": "path"}, "retries": {"max_attempts": 10}},
    )
    with open(STATE_PATH) as fh:
        state = json.load(fh)

    ok = True
    for day in days:
        rec = state.get(day, {})
        local = zarr.open_group(os.path.join(MIRROR, f"{day}.zarr"), mode="r")
        llon = np.asarray(local["longitude"][:], dtype=np.float64)
        lot = np.asarray(local["obs_type"][:])
        store = zarr.storage.FSStore(f"{BUCKET_PREFIX}/{day}.zarr", fs=fs)
        remote = zarr.open_group(store, mode="r")
        rlon = np.asarray(remote["longitude"][:], dtype=np.float64)
        rot = np.asarray(remote["obs_type"][:])

        with fs.open(f"{BUCKET_PREFIX}/{day}.manifest.json") as fh:
            manifest = json.load(fh)
        patches = [p for p in manifest.get("patches", []) if p["patch"] == "sla-longitude-normalization"]

        checks = {
            "local_lon_in_range": bool(np.all(np.abs(llon[np.isfinite(llon)]) <= 180.0)),
            "remote_lon_in_range": bool(np.all(np.abs(rlon[np.isfinite(rlon)]) <= 180.0)),
            "local_max_abs_lon": float(np.nanmax(np.abs(llon))),
            "remote_equals_local": digest(rlon) == digest(llon),
            "obs_type_equal": bool(np.array_equal(lot, rot)),
            "n_rows": int(llon.size),
            "n_rows_state": rec.get("n_rows"),
            "n_sla": int((lot == OBS_TYPE_SLA).sum()),
            "n_sla_state": rec.get("n_sla_rows"),
            "sla_count_unchanged": int((lot == OBS_TYPE_SLA).sum()) == rec.get("n_sla_rows"),
            "non_sla_lon_unchanged": digest(llon[lot != OBS_TYPE_SLA])
            == rec.get("non_sla_lon_sha256_before"),
            "non_sla_lon_matches_expected": digest(llon[lot != OBS_TYPE_SLA])
            == rec.get("non_sla_lon_sha256_after"),
            "non_sla_rows_changed": rec.get("n_non_sla_rows_changed"),
            "max_shift": rec.get("max_shift"),
            "rows_affected_state": rec.get("n_rows_changed"),
            "local_version": local.attrs.get("obs_basis_version"),
            "remote_version": remote.attrs.get("obs_basis_version"),
            "manifest_version": manifest.get("obs_basis_version"),
            "manifest_patch_entries": len(patches),
            "manifest_patch_rows": patches[0]["rows_affected"] if patches else None,
            "manifest_patch_builder_sha": patches[0]["builder_script_sha256"] if patches else None,
        }
        hard = [
            checks["local_lon_in_range"],
            checks["remote_lon_in_range"],
            checks["remote_equals_local"],
            checks["obs_type_equal"],
            checks["sla_count_unchanged"],
            checks["non_sla_lon_matches_expected"],
            checks["local_version"] == NEW_VERSION,
            checks["remote_version"] == NEW_VERSION,
            checks["manifest_version"] == NEW_VERSION,
            checks["manifest_patch_entries"] == 1,
        ]
        checks["VERDICT"] = "PASS" if all(hard) else "FAIL"
        ok = ok and all(hard)
        print(day, json.dumps(checks, sort_keys=True), flush=True)
    print("ALL_PASS" if ok else "SOME_FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
