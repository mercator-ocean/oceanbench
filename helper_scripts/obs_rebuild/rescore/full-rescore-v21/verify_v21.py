# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Verify the 2024-v2.1.0 currents rewrite on given days: mirror and remote store.

Gate a: recompute the expected default from inputs that the rewrite never wrote,
        two independent ways.
          a1  the LWS read-time view, materialised from the pre-rewrite store on
              2026-08-06, plus the 11/211 blanking
          a2  EWCT_FILTR/NSCT_FILTR read straight out of the raw netCDF archive,
              joined on obs_id exactly as materialize_strata.py does, minus the
              stored slippage, blanked on 11/211
Gate b: row count unchanged against the recorded pre-rewrite count
Gate c: every other float column byte-identical to its pre-rewrite digest
Gate d: kept fraction of currents rows
Gate e: remote store equals the local mirror

Usage: verify_v21.py YYYYMMDD [YYYYMMDD ...]
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas
import s3fs
import xarray
import zarr

sys.path.insert(0, "/scratch/jseillade/obs-rebuild")
import build_observations as builder

MIRROR = "/scratch/jseillade/obs-rebuild/store-v2"
LWS = "/scratch/jseillade/obs-rebuild/views2/LWS"
ARCHIVE = "/scratch/jseillade/obs-rebuild/raw-archive"
BUCKET_PREFIX = "oceanbench-bucket/dev/observations2024-v2"
ENDPOINT = "https://s3.waw3-1.cloudferro.com"
STATE_DIR = "/scratch/jseillade/obs-rebuild/rescore/full-rescore-v21/state"
NEW_VERSION = "2024-v2.1.0"
PATCH_NAME = "currents-wind-slippage-and-211-drop"
DROP_CODES = (11, 211)
OBS_TYPE_CURRENT = 3
UO = "eastward_sea_water_velocity"
VO = "northward_sea_water_velocity"
UNTOUCHED = (
    "depth", "latitude", "longitude", "sea_surface_height_above_geoid",
    "sea_water_potential_temperature", "sea_water_salinity",
    "uo_raw", "vo_raw", "uo_ws", "vo_ws", "sla_unfiltered",
    "temp_raw", "psal_raw", "temp_adjusted", "psal_adjusted",
)


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def same(a, b):
    return bool(np.array_equal(a, b, equal_nan=True))


def load_states():
    merged = {}
    for path in glob.glob(os.path.join(STATE_DIR, "rewrite-*.json")):
        with open(path) as fh:
            for day, record in json.load(fh).items():
                if record.get("status") == "ok" or day not in merged:
                    merged[day] = record
    return merged


def archive_filtr(day):
    path = os.path.join(ARCHIVE, day, f"GL_TS_DC_{day}_FILTR.nc")
    dataset = xarray.open_dataset(path)
    try:
        n = dataset.sizes["TIME"]
        frame = pandas.DataFrame({
            "obs_type": np.full(n, builder.OBS_TYPE_DRIFTER_CURRENT),
            "platform_code": builder.char_to_str(dataset["PLATFORM_CODE"].values, n),
            "depth": np.asarray(dataset["DEPH"].values)[:, 0].astype(np.float64),
        })
        frame["time_ns"] = builder.to_datetime_ns(dataset["TIME"].values)
        ids = builder.build_obs_ids(frame).to_numpy()
        uo = np.asarray(dataset["EWCT_FILTR"].values)[:, 0].astype(np.float64)
        vo = np.asarray(dataset["NSCT_FILTR"].values)[:, 0].astype(np.float64)
    finally:
        dataset.close()
    return {key: index for index, key in enumerate(ids)}, uo, vo


def main(argv):
    days = argv[1:]
    states = load_states()
    fs = s3fs.S3FileSystem(
        key=os.environ["CF_KEY"],
        secret=os.environ["CF_SECRET"],
        client_kwargs={"endpoint_url": ENDPOINT},
        config_kwargs={"s3": {"addressing_style": "path"}, "retries": {"max_attempts": 10}},
    )
    all_ok = True
    for day in days:
        record = states.get(day, {})
        group = zarr.open_group(os.path.join(MIRROR, f"{day}.zarr"), mode="r")
        uo = np.asarray(group[UO][:], dtype=np.float64)
        vo = np.asarray(group[VO][:], dtype=np.float64)
        uws = np.asarray(group["uo_ws"][:], dtype=np.float64)
        vws = np.asarray(group["vo_ws"][:], dtype=np.float64)
        obs_type = np.asarray(group["obs_type"][:])
        code = np.asarray(group["current_test"][:])
        obs_id = np.asarray(group["obs_id"][:])
        currents = obs_type == OBS_TYPE_CURRENT
        dropped = np.isin(code, DROP_CODES)

        # a1: LWS view plus the 11/211 blanking
        lws = xarray.open_dataset(os.path.join(LWS, f"{day}.zarr"), engine="zarr",
                                  decode_cf=False, consolidated=True)
        expect_uo = np.asarray(lws[UO].values, dtype=np.float64).copy()
        expect_vo = np.asarray(lws[VO].values, dtype=np.float64).copy()
        lws.close()
        expect_uo[dropped] = np.nan
        expect_vo[dropped] = np.nan
        gate_a1 = same(uo, expect_uo) and same(vo, expect_vo)

        # a2: straight from the raw FILTR netCDF, joined on obs_id
        index_map, arc_uo, arc_vo = archive_filtr(day)
        positions = np.full(uo.size, -1, dtype=np.int64)
        for position in np.flatnonzero(currents):
            positions[position] = index_map.get(obs_id[position], -1)
        matched = positions >= 0
        raw_uo = np.full(uo.size, np.nan)
        raw_vo = np.full(uo.size, np.nan)
        raw_uo[matched] = arc_uo[positions[matched]]
        raw_vo[matched] = arc_vo[positions[matched]]
        active = currents & matched & np.isfinite(raw_uo) & np.isfinite(raw_vo) & ~dropped
        # The builder blanks rows failing any other policy check too, so compare
        # only on the rows the store actually keeps.
        keep = active & np.isfinite(uo) & np.isfinite(vo)
        rebuilt_uo = raw_uo[keep] - np.where(np.isfinite(uws[keep]), uws[keep], 0.0)
        rebuilt_vo = raw_vo[keep] - np.where(np.isfinite(vws[keep]), vws[keep], 0.0)
        gate_a2 = same(uo[keep], rebuilt_uo) and same(vo[keep], rebuilt_vo)

        finite_after = np.isfinite(uo) & np.isfinite(vo)
        checks = {
            "gate_a1_matches_lws_view": gate_a1,
            "gate_a2_matches_raw_archive": gate_a2,
            "gate_a2_rows_compared": int(keep.sum()),
            "gate_b_row_count": int(uo.size),
            "gate_b_row_count_recorded": record.get("n_rows"),
            "gate_b_row_count_unchanged": int(uo.size) == record.get("n_rows"),
            "gate_c_untouched_identical": all(
                digest(np.asarray(group[name][:], dtype=np.float64))
                == record.get("untouched_sha256", {}).get(name)
                for name in UNTOUCHED),
            "gate_d_currents": int(currents.sum()),
            "gate_d_finite_after": int(finite_after.sum()),
            "gate_d_kept_fraction": round(float(finite_after.sum()) / max(int(currents.sum()), 1), 5),
            "gate_d_blanked_by_code": int((dropped & np.isin(code, DROP_CODES)).sum()),
            "n_no_slippage_rows": int((finite_after & ~(np.isfinite(uws) & np.isfinite(vws))).sum()),
            "local_version": group.attrs.get("obs_basis_version"),
        }

        remote = zarr.open_group(zarr.storage.FSStore(f"{BUCKET_PREFIX}/{day}.zarr", fs=fs), mode="r")
        ruo = np.asarray(remote[UO][:], dtype=np.float64)
        rvo = np.asarray(remote[VO][:], dtype=np.float64)
        with fs.open(f"{BUCKET_PREFIX}/{day}.manifest.json") as fh:
            manifest = json.load(fh)
        patches = [p for p in manifest.get("patches", []) if p["patch"] == PATCH_NAME]
        checks.update({
            "gate_e_remote_uo_equals_mirror": digest(ruo) == digest(uo),
            "gate_e_remote_vo_equals_mirror": digest(rvo) == digest(vo),
            "gate_e_remote_rows": int(ruo.size),
            "remote_version": remote.attrs.get("obs_basis_version"),
            "manifest_version": manifest.get("obs_basis_version"),
            "manifest_patch_entries": len(patches),
            "remote_policy_version": json.loads(remote.attrs["policy"])["policy_version"],
            "remote_drop_codes": json.loads(remote.attrs["policy"])["drop_undrogued_current_test"],
        })

        hard = [
            checks["gate_a1_matches_lws_view"],
            checks["gate_a2_matches_raw_archive"],
            checks["gate_a2_rows_compared"] > 0,
            checks["gate_b_row_count_unchanged"],
            checks["gate_c_untouched_identical"],
            checks["gate_e_remote_uo_equals_mirror"],
            checks["gate_e_remote_vo_equals_mirror"],
            checks["gate_e_remote_rows"] == checks["gate_b_row_count"],
            checks["local_version"] == NEW_VERSION,
            checks["remote_version"] == NEW_VERSION,
            checks["manifest_version"] == NEW_VERSION,
            checks["manifest_patch_entries"] == 1,
            checks["remote_policy_version"] == "v2.1.0",
            checks["remote_drop_codes"] == list(DROP_CODES),
        ]
        checks["VERDICT"] = "PASS" if all(hard) else "FAIL"
        all_ok = all_ok and all(hard)
        print(day, json.dumps(checks, sort_keys=True), flush=True)
    print("ALL_PASS" if all_ok else "SOME_FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
