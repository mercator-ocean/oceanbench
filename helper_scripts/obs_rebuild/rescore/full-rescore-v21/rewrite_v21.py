# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Rewrite observations2024-v2 in place to 2024-v2.1.0: adopted currents policy.

New default currents columns:
    uo = EWCT_FILTR - uo_ws  where uo_ws is finite, EWCT_FILTR otherwise
    vo = NSCT_FILTR - vo_ws  where vo_ws is finite, NSCT_FILTR otherwise
    blanked (NaN) where current_test is 11 (SAW 011) or 211

Sign convention is the LWS one proven on 2026-08-06: subtracting the wind
slippage improves scores, adding it degrades them. The arithmetic here is
byte-for-byte materialize_ws.py's LWS branch, applied to the published column
instead of a read-time view, plus the extra 211 blanking.

Nothing else moves: no other column, no row, no row count, no row identity.
Rows already NaN in the default columns stay NaN, because NaN minus anything
is NaN.

NOT idempotent: subtracting the slippage twice would be wrong. The guard is the
recorded obs_basis_version plus the manifest patch entry, both checked before
any write, plus a per-month state file.

Per day: patch the local mirror, edit .zattrs and the .zattrs entry inside
.zmetadata, edit the manifest, then upload only the changed objects to S3.

Usage: rewrite_v21.py 2024-01 [2024-02 ...]   (month prefixes of the day files)
"""
import datetime as dt
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
STATE_DIR = "/scratch/jseillade/obs-rebuild/rescore/full-rescore-v21/state"
BUILDER = "/scratch/jseillade/obs-rebuild/build_observations.py"
SELF = os.path.abspath(__file__)
PATCH_NAME = "currents-wind-slippage-and-211-drop"
OLD_VERSION = "2024-v2.0.1"
NEW_VERSION = "2024-v2.1.0"
OBS_TYPE_CURRENT = 3
DROP_CODES = (11, 211)
MAX_PLAUSIBLE_CORRECTION = 5.0  # m/s, a wind slippage larger than this is a bug

UO = "eastward_sea_water_velocity"
VO = "northward_sea_water_velocity"
UNTOUCHED = (
    "depth", "latitude", "longitude", "sea_surface_height_above_geoid",
    "sea_water_potential_temperature", "sea_water_salinity",
    "uo_raw", "vo_raw", "uo_ws", "vo_ws", "sla_unfiltered",
    "temp_raw", "psal_raw", "temp_adjusted", "psal_adjusted",
)


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def load_state(path):
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return {}


def save_state(state, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1, sort_keys=True)
    os.replace(tmp, path)


def patch_day(day, fs, builder_sha, script_sha):
    zpath = os.path.join(MIRROR, f"{day}.zarr")
    mpath = os.path.join(MIRROR, f"{day}.manifest.json")
    if not os.path.exists(os.path.join(zpath, ".zmetadata")):
        return {"status": "MISSING_MIRROR"}

    with open(mpath) as fh:
        manifest = json.load(fh)
    if any(p.get("patch") == PATCH_NAME for p in manifest.get("patches", [])):
        return {"status": "ALREADY_PATCHED_MANIFEST"}

    group = zarr.open_group(zpath, mode="r+")
    if group.attrs.get("obs_basis_version") != OLD_VERSION:
        return {"status": f"UNEXPECTED_VERSION:{group.attrs.get('obs_basis_version')}"}

    uo = np.asarray(group[UO][:], dtype=np.float64)
    vo = np.asarray(group[VO][:], dtype=np.float64)
    uws = np.asarray(group["uo_ws"][:], dtype=np.float64)
    vws = np.asarray(group["vo_ws"][:], dtype=np.float64)
    obs_type = np.asarray(group["obs_type"][:])
    code = np.asarray(group["current_test"][:])

    currents = obs_type == OBS_TYPE_CURRENT
    finite_before = np.isfinite(uo) & np.isfinite(vo)
    dropped = np.isin(code, DROP_CODES)

    correction_u = np.where(np.isfinite(uws), uws, 0.0)
    correction_v = np.where(np.isfinite(vws), vws, 0.0)
    new_uo = uo - correction_u
    new_vo = vo - correction_v
    new_uo[dropped] = np.nan
    new_vo[dropped] = np.nan
    finite_after = np.isfinite(new_uo) & np.isfinite(new_vo)

    changed_u = ~((new_uo == uo) | (np.isnan(new_uo) & np.isnan(uo)))
    changed_v = ~((new_vo == vo) | (np.isnan(new_vo) & np.isnan(vo)))
    changed = changed_u | changed_v
    shifted = finite_before & finite_after & changed
    blanked = finite_before & ~finite_after

    record = {
        "n_rows": int(uo.size),
        "n_currents": int(currents.sum()),
        "n_finite_before": int(finite_before.sum()),
        "n_finite_after": int(finite_after.sum()),
        "n_rows_changed": int(changed.sum()),
        "n_rows_shifted": int(shifted.sum()),
        "n_rows_blanked": int(blanked.sum()),
        "n_blanked_expected": int((finite_before & dropped).sum()),
        "n_ws_applied": int((finite_after & np.isfinite(uws) & np.isfinite(vws)).sum()),
        "n_changed_non_current": int((changed & ~currents).sum()),
        "max_abs_correction": float(max(np.abs(correction_u[shifted]).max(),
                                        np.abs(correction_v[shifted]).max())) if shifted.any() else 0.0,
        "uo_sha256_before": digest(uo),
        "vo_sha256_before": digest(vo),
        "uo_sha256_after": digest(new_uo),
        "vo_sha256_after": digest(new_vo),
        "untouched_sha256": {name: digest(np.asarray(group[name][:], dtype=np.float64))
                             for name in UNTOUCHED},
        "patched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder_sha256": builder_sha,
        "patch_script_sha256": script_sha,
    }

    # Refusal gates. Any of these means the assumption behind the rewrite broke.
    if record["n_changed_non_current"]:
        record["status"] = "REFUSED_NON_CURRENT_ROW_CHANGED"
        return record
    if record["n_rows_blanked"] != record["n_blanked_expected"]:
        record["status"] = "REFUSED_BLANK_COUNT_MISMATCH"
        return record
    if shifted.any() and record["max_abs_correction"] > MAX_PLAUSIBLE_CORRECTION:
        record["status"] = "REFUSED_CORRECTION_TOO_LARGE"
        return record
    only_expected = changed & ~(finite_before & (
        (np.isfinite(uws) & np.isfinite(vws)) | dropped))
    if bool(only_expected.any()):
        record["status"] = "REFUSED_UNEXPECTED_ROW_CHANGED"
        record["n_unexpected"] = int(only_expected.sum())
        return record

    group[UO][:] = new_uo
    group[VO][:] = new_vo

    old_policy = json.loads(group.attrs["policy"])
    new_policy = dict(old_policy)
    new_policy["policy_version"] = "v2.1.0"
    new_policy["drop_undrogued_current_test"] = list(DROP_CODES)
    new_policy["current_wind_slippage_removed"] = True
    new_policy["current_wind_slippage_source"] = "uo_ws/vo_ws, subtracted where finite"
    old_counts = json.loads(group.attrs["row_counts_after_policy"])
    new_counts = dict(old_counts)
    new_counts["currents"] = record["n_finite_after"]

    for path, key in ((os.path.join(zpath, ".zattrs"), None),
                      (os.path.join(zpath, ".zmetadata"), ".zattrs")):
        with open(path) as fh:
            blob = json.load(fh)
        target = blob if key is None else blob["metadata"][key]
        target["obs_basis_version"] = NEW_VERSION
        target["policy"] = json.dumps(new_policy)
        target["row_counts_after_policy"] = json.dumps(new_counts)
        with open(path, "w") as fh:
            json.dump(blob, fh, indent=4, sort_keys=True)

    manifest["obs_basis_version"] = NEW_VERSION
    manifest.setdefault("patches", []).append({
        "patch": PATCH_NAME,
        "applied_at_utc": record["patched_at_utc"],
        "builder_script_sha256": builder_sha,
        "patch_script_sha256": script_sha,
        "obs_basis_version_from": OLD_VERSION,
        "obs_basis_version_to": NEW_VERSION,
        "rows_affected": record["n_rows_changed"],
        "rows_shifted_by_wind_slippage": record["n_rows_shifted"],
        "rows_blanked_by_code_drop": record["n_rows_blanked"],
        "currents_finite_before": record["n_finite_before"],
        "currents_finite_after": record["n_finite_after"],
        "row_counts_after_policy_before": old_counts,
        "policy_before": old_policy,
        "description": (
            "default current columns rewritten to EWCT_FILTR/NSCT_FILTR minus the "
            "wind slippage uo_ws/vo_ws where the slippage is finite, unchanged where "
            "it is missing, and blanked where current_test is 11 (SAW 011) or 211; "
            "raw and filtered inputs, every other column, row identity and row count "
            "are unchanged"
        ),
    })
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    uploads = [f"{day}.manifest.json", f"{day}.zarr/.zattrs", f"{day}.zarr/.zmetadata"]
    for column in (UO, VO):
        cdir = os.path.join(zpath, column)
        uploads += sorted(f"{day}.zarr/{column}/{c}" for c in os.listdir(cdir)
                          if not c.startswith("."))
    for rel in uploads:
        fs.put_file(os.path.join(MIRROR, rel), f"{BUCKET_PREFIX}/{rel}")
    record["uploaded"] = uploads
    record["status"] = "ok"
    return record


def main(argv):
    months = argv[1:]
    if not months:
        raise SystemExit("give one or more YYYY-MM month prefixes")
    os.makedirs(STATE_DIR, exist_ok=True)
    builder_sha = sha256_of_file(BUILDER)
    script_sha = sha256_of_file(SELF)
    fs = s3fs.S3FileSystem(
        key=os.environ["CF_KEY"],
        secret=os.environ["CF_SECRET"],
        client_kwargs={"endpoint_url": ENDPOINT},
        config_kwargs={"s3": {"addressing_style": "path"}, "retries": {"max_attempts": 10}},
    )
    state_path = os.path.join(STATE_DIR, f"rewrite-{'-'.join(months)}.json")
    state = load_state(state_path)
    # A YYYY-MM argument selects a month, a YYYYMMDD argument selects one day.
    prefixes = tuple(m.replace("-", "") for m in months)
    days = sorted(
        name[: -len(".zarr")]
        for name in os.listdir(MIRROR)
        if name.endswith(".zarr") and name.startswith(prefixes)
    )
    print(f"builder sha {builder_sha}", flush=True)
    print(f"patch script sha {script_sha}", flush=True)
    print(f"{len(days)} days selected for {months}", flush=True)
    totals = {"changed": 0, "shifted": 0, "blanked": 0, "finite_after": 0}
    for day in days:
        if state.get(day, {}).get("status") == "ok":
            print(f"{day} skip (already ok)", flush=True)
            continue
        record = patch_day(day, fs, builder_sha, script_sha)
        state[day] = record
        save_state(state, state_path)
        totals["changed"] += record.get("n_rows_changed", 0)
        totals["shifted"] += record.get("n_rows_shifted", 0)
        totals["blanked"] += record.get("n_rows_blanked", 0)
        totals["finite_after"] += record.get("n_finite_after", 0)
        print(
            f"{day} {record['status']} rows={record.get('n_rows')} "
            f"changed={record.get('n_rows_changed')} shifted={record.get('n_rows_shifted')} "
            f"blanked={record.get('n_rows_blanked')} finite={record.get('n_finite_before')}"
            f"->{record.get('n_finite_after')}",
            flush=True,
        )
        if record["status"] not in ("ok", "ALREADY_PATCHED_MANIFEST"):
            raise SystemExit(f"{day}: {record['status']}")
    print(f"BATCH DONE days={len(days)} {totals}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
