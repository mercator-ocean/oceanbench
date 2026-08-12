# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Patch the observations2024-v2 store in place: longitude onto [-180, 180).

The builder passed DUACS L3 SLA longitude through on 0..360 while every in-situ
stream is on -180..180, so SLA rows east of 180 fell off the model grid. The
defect touches exactly one float64 column on a subset of rows and obs_id never
includes longitude, so a full rebuild is not needed.

Per day: patch the local mirror, edit .zattrs and the .zattrs entry inside
.zmetadata, edit the manifest, then upload only the changed objects to S3.
Idempotent: normalising twice is a no-op and the manifest patch entry is written
once. Progress is appended to a state file so a batch can be rerun safely.

Usage: patch_lon.py 2024-01 [2024-02 ...]   (month prefixes of the day files)
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
STATE_PATH = "/scratch/jseillade/obs-rebuild/patch-lon-state.json"
BUILDER = "/scratch/jseillade/obs-rebuild/build_observations.py"
PATCH_NAME = "sla-longitude-normalization"
OLD_VERSION = "2024-v2.0.0"
NEW_VERSION = "2024-v2.0.1"
OBS_TYPE_SLA = 4


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def normalise(lon):
    """Exactly the wrap the fixed builder applies, bit for bit.

    A single exact subtraction or addition of 360, never a modulo, so a value
    already inside the range keeps its exact bits.
    """
    out = lon.copy()
    finite = np.isfinite(out)
    out[finite & (out >= 180.0)] -= 360.0
    out[finite & (out < -180.0)] += 360.0
    still_out = finite & ((out >= 180.0) | (out < -180.0))
    if still_out.any():
        raise SystemExit(f"longitude outside [-540, 540) on {int(still_out.sum())} rows")
    return out


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as fh:
            return json.load(fh)
    return {}


def save_state(state):
    tmp = STATE_PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1, sort_keys=True)
    os.replace(tmp, STATE_PATH)


def patch_day(day, fs, builder_sha):
    """day is the YYYYMMDD stem. Returns the state record."""
    zpath = os.path.join(MIRROR, f"{day}.zarr")
    mpath = os.path.join(MIRROR, f"{day}.manifest.json")
    if not os.path.exists(os.path.join(zpath, ".zmetadata")):
        return {"status": "MISSING_MIRROR"}

    group = zarr.open_group(zpath, mode="r+")
    lon = np.asarray(group["longitude"][:], dtype=np.float64)
    obs_type = np.asarray(group["obs_type"][:])
    new_lon = normalise(lon)

    changed = ~((new_lon == lon) | (np.isnan(new_lon) & np.isnan(lon)))
    n_changed = int(changed.sum())
    changed_types = sorted(int(t) for t in np.unique(obs_type[changed])) if n_changed else []
    non_sla = obs_type != OBS_TYPE_SLA
    n_non_sla_changed = int((changed & non_sla).sum())
    shifts = np.abs(new_lon[changed] - lon[changed]) if n_changed else np.zeros(0)

    record = {
        "n_rows": int(lon.size),
        "n_sla_rows": int((~non_sla).sum()),
        "n_rows_changed": n_changed,
        "n_non_sla_rows_changed": n_non_sla_changed,
        "changed_obs_types": changed_types,
        "max_shift": float(shifts.max()) if n_changed else 0.0,
        "non_sla_lon_sha256_before": digest(lon[non_sla]),
        "non_sla_lon_sha256_after": digest(new_lon[non_sla]),
        "lon_min_before": float(np.nanmin(lon)) if lon.size else None,
        "lon_max_before": float(np.nanmax(lon)) if lon.size else None,
        "lon_min_after": float(np.nanmin(new_lon)) if new_lon.size else None,
        "lon_max_after": float(np.nanmax(new_lon)) if new_lon.size else None,
        "patched_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "builder_sha256": builder_sha,
    }
    # Every legitimate change is a whole 360 degree shift of the same point.
    # A non-SLA row can be caught by it only when its longitude is exactly
    # 180.0, which maps to -180.0, the same location on the fixed convention.
    if n_changed and not bool(np.all(np.abs(shifts - 360.0) < 1e-9)):
        record["status"] = "REFUSED_SHIFT_NOT_360"
        return record
    if n_non_sla_changed and not bool(np.all(np.abs(lon[changed & non_sla]) == 180.0)):
        record["status"] = "REFUSED_NON_SLA_ROW_NOT_AT_180"
        return record

    if n_changed:
        group["longitude"][:] = new_lon

    # Root attributes, both the loose copy and the consolidated copy.
    zattrs_path = os.path.join(zpath, ".zattrs")
    with open(zattrs_path) as fh:
        zattrs = json.load(fh)
    zattrs["obs_basis_version"] = NEW_VERSION
    with open(zattrs_path, "w") as fh:
        json.dump(zattrs, fh, indent=4, sort_keys=True)

    zmeta_path = os.path.join(zpath, ".zmetadata")
    with open(zmeta_path) as fh:
        zmeta = json.load(fh)
    zmeta["metadata"][".zattrs"]["obs_basis_version"] = NEW_VERSION
    with open(zmeta_path, "w") as fh:
        json.dump(zmeta, fh, indent=4, sort_keys=True)

    with open(mpath) as fh:
        manifest = json.load(fh)
    manifest["obs_basis_version"] = NEW_VERSION
    patches = manifest.setdefault("patches", [])
    if not any(p.get("patch") == PATCH_NAME for p in patches):
        patches.append(
            {
                "patch": PATCH_NAME,
                "applied_at_utc": record["patched_at_utc"],
                "builder_script_sha256": builder_sha,
                "rows_affected": n_changed,
                "rows_affected_non_sla": n_non_sla_changed,
                "obs_basis_version_from": OLD_VERSION,
                "obs_basis_version_to": NEW_VERSION,
                "description": (
                    "longitude normalised to [-180, 180); every affected row moved by "
                    "exactly 360 degrees, effectively all obs_type 4, plus any in-situ "
                    "row sitting exactly on 180.0 which becomes -180.0, the same point; "
                    "no other column, row identity or row count changed"
                ),
            }
        )
    else:
        record["manifest_entry"] = "already_present"
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)

    uploads = [f"{day}.manifest.json", f"{day}.zarr/.zattrs", f"{day}.zarr/.zmetadata"]
    lon_dir = os.path.join(zpath, "longitude")
    uploads += sorted(
        f"{day}.zarr/longitude/{c}" for c in os.listdir(lon_dir) if not c.startswith(".")
    )
    for rel in uploads:
        fs.put_file(os.path.join(MIRROR, rel), f"{BUCKET_PREFIX}/{rel}")
    record["uploaded"] = uploads
    record["status"] = "ok"
    return record


def main(argv):
    months = argv[1:]
    if not months:
        raise SystemExit("give one or more YYYY-MM month prefixes")
    builder_sha = sha256_of_file(BUILDER)
    fs = s3fs.S3FileSystem(
        key=os.environ["CF_KEY"],
        secret=os.environ["CF_SECRET"],
        client_kwargs={"endpoint_url": ENDPOINT},
        config_kwargs={"s3": {"addressing_style": "path"}, "retries": {"max_attempts": 10}},
    )
    state = load_state()
    days = sorted(
        name[: -len(".zarr")]
        for name in os.listdir(MIRROR)
        if name.endswith(".zarr")
        and any(name.startswith(m.replace("-", "")) for m in months)
    )
    print(f"builder sha {builder_sha}", flush=True)
    print(f"{len(days)} days selected for {months}", flush=True)
    total_changed = 0
    for day in days:
        if state.get(day, {}).get("status") == "ok":
            total_changed += state[day].get("n_rows_changed", 0)
            print(f"{day} skip (already ok)", flush=True)
            continue
        record = patch_day(day, fs, builder_sha)
        state[day] = record
        save_state(state)
        total_changed += record.get("n_rows_changed", 0)
        print(
            f"{day} {record['status']} rows={record.get('n_rows')} "
            f"sla={record.get('n_sla_rows')} changed={record.get('n_rows_changed')}",
            flush=True,
        )
        if record["status"] != "ok":
            raise SystemExit(f"{day}: {record['status']}")
    print(f"BATCH DONE days={len(days)} rows_changed_in_batch={total_changed}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
