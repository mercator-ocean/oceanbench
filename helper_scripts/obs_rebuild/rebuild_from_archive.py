# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Rebuild one day from the local raw archive with the current builder.

No network for the sources: the in-situ downloader and the SLA downloader are
replaced by readers over /scratch/jseillade/obs-rebuild/raw-archive, so the
result depends only on the builder code and the archived netCDF bytes.

Modes:
  --compare  build to a throwaway target and compare with the mirror
  --repair   additionally write the freshly built longitude into the mirror and
             upload it, for days patched with an earlier normalisation

Usage: rebuild_from_archive.py YYYY-MM-DD --workdir DIR [--repair]
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, "/scratch/jseillade/obs-rebuild")
import build_observations as bo  # noqa: E402

ARCHIVE = Path("/scratch/jseillade/obs-rebuild/raw-archive")
MIRROR = "/scratch/jseillade/obs-rebuild/store-v2"
BUCKET_PREFIX = "oceanbench-bucket/dev/observations2024-v2"
ENDPOINT = "https://s3.waw3-1.cloudferro.com"
STATE_PATH = "/scratch/jseillade/obs-rebuild/patch-lon-state.json"
PATCH_NAME = "sla-longitude-normalization"
OBS_TYPE_SLA = 4


def digest(values):
    return hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()


def fake_download(source_fs, key, local_path):
    """Copy the archived netCDF instead of fetching it."""
    name = Path(key).name
    date_dir = ARCHIVE / name.split("_")[-2].split(".")[0][:8]
    candidates = list(ARCHIVE.glob(f"*/{name}"))
    src = (date_dir / name) if (date_dir / name).exists() else (candidates[0] if candidates else None)
    if src is None:
        return None
    shutil.copy2(src, local_path)
    return {
        "key": key,
        "size": src.stat().st_size,
        "etag": bo.sha256_of_file(src)[:32],
        "etag_kind": "sha256_prefix",
        "download_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "local_name": name,
    }


def fake_extract_sla(date, tmp_dir, archive_dir, retries=3):
    frames, files, satellites_found = [], [], []
    day_dir = ARCHIVE / f"{date:%Y%m%d}"
    for mission, dataset_id in bo.SLA_SATELLITES.items():
        paths = sorted(day_dir.glob(f"dt_global_{mission}_phy_l3_1hz_{date:%Y%m%d}_*.nc"))
        if not paths:
            continue
        satellites_found.append(mission)
        for path in paths:
            files.append(
                {
                    "key": f"{dataset_id}/{path.name}",
                    "size": path.stat().st_size,
                    "etag": bo.sha256_of_file(path)[:32],
                    "etag_kind": "sha256_prefix",
                    "download_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "stream": "sla",
                    "mission": mission,
                }
            )
            frame = bo.extract_sla_nc(path, mission)
            files[-1]["n_rows"] = int(len(frame))
            if len(frame) > 0:
                frames.append(frame)
    if not frames:
        return bo.empty_frame(), files, satellites_found
    import pandas as pd

    return pd.concat(frames, ignore_index=True), files, satellites_found


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("date")
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--repair", action="store_true")
    args = parser.parse_args(argv)

    date = dt.date.fromisoformat(args.date)
    day = f"{date:%Y%m%d}"
    workdir = Path(args.workdir)
    target = workdir / "fresh"
    target.mkdir(parents=True, exist_ok=True)

    bo.download_source_file = fake_download
    bo.extract_sla = fake_extract_sla

    build_args = argparse.Namespace(
        target=str(target),
        obs_basis_version=bo.DEFAULT_OBS_BASIS_VERSION,
        archive_dir=None,
        tmp_root=str(workdir),
        overwrite=True,
        allow_missing_sla=True,
        min_satellites=bo.DEFAULT_MIN_SATELLITES,
    )
    script_sha = bo.sha256_of_file(Path("/scratch/jseillade/obs-rebuild/build_observations.py"))
    result = bo.process_day(date, build_args, script_sha)
    print(json.dumps(result, default=str), flush=True)

    fresh = zarr.open_group(str(target / f"{day}.zarr"), mode="r")
    stored = zarr.open_group(os.path.join(MIRROR, f"{day}.zarr"), mode="r")
    flon = np.asarray(fresh["longitude"][:], dtype=np.float64)
    slon = np.asarray(stored["longitude"][:], dtype=np.float64)
    fid = np.asarray(fresh["obs_id"][:])
    sid = np.asarray(stored["obs_id"][:])
    fot = np.asarray(fresh["obs_type"][:])
    sot = np.asarray(stored["obs_type"][:])

    same_ids = bool(np.array_equal(fid, sid))
    report = {
        "day": day,
        "fresh_rows": int(flon.size),
        "stored_rows": int(slon.size),
        "obs_id_identical_in_order": same_ids,
        "obs_type_identical": bool(np.array_equal(fot, sot)),
        "lon_identical": bool(np.array_equal(flon, slon)),
        "sla_lon_identical": bool(np.array_equal(flon[fot == OBS_TYPE_SLA], slon[sot == OBS_TYPE_SLA])),
        "max_abs_lon_diff": float(np.nanmax(np.abs(flon - slon))) if flon.size == slon.size else None,
        "n_lon_differing": int((~((flon == slon) | (np.isnan(flon) & np.isnan(slon)))).sum())
        if flon.size == slon.size
        else None,
        "fresh_lon_in_range": bool(np.all(np.abs(flon[np.isfinite(flon)]) <= 180.0)),
    }

    if same_ids:
        order_f = np.argsort(fid, kind="stable")
        order_s = np.argsort(sid, kind="stable")
        report["sla_lon_identical_sorted_by_obs_id"] = bool(
            np.array_equal(flon[order_f][fot[order_f] == OBS_TYPE_SLA], slon[order_s][sot[order_s] == OBS_TYPE_SLA])
        )

    if args.repair:
        if not same_ids:
            report["repair"] = "REFUSED_OBS_ID_MISMATCH"
            print(json.dumps(report, indent=1, sort_keys=True))
            return 1
        import s3fs

        fs = s3fs.S3FileSystem(
            key=os.environ["CF_KEY"],
            secret=os.environ["CF_SECRET"],
            client_kwargs={"endpoint_url": ENDPOINT},
            config_kwargs={"s3": {"addressing_style": "path"}, "retries": {"max_attempts": 10}},
        )
        target_group = zarr.open_group(os.path.join(MIRROR, f"{day}.zarr"), mode="r+")
        target_group["longitude"][:] = flon
        # Rows the original defect touched: SLA rows now west of the meridian
        # came from 0..360 east of 180, plus any in-situ row exactly on -180.
        rows_affected = int(((fot == OBS_TYPE_SLA) & (flon < 0.0)).sum())
        rows_affected_non_sla = int(((fot != OBS_TYPE_SLA) & (flon == -180.0)).sum())
        rows_affected += rows_affected_non_sla

        mpath = os.path.join(MIRROR, f"{day}.manifest.json")
        with open(mpath) as fh:
            manifest = json.load(fh)
        for entry in manifest.get("patches", []):
            if entry["patch"] == PATCH_NAME:
                entry["rows_affected"] = rows_affected
                entry["rows_affected_non_sla"] = rows_affected_non_sla
                entry["builder_script_sha256"] = script_sha
        with open(mpath, "w") as fh:
            json.dump(manifest, fh, indent=2, sort_keys=True)

        uploads = [f"{day}.manifest.json"] + sorted(
            f"{day}.zarr/longitude/{c}"
            for c in os.listdir(os.path.join(MIRROR, f"{day}.zarr", "longitude"))
            if not c.startswith(".")
        )
        for rel in uploads:
            fs.put_file(os.path.join(MIRROR, rel), f"{BUCKET_PREFIX}/{rel}")

        with open(STATE_PATH) as fh:
            state = json.load(fh)
        rec = state.get(day, {})
        rec.update(
            {
                "status": "ok",
                "n_rows_changed": rows_affected,
                "n_non_sla_rows_changed": rows_affected_non_sla,
                "max_shift": 360.0,
                "non_sla_lon_sha256_before": digest(flon[fot != OBS_TYPE_SLA].astype(np.float64)),
                "non_sla_lon_sha256_after": digest(flon[fot != OBS_TYPE_SLA].astype(np.float64)),
                "repaired_from_archive_rebuild": True,
                "builder_sha256": script_sha,
                "repaired_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
        )
        state[day] = rec
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=1, sort_keys=True)
        os.replace(tmp, STATE_PATH)
        report["repair"] = "done"
        report["rows_affected"] = rows_affected
        report["rows_affected_non_sla"] = rows_affected_non_sla
        report["uploaded"] = uploads

    print(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
