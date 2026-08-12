# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Sweep every day of the store, local mirror and remote, for the patch state."""
import json
import os
from concurrent.futures import ThreadPoolExecutor

import s3fs

MIRROR = "/scratch/jseillade/obs-rebuild/store-v2"
BUCKET_PREFIX = "oceanbench-bucket/dev/observations2024-v2"
ENDPOINT = "https://s3.waw3-1.cloudferro.com"
NEW = "2024-v2.0.1"

fs = s3fs.S3FileSystem(
    key=os.environ["CF_KEY"],
    secret=os.environ["CF_SECRET"],
    client_kwargs={"endpoint_url": ENDPOINT},
    config_kwargs={"s3": {"addressing_style": "path"}, "retries": {"max_attempts": 10}},
)
days = sorted(n[:-5] for n in os.listdir(MIRROR) if n.endswith(".zarr"))


def check(day):
    out = {"day": day}
    with open(os.path.join(MIRROR, f"{day}.zarr/.zattrs")) as fh:
        out["local_zattrs"] = json.load(fh)["obs_basis_version"]
    with open(os.path.join(MIRROR, f"{day}.zarr/.zmetadata")) as fh:
        out["local_zmeta"] = json.load(fh)["metadata"][".zattrs"]["obs_basis_version"]
    with open(os.path.join(MIRROR, f"{day}.manifest.json")) as fh:
        man = json.load(fh)
    out["local_manifest"] = man["obs_basis_version"]
    out["local_patches"] = len([p for p in man.get("patches", []) if p["patch"] == "sla-longitude-normalization"])
    out["local_rows_affected"] = man["patches"][0]["rows_affected"]
    with fs.open(f"{BUCKET_PREFIX}/{day}.zarr/.zattrs") as fh:
        out["remote_zattrs"] = json.load(fh)["obs_basis_version"]
    with fs.open(f"{BUCKET_PREFIX}/{day}.zarr/.zmetadata") as fh:
        out["remote_zmeta"] = json.load(fh)["metadata"][".zattrs"]["obs_basis_version"]
    with fs.open(f"{BUCKET_PREFIX}/{day}.manifest.json") as fh:
        rman = json.load(fh)
    out["remote_manifest"] = rman["obs_basis_version"]
    out["remote_patches"] = len([p for p in rman.get("patches", []) if p["patch"] == "sla-longitude-normalization"])
    out["remote_rows_affected"] = rman["patches"][0]["rows_affected"]
    out["manifests_identical"] = man == rman
    return out


with ThreadPoolExecutor(max_workers=16) as ex:
    rows = list(ex.map(check, days))

bad = [
    r
    for r in rows
    if not (
        r["local_zattrs"] == r["local_zmeta"] == r["local_manifest"] == NEW
        and r["remote_zattrs"] == r["remote_zmeta"] == r["remote_manifest"] == NEW
        and r["local_patches"] == r["remote_patches"] == 1
        and r["manifests_identical"]
        and r["local_rows_affected"] == r["remote_rows_affected"]
    )
]
print(f"days checked {len(rows)}")
print(f"sum rows_affected in manifests {sum(r['local_rows_affected'] for r in rows)}")
print(f"non conforming days {len(bad)}")
for r in bad[:10]:
    print(json.dumps(r, sort_keys=True))
print("SWEEP_PASS" if not bad else "SWEEP_FAIL")
