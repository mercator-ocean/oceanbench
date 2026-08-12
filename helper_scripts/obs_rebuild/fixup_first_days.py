# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Backfill rows_affected_non_sla on the days patched before the field existed.

Those days had zero non-SLA rows changed, which the state record proves by its
before and after non-SLA digests being equal.
"""
import json
import os
import sys

import s3fs

MIRROR = "/scratch/jseillade/obs-rebuild/store-v2"
BUCKET_PREFIX = "oceanbench-bucket/dev/observations2024-v2"
ENDPOINT = "https://s3.waw3-1.cloudferro.com"
STATE_PATH = "/scratch/jseillade/obs-rebuild/patch-lon-state.json"
PATCH_NAME = "sla-longitude-normalization"

fs = s3fs.S3FileSystem(
    key=os.environ["CF_KEY"],
    secret=os.environ["CF_SECRET"],
    client_kwargs={"endpoint_url": ENDPOINT},
    config_kwargs={"s3": {"addressing_style": "path"}, "retries": {"max_attempts": 10}},
)
with open(STATE_PATH) as fh:
    state = json.load(fh)

for day in sys.argv[1:]:
    rec = state[day]
    assert rec["status"] == "ok", (day, rec["status"])
    assert rec["non_sla_lon_sha256_before"] == rec["non_sla_lon_sha256_after"], day
    rec["n_non_sla_rows_changed"] = 0
    rec["max_shift"] = 360.0
    mpath = os.path.join(MIRROR, f"{day}.manifest.json")
    with open(mpath) as fh:
        manifest = json.load(fh)
    for entry in manifest["patches"]:
        if entry["patch"] == PATCH_NAME:
            entry["rows_affected_non_sla"] = 0
            entry["description"] = (
                "longitude normalised to [-180, 180); every affected row moved by "
                "exactly 360 degrees, effectively all obs_type 4, plus any in-situ "
                "row sitting exactly on 180.0 which becomes -180.0, the same point; "
                "no other column, row identity or row count changed"
            )
    with open(mpath, "w") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    fs.put_file(mpath, f"{BUCKET_PREFIX}/{day}.manifest.json")
    print(f"{day} manifest backfilled and uploaded", flush=True)

tmp = STATE_PATH + ".tmp"
with open(tmp, "w") as fh:
    json.dump(state, fh, indent=1, sort_keys=True)
os.replace(tmp, STATE_PATH)
print("state updated")
