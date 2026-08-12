# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""Copy the observations2024-v2 store from CloudFerro to local scratch."""
import os, sys, time
from concurrent.futures import ThreadPoolExecutor
import s3fs

SRC = "oceanbench-bucket/dev/observations2024-v2"
DST = "/scratch/jseillade/obs-rebuild/store-v2"

fs = s3fs.S3FileSystem(
    key=os.environ["CF_KEY"], secret=os.environ["CF_SECRET"],
    client_kwargs={"endpoint_url": "https://s3.waw3-1.cloudferro.com"},
)
keys = fs.find(SRC)
print(f"listed {len(keys)} objects", flush=True)

def one(k):
    rel = k[len(SRC) + 1:]
    out = os.path.join(DST, rel)
    if os.path.exists(out):
        return 0
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    fs.get_file(k, tmp)
    os.replace(tmp, out)
    return 1

t0 = time.time()
done = 0
with ThreadPoolExecutor(max_workers=16) as ex:
    for i, r in enumerate(ex.map(one, keys)):
        done += r
        if i % 5000 == 0:
            print(f"{i}/{len(keys)} fetched={done} {time.time()-t0:.0f}s", flush=True)
print(f"DONE fetched={done} of {len(keys)} in {time.time()-t0:.0f}s", flush=True)
