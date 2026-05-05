# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import re
from collections import defaultdict

import s3fs


ENDPOINT_URL = "https://s3.waw3-1.cloudferro.com"
BUCKET = "oceanbench-bucket"
PREFIXES = {
    "GLO12": "dev/additionnal-data/GLO12",
    "IFS": "dev/additionnal-data/IFS",
}


def _extract_year(dataset_name: str) -> str | None:
    match = re.search(r"R(\d{4})\d{4}\.zarr$", dataset_name)
    if match is None:
        return None
    return match.group(1)


def main() -> None:
    filesystem = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": ENDPOINT_URL},
    )

    for dataset_kind, prefix in PREFIXES.items():
        print(f"\n=== {dataset_kind} ===")
        object_paths = filesystem.find(f"{BUCKET}/{prefix}", withdirs=False)
        dataset_paths = sorted(path for path in object_paths if path.endswith(".zarr/.zgroup"))
        dataset_roots = [path.removesuffix("/.zgroup") for path in dataset_paths]

        if not dataset_roots:
            print("No datasets found.")
            continue

        counts_by_year: dict[str, int] = defaultdict(int)
        for dataset_root in dataset_roots:
            year = _extract_year(dataset_root)
            if year is not None:
                counts_by_year[year] += 1

        print(f"total datasets: {len(dataset_roots)}")
        print("years:", dict(sorted(counts_by_year.items())))
        print("first datasets:")
        for dataset_root in dataset_roots[:10]:
            print(dataset_root)
        if len(dataset_roots) > 10:
            print("last datasets:")
            for dataset_root in dataset_roots[-10:]:
                print(dataset_root)


if __name__ == "__main__":
    main()
