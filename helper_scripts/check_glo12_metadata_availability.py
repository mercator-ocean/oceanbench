# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime, timedelta

import s3fs


ENDPOINT_URL = "https://s3.waw3-1.cloudferro.com"
BUCKET_PREFIX = "oceanbench-bucket/dev/additionnal-data/GLO12"
START_DATE = datetime(2023, 1, 4)
END_DATE = datetime(2023, 12, 27)
STEP = timedelta(days=7)
METADATA_FILES = (".zgroup", ".zmetadata", "zarr.json")


def _dataset_prefix(date_value: datetime) -> str:
    return f"{BUCKET_PREFIX}/glo12_rg_1d-m_nwct_R{date_value.strftime('%Y%m%d')}.zarr"


def main() -> None:
    filesystem = s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"endpoint_url": ENDPOINT_URL},
    )

    available_dates: list[str] = []
    missing_dates: list[str] = []

    current_date = START_DATE
    while current_date <= END_DATE:
        dataset_prefix = _dataset_prefix(current_date)
        has_metadata = False

        for metadata_file in METADATA_FILES:
            if filesystem.exists(f"{dataset_prefix}/{metadata_file}"):
                has_metadata = True
                break

        if has_metadata:
            available_dates.append(current_date.strftime("%Y-%m-%d"))
        else:
            missing_dates.append(current_date.strftime("%Y-%m-%d"))

        current_date += STEP

    print("Available dates:")
    for date_str in available_dates:
        print(date_str)

    print("\nMissing dates:")
    for date_str in missing_dates:
        print(date_str)

    print(f"\nSummary: available={len(available_dates)} missing={len(missing_dates)}")


if __name__ == "__main__":
    main()
