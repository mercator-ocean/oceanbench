# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

import xarray


STORAGE_OPTIONS = {
    "anon": True,
    "client_kwargs": {"endpoint_url": "https://s3.waw3-1.cloudferro.com"},
}

TEST_CASES = (
    (
        "glo12_so",
        "s3://oceanbench-bucket/dev/additionnal-data/GLO12/glo12_rg_1d-m_nwct_R20230104.zarr",
        "so",
    ),
    (
        "glo12_thetao",
        "s3://oceanbench-bucket/dev/additionnal-data/GLO12/glo12_rg_1d-m_nwct_R20230111.zarr",
        "thetao",
    ),
    (
        "ifs_sotemair",
        "s3://oceanbench-bucket/dev/additionnal-data/IFS/ifs_forcing_rg_forecasts_R20230103.zarr",
        "sotemair",
    ),
)


def main() -> None:
    for case_name, dataset_url, group_name in TEST_CASES:
        print(f"\n=== {case_name} ===")
        print("url:", dataset_url)
        print("group:", group_name)
        dataset = xarray.open_dataset(
            dataset_url,
            engine="zarr",
            group=group_name,
            storage_options=STORAGE_OPTIONS,
        )
        print(dataset)
        print("sizes:", dict(dataset.sizes))
        print("data_vars:", list(dataset.data_vars))


if __name__ == "__main__":
    main()
