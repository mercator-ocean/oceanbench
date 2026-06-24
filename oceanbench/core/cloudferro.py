# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

PUBLIC_BUCKET_URL = "s3://oceanbench-bucket/public"

CLOUDFERRO_STORAGE_OPTIONS = {
    "anon": True,
    "client_kwargs": {
        "endpoint_url": "https://s3.waw3-1.cloudferro.com",
    },
}


def cloudferro_public_url(*path_parts: str) -> str:
    return "/".join([PUBLIC_BUCKET_URL, *(path_part.strip("/") for path_part in path_parts)])


def zarr_storage_options(dataset_path: str) -> dict | None:
    if dataset_path.startswith("s3://oceanbench-bucket/"):
        return CLOUDFERRO_STORAGE_OPTIONS
    return None


def zarr_open_kwargs(dataset_path: str) -> dict:
    storage_options = zarr_storage_options(dataset_path)
    return {} if storage_options is None else {"storage_options": storage_options}
