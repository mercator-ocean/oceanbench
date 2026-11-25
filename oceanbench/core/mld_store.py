# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Persistent store of reference mixed layer depths, one small zarr per valid date.

A reference mixed layer depth only depends on the reference, its grid, the depth levels it is computed on (after
the selection at the challenger depths and the depth cap) and the valid date, so it can be computed once and read
by every later run. The store path is::

    <OCEANBENCH_MLD_STORE>/<reference>/grid-<hash>/levels-<count>-<hash>/v<version>/<YYYY-MM-DD>.zarr

Each zarr holds the mixed layer depth field with its coordinates and the depth levels, and attributes that guard its
use: the store version, a hash of the mixed layer depth code, the gsw and dask versions, and a sha256 of the field
bytes. A date whose entry is missing, whose guard differs from this run or whose checksum fails is computed as before.

``OCEANBENCH_MLD_STORE`` is ``s3://bucket/prefix`` (unset: no store). ``OCEANBENCH_MLD_STORE_WRITE=1`` uploads the
dates this run computed. ``OCEANBENCH_MLD_STORE_VERIFY=N`` recomputes the first N dates read from the store and
fails if they differ.
"""

import hashlib
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
from importlib import metadata
from pathlib import Path

import numpy
import xarray
import zarr

from oceanbench.core import mixed_layer_depth

STORE_VERSION = 1
VARIABLE = "mixed_layer_depth"


def store_url() -> str:
    return os.environ.get("OCEANBENCH_MLD_STORE", "").rstrip("/")


def write_enabled() -> bool:
    return os.environ.get("OCEANBENCH_MLD_STORE_WRITE", "0") in ("1", "true", "True", "yes")


def verify_count() -> int:
    return int(os.environ.get("OCEANBENCH_MLD_STORE_VERIFY", "0") or 0)


def _short_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:12]


def code_hash() -> str:
    source = Path(mixed_layer_depth.__file__).read_bytes()
    return _short_hash(source + f"MAXIMUM_MIXED_LAYER_DEPTH={mixed_layer_depth.MAXIMUM_MIXED_LAYER_DEPTH}".encode())


def guard() -> dict:
    return {
        "mld_store_version": STORE_VERSION,
        "mld_code_hash": code_hash(),
        "gsw_version": metadata.version("gsw"),
        "dask_version": metadata.version("dask"),
    }


def levels(reference_dataset: xarray.Dataset) -> numpy.ndarray:
    """The depth levels the mixed layer depth is computed on, as compute_mixed_layer_depth selects them."""
    capped = mixed_layer_depth._cap_depth(mixed_layer_depth._harmonise_dataset(reference_dataset))
    return numpy.asarray(capped[mixed_layer_depth.Dimension.DEPTH.key()].values, dtype="float64")


def _coordinate_hash(field: xarray.DataArray) -> str:
    parts = []
    for dimension in field.dims:
        values = (
            numpy.asarray(field[dimension].values)
            if dimension in field.coords
            else numpy.arange(field.sizes[dimension])
        )
        parts.append(
            f"{dimension}:{values.size}:".encode() + numpy.ascontiguousarray(values.astype("float64")).tobytes()
        )
    return _short_hash(b"|".join(parts))


def entry_prefix(reference_name: str, field: xarray.DataArray, depth_levels: numpy.ndarray) -> str:
    level_hash = _short_hash(numpy.round(depth_levels, 4).tobytes())
    return f"{reference_name}/grid-{_coordinate_hash(field)}/levels-{depth_levels.size}-{level_hash}/v{STORE_VERSION}"


def data_checksum(values: numpy.ndarray) -> str:
    return hashlib.sha256(numpy.ascontiguousarray(values).tobytes()).hexdigest()


def _split(url: str) -> tuple[str, str]:
    assert url.startswith("s3://"), f"OCEANBENCH_MLD_STORE must be s3://bucket/prefix, not {url}"
    bucket, _, prefix = url[len("s3://") :].partition("/")
    return bucket, prefix


def _client():
    import boto3

    endpoint_url = os.environ.get("BOTO3_ENDPOINT_URL") or f"https://{os.environ['AWS_S3_ENDPOINT']}"
    return boto3.client("s3", endpoint_url=endpoint_url)


def _download(client, bucket: str, key_prefix: str, local: str) -> bool:
    keys = []
    for page in client.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=key_prefix + "/"):
        keys += [item["Key"] for item in page.get("Contents", [])]
    if not keys:
        return False
    for key in keys:
        target = os.path.join(local, key[len(key_prefix) + 1 :])
        os.makedirs(os.path.dirname(target), exist_ok=True)
        client.download_file(bucket, key, target)
    return True


def fetch(entries: dict[str, str], local_directory: str, field_template: xarray.DataArray) -> dict[str, str]:
    """
    Download the store entries (date -> store key prefix) to ``local_directory`` and keep those whose guard matches
    this run and whose data checksum is right. Returns date -> local zarr path.
    """
    bucket, _ = _split(store_url())
    client = _client()
    expected = guard()

    def one(item):
        date, key_prefix = item
        local = os.path.join(local_directory, f"{date}.zarr")
        try:
            if not _download(client, bucket, key_prefix, local):
                return date, None, "missing"
            group = zarr.open_group(local, mode="r")
            attributes = dict(group[VARIABLE].attrs)
            mismatched = [name for name, value in expected.items() if attributes.get(name) != value]
            if mismatched:
                return date, None, "guard " + ",".join(mismatched)
            values = group[VARIABLE][...]
            if (
                values.shape != field_template.shape
                or values.dtype != field_template.dtype
                or data_checksum(values) != attributes.get("sha256")
            ):
                return date, None, "checksum"
            return date, local, "hit"
        except Exception as error:  # a broken entry is recomputed, never used
            return date, None, f"error {type(error).__name__}"

    found, reasons = {}, {}
    with ThreadPoolExecutor(8) as pool:
        for date, local, reason in pool.map(one, entries.items()):
            reasons[reason] = reasons.get(reason, 0) + 1
            if local is None:
                shutil.rmtree(os.path.join(local_directory, f"{date}.zarr"), ignore_errors=True)
            else:
                found[date] = local
    print(f"[mld-store] read {len(entries)} dates: {reasons}")
    return found


def upload(fields: dict[str, tuple[str, xarray.DataArray]], depth_levels: numpy.ndarray, local_directory: str) -> None:
    """Write each date (date -> (store key prefix, field)) as a zarr with its guard and upload it."""
    bucket, _ = _split(store_url())
    client = _client()
    attributes = guard()

    def one(item):
        date, (key_prefix, field) = item
        values = numpy.asarray(field.values)
        local = os.path.join(local_directory, f"upload-{date}.zarr")
        shutil.rmtree(local, ignore_errors=True)
        dataset = xarray.Dataset(
            {VARIABLE: field.copy(data=values).assign_attrs(**attributes, sha256=data_checksum(values), date=date)},
            coords={"depth": ("depth", depth_levels)},
        )
        dataset.to_zarr(local, mode="w", consolidated=False)
        for path in sorted(Path(local).rglob("*")):
            if path.is_file():
                client.upload_file(str(path), bucket, f"{key_prefix}/{path.relative_to(local).as_posix()}")
        shutil.rmtree(local, ignore_errors=True)

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(one, fields.items()))
    print(f"[mld-store] uploaded {len(fields)} dates")


def build(challenger_python_code: str, starts_per_batch: int = 13) -> None:
    """
    Compute and upload the reference mixed layer depths of every valid date of the challenger start dates, for the
    GLORYS reanalysis and the GLO12 analysis, in batches of start dates (dates already in the store are skipped).
    """
    from time import perf_counter

    from oceanbench.core import mld_reuse
    from oceanbench.core.dataset_utils import Dimension
    from oceanbench.core.derived_quantities import compute_mixed_layer_depth
    from oceanbench.core.evaluate import open_challenger_dataset
    from oceanbench.core.references.glo12 import glo12_analysis_dataset
    from oceanbench.core.references.glorys import glorys_reanalysis_dataset
    from oceanbench.core.regions import GLOBAL_REGION_NAME, subset_dataset_to_region

    os.environ["OCEANBENCH_MLD_STORE_WRITE"] = "1"
    challenger_dataset = subset_dataset_to_region(open_challenger_dataset(challenger_python_code), GLOBAL_REGION_NAME)
    start_count = challenger_dataset.sizes[Dimension.FIRST_DAY_DATETIME.key()]
    for reference_dataset_of in (glorys_reanalysis_dataset, glo12_analysis_dataset):
        for first in range(0, start_count, starts_per_batch):
            start = perf_counter()
            batch = challenger_dataset.isel(
                {Dimension.FIRST_DAY_DATETIME.key(): slice(first, first + starts_per_batch)}
            )
            key = ("build", first, reference_dataset_of.__name__)
            mld_reuse.reference_mixed_layer_depth(
                key,
                subset_dataset_to_region(reference_dataset_of(batch), GLOBAL_REGION_NAME),
                compute_mixed_layer_depth,
            )
            if ("reference",) + key not in mld_reuse._state["entries"]:
                raise RuntimeError("reference mixed layer depth was not split by valid date, nothing stored")
            mld_reuse.clear()
            last = first + batch.sizes[Dimension.FIRST_DAY_DATETIME.key()] - 1
            name = reference_dataset_of.__name__
            print(f"[mld-store] build {name} starts {first}..{last} in {perf_counter() - start:.1f} s")
