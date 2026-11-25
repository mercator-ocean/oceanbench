# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
Mixed layer depths computed once per evaluation run.

The challenger mixed layer depth is scored against GLORYS and against the GLO12 analysis, and weekly start dates
with 10 lead days share 3 valid dates with the next start, so without reuse the same fields are computed several
times. Each field is computed once here, written to a local zarr store on disk (not kept in memory), and read back
with the chunks it was computed with, so that the scores do not move.

``OCEANBENCH_MLD_SPILL_DIR`` sets the directory (default: the system temporary directory); ``OCEANBENCH_MLD_REUSE=0``
turns the reuse off. A directory on a memory file system (tmpfs, ramfs) is refused: the mixed layer depths are then
computed as before.
"""

import atexit
import os
import shutil
import tempfile
from collections.abc import Callable
from time import perf_counter

import dask
import dask.array
import numpy
import xarray

from oceanbench.core.dataset_utils import Dimension, Variable

MEMORY_FILE_SYSTEMS = {"tmpfs", "ramfs", "shm"}

_state: dict = {"directory": None, "entries": {}, "counter": 0}


def reuse_enabled() -> bool:
    return os.environ.get("OCEANBENCH_MLD_REUSE", "1") not in ("0", "false", "False", "no")


def _file_system_type(path: str) -> str:
    path = os.path.realpath(path)
    best, best_type = "", "unknown"
    try:
        with open("/proc/mounts", encoding="utf-8") as mounts:
            for line in mounts:
                fields = line.split()
                if len(fields) >= 3:
                    mount_point = fields[1]
                    if (path == mount_point or path.startswith(mount_point.rstrip("/") + "/")) and len(
                        mount_point
                    ) > len(best):
                        best, best_type = mount_point, fields[2]
    except OSError:
        pass
    return best_type


def _spill_directory() -> str | None:
    if _state["directory"] is not None:
        return _state["directory"]
    parent = os.environ.get("OCEANBENCH_MLD_SPILL_DIR") or tempfile.gettempdir()
    os.makedirs(parent, exist_ok=True)
    file_system = _file_system_type(parent)
    free_gb = shutil.disk_usage(parent).free / 1e9
    if file_system in MEMORY_FILE_SYSTEMS:
        print(f"[mld-reuse] {parent} is on {file_system} (memory): mixed layer depths not spilled, reuse off")
        _state["directory"] = ""
        return None
    directory = tempfile.mkdtemp(prefix="oceanbench-mld-", dir=parent)
    atexit.register(shutil.rmtree, directory, True)
    print(f"[mld-reuse] spill directory {directory} on {file_system}, {free_gb:.0f} GB free")
    _state["directory"] = directory
    return directory


def clear() -> None:
    """Forget the stored mixed layer depths of this run and delete their files."""
    directory = _state["directory"]
    _state["entries"].clear()
    _state["directory"] = None
    if directory:
        shutil.rmtree(directory, ignore_errors=True)


def _new_store_path(name: str) -> str:
    _state["counter"] += 1
    return os.path.join(_spill_directory(), f"{_state['counter']:03d}-{name}.zarr")


def _uniform_chunks(array: dask.array.Array) -> bool:
    return all(len(set(axis_chunks[:-1])) <= 1 and axis_chunks[-1] <= axis_chunks[0] for axis_chunks in array.chunks)


def _write_and_read_back(array: dask.array.Array, name: str) -> dask.array.Array:
    path = _new_store_path(name)
    dask.array.to_zarr(array, path, overwrite=True)
    read_back = dask.array.from_zarr(path, chunks=array.chunks)
    assert read_back.chunks == array.chunks and read_back.dtype == array.dtype
    return read_back


def _mld_variable(mld_dataset: xarray.Dataset) -> xarray.DataArray:
    return mld_dataset[Variable.MIXED_LAYER_DEPTH.key()]


def _with_mld_data(mld_dataset: xarray.Dataset, data: dask.array.Array) -> xarray.Dataset:
    key = Variable.MIXED_LAYER_DEPTH.key()
    return mld_dataset.assign({key: mld_dataset[key].copy(data=data)})


def _spillable(mld_dataset: xarray.Dataset) -> bool:
    data = _mld_variable(mld_dataset).data
    return isinstance(data, dask.array.Array) and _uniform_chunks(data) and _spill_directory() is not None


def challenger_mixed_layer_depth(
    key: tuple, challenger_dataset: xarray.Dataset, compute: Callable[[xarray.Dataset], xarray.Dataset]
) -> xarray.Dataset:
    """The challenger mixed layer depth, computed and written once per ``key``, then read back."""
    entry = _state["entries"].get(("challenger",) + key)
    if entry is not None:
        print(f"[mld-reuse] challenger mixed layer depth reused from {entry['path']}")
        return entry["dataset"]
    mld_dataset = compute(challenger_dataset)
    if not reuse_enabled() or not _spillable(mld_dataset):
        return mld_dataset
    start = perf_counter()
    data = _write_and_read_back(_mld_variable(mld_dataset).data, "challenger")
    stored = _with_mld_data(mld_dataset, data)
    _state["entries"][("challenger",) + key] = {"dataset": stored, "path": "challenger", "keep": challenger_dataset}
    print(f"[mld-reuse] challenger mixed layer depth computed and written in {perf_counter() - start:.1f} s")
    return stored


def _valid_date_indices(mld_variable: xarray.DataArray) -> tuple[list[tuple[int, int]], numpy.ndarray]:
    """The (start, lead day) of the first occurrence of each valid date, and the date number of each field."""
    first_days = mld_variable[Dimension.FIRST_DAY_DATETIME.key()].values.astype("datetime64[D]")
    lead_days_count = mld_variable.sizes[Dimension.LEAD_DAY_INDEX.key()]
    date_numbers = numpy.empty((len(first_days), lead_days_count), dtype=int)
    first_fields: list[tuple[int, int]] = []
    seen: dict = {}
    for start_index, first_day in enumerate(first_days):
        for lead_day in range(lead_days_count):
            valid_date = first_day + numpy.timedelta64(lead_day, "D")
            if valid_date not in seen:
                seen[valid_date] = len(first_fields)
                first_fields.append((start_index, lead_day))
            date_numbers[start_index, lead_day] = seen[valid_date]
    return first_fields, date_numbers


def reference_mixed_layer_depth(
    key: tuple, reference_dataset: xarray.Dataset, compute: Callable[[xarray.Dataset], xarray.Dataset], keep=None
) -> xarray.Dataset:
    """
    The reference mixed layer depth, with each valid date computed and written once (a valid date shared by two
    start dates is computed for the first one), then read back with the chunks of the computed fields.
    """
    entry = _state["entries"].get(("reference",) + key)
    if entry is not None:
        print("[mld-reuse] reference mixed layer depth reused")
        return entry["dataset"]
    mld_dataset = compute(reference_dataset)
    if not reuse_enabled() or not _spillable(mld_dataset):
        return mld_dataset
    variable = _mld_variable(mld_dataset)
    start_dimension, lead_dimension = Dimension.FIRST_DAY_DATETIME.key(), Dimension.LEAD_DAY_INDEX.key()
    if start_dimension not in variable.dims or lead_dimension not in variable.dims:
        return mld_dataset
    other_dimensions = [dimension for dimension in variable.dims if dimension not in (start_dimension, lead_dimension)]
    ordered = variable.transpose(start_dimension, lead_dimension, *other_dimensions).data
    if ordered.chunks[0] != (1,) * ordered.shape[0] or ordered.chunks[1] != (1,) * ordered.shape[1]:
        return mld_dataset
    first_fields, date_numbers = _valid_date_indices(variable)
    start = perf_counter()
    by_date = dask.array.stack([ordered[start_index, lead_day] for start_index, lead_day in first_fields])
    stored_by_date = _write_and_read_back(by_date, "reference-by-date")
    fields = dask.array.stack(
        [
            dask.array.stack([stored_by_date[date_numbers[start_index, lead_day]] for lead_day in range(ordered.shape[1])])
            for start_index in range(ordered.shape[0])
        ]
    )
    assert fields.chunks == ordered.chunks
    data = xarray.DataArray(fields, dims=(start_dimension, lead_dimension, *other_dimensions)).transpose(
        *variable.dims
    ).data
    stored = _with_mld_data(mld_dataset, data)
    _state["entries"][("reference",) + key] = {"dataset": stored, "keep": keep}
    print(
        f"[mld-reuse] reference mixed layer depth: {len(first_fields)} valid dates for {date_numbers.size} fields,"
        f" computed and written in {perf_counter() - start:.1f} s"
    )
    return stored
