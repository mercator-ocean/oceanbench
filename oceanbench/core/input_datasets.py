# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

from collections.abc import Callable
from datetime import datetime
from functools import reduce

import numpy
import xarray
import zarr

from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.remote_http import open_remote_zarr, remote_zarr_store


_CLOUDFERRO_STORAGE_OPTIONS = {
    "anon": True,
    "client_kwargs": {
        "endpoint_url": "https://s3.waw3-1.cloudferro.com",
    },
}

_IFS_NOWCAST_DATASET_ROOT = "s3://oceanbench-bucket/public/ifs-nowcasts24"

_IFS_NOWCAST_ANALYSIS_VARIABLES = (
    "sotemair",
    "sotemhum",
    "sohumspe",
    "sowinu10",
    "sowinv10",
    "skt",
    "somslpre",
    "sp",
)

_IFS_NOWCAST_ACCUMULATED_VARIABLES = (
    "sosudosw",
    "sosudolw",
    "sowaprec",
    "cp",
    "sosnowfa",
    "ewss",
    "nsss",
)


def _glo12_nowcast_datetimes() -> list[datetime]:
    return generate_dates("2023-01-04", "2025-12-31", 7)


def _ifs_forcing_datetimes() -> list[datetime]:
    return generate_dates("2023-01-03", "2025-12-30", 7)


def _ifs_nowcast_datetimes() -> list[datetime]:
    return generate_dates("2024-01-02", "2024-12-24", 7)


def glo12_nowcasts() -> xarray.Dataset:
    return _open_weekly_zarr_datasets(
        _glo12_nowcast_datetimes(),
        _glo12_nowcast_dataset_path,
        concat_dim="time",
    )


def _glo12_nowcast_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"s3://oceanbench-bucket/dev/additionnal-data/GLO12/glo12_rg_1d-m_nwct_R{start_datetime_string}.zarr"


def ifs_forcings() -> xarray.Dataset:
    return _open_weekly_zarr_datasets(
        _ifs_forcing_datetimes(),
        _ifs_forcing_dataset_path,
        concat_dim="first_day_datetime",
    )


def _ifs_forcing_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"s3://oceanbench-bucket/dev/additionnal-data/IFS/ifs_forcing_rg_forecasts_R{start_datetime_string}.zarr"


def ifs_nowcasts() -> xarray.Dataset:
    datasets = [_open_ifs_nowcast_dataset(_ifs_nowcast_dataset_path(dt)) for dt in _ifs_nowcast_datetimes()]
    return xarray.concat(
        datasets,
        dim="time",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        join="exact",
        combine_attrs="override",
    ).sortby("time")


def _ifs_nowcast_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"{_IFS_NOWCAST_DATASET_ROOT}/{start_datetime_string}.zarr"


def _open_ifs_nowcast_dataset(dataset_path: str) -> xarray.Dataset:
    dataset = open_remote_zarr(
        dataset_path,
        _CLOUDFERRO_STORAGE_OPTIONS,
        consolidated=False,
        chunks={},
    )
    dataset = _deduplicate_indexed_dimensions(dataset)
    return _validate_ifs_nowcast_dataset(_repair_ifs_nowcast_longitude(dataset))


def _repair_ifs_nowcast_longitude(dataset: xarray.Dataset) -> xarray.Dataset:
    longitude = dataset["lon"]
    missing = longitude.isnull()
    if not missing.any().item():
        return dataset
    if missing.sum().item() != 1 or not missing.isel(lon=0).item():
        raise ValueError("IFS nowcast longitude contains unexpected missing coordinates")

    # The published F1280 stores contain one missing first longitude value;
    # reconstruct the regular coordinate only, leaving all data variables untouched.
    repaired_longitude = xarray.DataArray(
        numpy.linspace(0.0, longitude.isel(lon=-1).item(), longitude.size),
        dims=("lon",),
        name="lon",
        attrs=longitude.attrs,
    )

    return dataset.assign_coords(lon=repaired_longitude)


def _validate_ifs_nowcast_dataset(dataset: xarray.Dataset) -> xarray.Dataset:
    expected_dimensions = {"time", "lat", "lon"}
    if set(dataset.dims) != expected_dimensions:
        raise ValueError(
            f"IFS nowcast datasets must have dimensions {sorted(expected_dimensions)}, " f"got {sorted(dataset.dims)}"
        )

    expected_variables = set(_IFS_NOWCAST_ANALYSIS_VARIABLES + _IFS_NOWCAST_ACCUMULATED_VARIABLES)
    missing_variables = expected_variables - set(dataset.data_vars)
    unexpected_variables = set(dataset.data_vars) - expected_variables
    if missing_variables or unexpected_variables:
        raise ValueError(
            "IFS nowcast datasets must contain the expected 15 variables; "
            f"missing={sorted(missing_variables)}, unexpected={sorted(unexpected_variables)}"
        )

    expected_spatial_sizes = {"lat": 2560, "lon": 5120}
    wrong_spatial_sizes = {
        dimension: dataset.sizes[dimension]
        for dimension, expected_size in expected_spatial_sizes.items()
        if dataset.sizes[dimension] != expected_size
    }
    if wrong_spatial_sizes:
        raise ValueError(f"IFS nowcast datasets have unexpected spatial sizes: {wrong_spatial_sizes}")

    wrong_variable_dimensions = {
        variable: dataset[variable].dims
        for variable in expected_variables
        if dataset[variable].dims != ("time", "lat", "lon")
    }
    if wrong_variable_dimensions:
        raise ValueError(f"IFS nowcast variables must have dimensions (time, lat, lon): {wrong_variable_dimensions}")

    if not dataset["lat"].notnull().all().item() or not dataset["lon"].notnull().all().item():
        raise ValueError("IFS nowcast latitude and longitude coordinates must be finite")
    if not dataset.indexes["lat"].is_monotonic_decreasing or not dataset.indexes["lon"].is_monotonic_increasing:
        raise ValueError("IFS nowcast latitude must decrease and longitude must increase")
    if dataset.sizes["time"] != 5:
        raise ValueError(f"IFS nowcast datasets must contain 5 time steps, got {dataset.sizes['time']}")

    # The final timestamp is intentionally NaN for accumulated variables because
    # there is no following six-hour interval. Do not fill or transform it here.
    return dataset


def _open_weekly_zarr_datasets(
    datetimes: list[datetime],
    dataset_path_from_datetime: Callable[[datetime], str],
    concat_dim: str,
    rename: dict[str, str] | None = None,
) -> xarray.Dataset:
    datasets = [_open_grouped_zarr_dataset(dataset_path_from_datetime(dt)) for dt in datetimes]
    dataset = xarray.concat(datasets, dim=concat_dim, combine_attrs="override").sortby(concat_dim)
    return dataset.rename(rename) if rename is not None else dataset


def _open_grouped_zarr_dataset(dataset_path: str) -> xarray.Dataset:
    groups = tuple(zarr.open_group(remote_zarr_store(dataset_path, _CLOUDFERRO_STORAGE_OPTIONS), mode="r").group_keys())
    datasets = [_open_zarr_group_dataset(dataset_path, group) for group in groups]
    return xarray.merge(datasets, compat="override", combine_attrs="override", join="override")


def _open_zarr_group_dataset(dataset_path: str, group: str) -> xarray.Dataset:
    dataset = open_remote_zarr(
        dataset_path,
        _CLOUDFERRO_STORAGE_OPTIONS,
        group=group,
        chunks={},
    )
    return _deduplicate_indexed_dimensions(dataset)


def _deduplicate_indexed_dimensions(dataset: xarray.Dataset) -> xarray.Dataset:
    return reduce(_deduplicate_dimension_index, dataset.indexes, dataset)


def _deduplicate_dimension_index(dataset: xarray.Dataset, dimension: str) -> xarray.Dataset:
    index = dataset.indexes[dimension]
    return dataset.isel({dimension: ~index.duplicated()}) if index.has_duplicates else dataset
