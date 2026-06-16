# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

from collections.abc import Callable
from datetime import datetime
from functools import reduce

import xarray
import zarr

from oceanbench.core.datetime_utils import generate_dates


_CLOUDFERRO_STORAGE_OPTIONS = {
    "anon": True,
    "client_kwargs": {
        "endpoint_url": "https://s3.waw3-1.cloudferro.com",
    },
}


def _glo12_nowcast_datetimes() -> list[datetime]:
    return generate_dates("2023-01-04", "2025-12-31", 7)


def _ifs_forcing_datetimes() -> list[datetime]:
    return generate_dates("2023-01-03", "2025-12-30", 7)


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
    groups = tuple(zarr.open_group(dataset_path, mode="r", storage_options=_CLOUDFERRO_STORAGE_OPTIONS).group_keys())
    datasets = [_open_zarr_group_dataset(dataset_path, group) for group in groups]
    return xarray.merge(datasets, compat="override", combine_attrs="override", join="override")


def _open_zarr_group_dataset(dataset_path: str, group: str) -> xarray.Dataset:
    dataset = xarray.open_dataset(
        dataset_path,
        engine="zarr",
        group=group,
        storage_options=_CLOUDFERRO_STORAGE_OPTIONS,
        chunks={},
    )
    return _deduplicate_indexed_dimensions(dataset)


def _deduplicate_indexed_dimensions(dataset: xarray.Dataset) -> xarray.Dataset:
    return reduce(_deduplicate_dimension_index, dataset.indexes, dataset)


def _deduplicate_dimension_index(dataset: xarray.Dataset, dimension: str) -> xarray.Dataset:
    index = dataset.indexes[dimension]
    return dataset.isel({dimension: ~index.duplicated()}) if index.has_duplicates else dataset
