# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

"""
This module exposes the challenger datasets evaluated in the benchmark.
"""

import logging
from datetime import datetime

import xarray
from zarr.errors import GroupNotFoundError

from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.dataset_utils import LEAD_DAYS_COUNT


LOGGER = logging.getLogger(__name__)

_CLOUDFERRO_STORAGE_OPTIONS = {
    "anon": True,
    "client_kwargs": {
        "endpoint_url": "https://s3.waw3-1.cloudferro.com",
    },
}

_GLO12_FORCING_GROUPS = (
    "siconc",
    "sithick",
    "so",
    "thetao",
    "uo",
    "usi",
    "vo",
    "vsi",
    "zos",
)

_IFS_FORCING_ZARR_GROUPS = (
    "cp",
    "ewss",
    "nsss",
    "sohumspe",
    "somslpre",
    "sosnowfa",
    "sosudolw",
    "sosudosw",
    "sotemair",
    "sotemhum",
    "sowaprec",
    "sowinu10",
    "sowinv10",
    "sp",
)


def _weekly_first_day_datetimes_2024() -> list[datetime]:
    return generate_dates("2024-01-03", "2024-12-25", 7)


def _weekly_first_day_datetimes_2023() -> list[datetime]:
    return generate_dates("2023-01-04", "2023-12-27", 7)


def glo12_nowcasts() -> xarray.Dataset:
    first_day_datetimes = _weekly_first_day_datetimes_2024()
    dataset: xarray.Dataset = xarray.open_mfdataset(
        list(map(_glo12_nowcast_dataset_path, first_day_datetimes)),
        engine="zarr",
        parallel=False,
    )
    return dataset


def _glo12_nowcast_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/GLO12_NOWCAST/{start_datetime_string}.zarr"


def ifs_forcings() -> xarray.Dataset:
    first_day_datetimes = _weekly_first_day_datetimes_2024()
    dataset: xarray.Dataset = xarray.open_mfdataset(
        list(map(_ifs_forcing_dataset_path, first_day_datetimes)),
        engine="netcdf4",
        preprocess=lambda dataset: dataset.rename({"time_counter": "lead_day_index"}).assign(
            {"lead_day_index": range(LEAD_DAYS_COUNT)}
        ),
        combine="nested",
        concat_dim="first_day_datetime",
        decode_timedelta=False,
    ).assign({"first_day_datetime": first_day_datetimes})
    return dataset


def _ifs_forcing_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/IFS/IFS_{start_datetime_string}.nc#mode=bytes"


def glo12_forcings() -> xarray.Dataset:
    datasets = _open_available_grouped_zarr_datasets(
        _weekly_first_day_datetimes_2023(),
        _glo12_forcing_dataset_path,
        _GLO12_FORCING_GROUPS,
    )
    return xarray.concat(datasets, dim="time").sortby("time")


def _glo12_forcing_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"s3://oceanbench-bucket/dev/additionnal-data/GLO12/glo12_rg_1d-m_nwct_R{start_datetime_string}.zarr"


def ifs_forcings_zarr() -> xarray.Dataset:
    datasets = _open_available_grouped_zarr_datasets(
        _weekly_first_day_datetimes_2023(),
        _ifs_forcing_zarr_dataset_path,
        _IFS_FORCING_ZARR_GROUPS,
    )
    return (
        xarray.concat(datasets, dim="first_day_datetime")
        .rename({"lat": "latitude", "lon": "longitude"})
        .sortby("first_day_datetime")
    )


def _ifs_forcing_zarr_dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"s3://oceanbench-bucket/dev/additionnal-data/IFS/ifs_forcing_rg_forecasts_R{start_datetime_string}.zarr"


def _open_grouped_zarr_dataset(dataset_path: str, groups: tuple[str, ...]) -> xarray.Dataset:
    return xarray.merge(
        [
            _deduplicate_dataset_indexes(
                xarray.open_dataset(
                    dataset_path,
                    engine="zarr",
                    group=group,
                    storage_options=_CLOUDFERRO_STORAGE_OPTIONS,
                )
            )
            for group in groups
        ],
        compat="override",
        combine_attrs="override",
        join="outer",
    )


def _deduplicate_dataset_indexes(dataset: xarray.Dataset) -> xarray.Dataset:
    for dimension in dataset.dims:
        if dimension in dataset.indexes:
            duplicated_mask = dataset.indexes[dimension].duplicated()
            if duplicated_mask.any():
                dataset = dataset.isel({dimension: ~duplicated_mask})
            dataset = dataset.sortby(dimension)
    return dataset


def _open_available_grouped_zarr_datasets(
    datetimes: list[datetime],
    dataset_path_from_datetime,
    groups: tuple[str, ...],
) -> list[xarray.Dataset]:
    datasets: list[xarray.Dataset] = []
    for dt in datetimes:
        dataset_path = dataset_path_from_datetime(dt)
        try:
            datasets.append(_open_grouped_zarr_dataset(dataset_path, groups))
        except (FileNotFoundError, GroupNotFoundError):
            LOGGER.warning("Skipping unavailable grouped zarr dataset: %s", dataset_path)
    if not datasets:
        raise FileNotFoundError("No grouped zarr datasets could be opened from the configured additional-data paths.")
    return datasets
