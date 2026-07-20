# SPDX-FileCopyrightText: 2026 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

from datetime import datetime

import numpy
import pandas
import xarray

from oceanbench.core.climate_forecast_standard_names import (
    rename_dataset_with_standard_names,
)
from oceanbench.core.dataset_source import get_dataset_source
from oceanbench.core.dataset_utils import Dimension, Variable
from oceanbench.core.datetime_utils import generate_dates
from oceanbench.core.remote_http import require_remote_dataset_dimensions
from oceanbench.core.resolution import get_dataset_resolution

GLO36V1_BASE_URL = "https://minio.dive.edito.eu/project-moi-glo36-oceanbench/public"
GLONET_HIGH_RESOLUTION_BASE_URL = (
    "https://minio.dive.edito.eu/moiai-octo-bucket/public/octo/v0/ai-gallery/octo-glonet-hr-p1d"
)
GLO36V1_LEAD_DAYS_COUNT = 7
GLO36V1_FIRST_DAY_DATETIMES = generate_dates("2023-01-04", "2024-01-03", 7)
GLO36V1_SUPER_RESOLUTION_SOURCE_NAMES = {
    "glo36v1",
    "glonet_high_resolution",
    "glonet_super_resolution",
}


class Glo36V1ReferenceDataUnavailableError(ValueError):
    pass


def glo36v1_dataset_path(first_day_datetime: datetime | numpy.datetime64) -> str:
    first_day = pandas.Timestamp(first_day_datetime).strftime("%Y%m%d")
    return f"{GLO36V1_BASE_URL}/{first_day}.zarr"


def glonet_high_resolution_dataset_path(
    first_day_datetime: datetime | numpy.datetime64,
) -> str:
    first_day = pandas.Timestamp(first_day_datetime).strftime("%Y%m%d")
    return f"{GLONET_HIGH_RESOLUTION_BASE_URL}/{first_day}.zarr"


def _rename_glo36v1_dimensions(dataset: xarray.Dataset) -> xarray.Dataset:
    return dataset.rename(
        {
            name: standard_name
            for name, standard_name in {"lat": "latitude", "lon": "longitude"}.items()
            if name in dataset
        }
    )


def _assign_standard_name(dataset: xarray.Dataset, name: str, standard_name: str) -> None:
    if name in dataset:
        dataset[name].attrs["standard_name"] = standard_name


def assign_glo36v1_standard_names(dataset: xarray.Dataset) -> xarray.Dataset:
    dataset = _rename_glo36v1_dimensions(dataset)
    standard_names = {
        "zos": Variable.SEA_SURFACE_HEIGHT_ABOVE_GEOID.key(),
        "thetao": Variable.SEA_WATER_POTENTIAL_TEMPERATURE.key(),
        "so": Variable.SEA_WATER_SALINITY.key(),
        "uo": Variable.EASTWARD_SEA_WATER_VELOCITY.key(),
        "vo": Variable.NORTHWARD_SEA_WATER_VELOCITY.key(),
        Dimension.DEPTH.key(): Dimension.DEPTH.key(),
        Dimension.LATITUDE.key(): Dimension.LATITUDE.key(),
        Dimension.LONGITUDE.key(): Dimension.LONGITUDE.key(),
    }
    for name, standard_name in standard_names.items():
        _assign_standard_name(dataset, name, standard_name)
    return dataset


def prepare_glo36v1_week_dataset(
    dataset: xarray.Dataset,
    lead_days_count: int,
    operation_name: str,
    first_day_datetime: datetime | numpy.datetime64 | None = None,
) -> xarray.Dataset:
    if Dimension.LEAD_DAY_INDEX.key() not in dataset.dims and Dimension.TIME.key() in dataset.dims:
        dataset = dataset.rename({Dimension.TIME.key(): Dimension.LEAD_DAY_INDEX.key()})
    if first_day_datetime is not None and Dimension.FIRST_DAY_DATETIME.key() not in dataset.dims:
        dataset = dataset.expand_dims({Dimension.FIRST_DAY_DATETIME.key(): [first_day_datetime]})
    week_dataset = require_remote_dataset_dimensions(
        dataset,
        [Dimension.LEAD_DAY_INDEX.key()],
        operation_name,
    )
    week_dataset = assign_glo36v1_standard_names(week_dataset)
    week_dataset = week_dataset.isel({Dimension.LEAD_DAY_INDEX.key(): slice(0, lead_days_count)})
    week_lead_days_count = week_dataset.sizes[Dimension.LEAD_DAY_INDEX.key()]
    return week_dataset.assign_coords({Dimension.LEAD_DAY_INDEX.key(): range(week_lead_days_count)})


def normalised_first_day(
    first_day_datetime: datetime | numpy.datetime64,
) -> numpy.datetime64:
    return numpy.datetime64(pandas.Timestamp(first_day_datetime).strftime("%Y-%m-%d"))


def available_glo36v1_first_days() -> set[numpy.datetime64]:
    return {normalised_first_day(first_day_datetime) for first_day_datetime in GLO36V1_FIRST_DAY_DATETIMES}


def matching_glo36v1_first_day_datetimes(
    challenger_dataset: xarray.Dataset,
) -> numpy.ndarray:
    first_day_key = Dimension.FIRST_DAY_DATETIME.key()
    requested_first_day_datetimes = challenger_dataset[first_day_key].values
    available_first_days = available_glo36v1_first_days()
    matching_first_day_datetimes = numpy.array(
        [
            first_day_datetime
            for first_day_datetime in requested_first_day_datetimes
            if normalised_first_day(first_day_datetime) in available_first_days
        ]
    )
    if matching_first_day_datetimes.size == 0:
        first_available_day = pandas.Timestamp(GLO36V1_FIRST_DAY_DATETIMES[0]).strftime("%Y-%m-%d")
        last_available_day = pandas.Timestamp(GLO36V1_FIRST_DAY_DATETIMES[-1]).strftime("%Y-%m-%d")
        raise Glo36V1ReferenceDataUnavailableError(
            "GLO36V1 reference scores were not computed for this challenger. "
            f"The GLO36V1 reference is available for weekly first_day_datetime values from "
            f"{first_available_day} to {last_available_day}."
        )
    return matching_first_day_datetimes


def is_super_resolution_dataset(dataset: xarray.Dataset) -> bool:
    dataset_source = get_dataset_source(dataset)
    if dataset_source and dataset_source.name in GLO36V1_SUPER_RESOLUTION_SOURCE_NAMES:
        return True
    try:
        return get_dataset_resolution(rename_dataset_with_standard_names(dataset)) == "thirty_sixth_degree"
    except Exception:
        return False
