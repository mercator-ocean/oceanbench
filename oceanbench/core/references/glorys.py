from datetime import datetime, timedelta
from functools import reduce
from typing import List
from xarray import Dataset, open_dataset
import copernicusmarine
import logging

import xarray

logger = logging.getLogger("copernicusmarine")
logger.setLevel(level=logging.WARNING)


def _glorys_subset(start_datetime: datetime) -> Dataset:
    return copernicusmarine.open_dataset(
        dataset_id="cmems_mod_glo_phy_myint_0.083deg_P1D-m",
        dataset_version="202311",
        variables=["thetao", "zos", "uo", "vo", "so"],
        start_datetime=start_datetime,
        end_datetime=start_datetime + timedelta(days=10),
    )


LATITUDE_CLIMATE_FORECAST_ATTRIBUTES = {
    "standard_name": "latitude",
    "long_name": "Latitude",
    "units": "degrees_north",
    "units_long": "Degrees North",
    "axis": "Y",
}

LONGITUDE_CLIMATE_FORECAST_ATTRIBUTES = {
    "standard_name": "longitude",
    "long_name": "Longitude",
    "units": "degrees_east",
    "units_long": "Degrees East",
    "axis": "X",
}

DEPTH_CLIMATE_FORECAST_ATTRIBUTES = {
    "standard_name": "depth",
    "long_name": "Depth",
    "units": "m",
    "units_long": "Meters",
}

TIME_CLIMATE_FORECAST_ATTRIBUTES = {
    "standard_name": "time",
    "long_name": "Time",
    "axis": "T",
}


def _update_variable_attributes(
    dataset: xarray.Dataset,
    variable_name_and_attributes: tuple[str, dict[str, str]],
) -> xarray.Dataset:
    variable_name, attributes = variable_name_and_attributes
    dataset[variable_name].attrs = attributes
    return dataset


def _add_climate_forecast_attributes(
    dataset: xarray.Dataset,
) -> xarray.Dataset:
    return reduce(
        _update_variable_attributes,
        zip(
            ["lat", "lon", "depth", "time"],
            [
                LATITUDE_CLIMATE_FORECAST_ATTRIBUTES,
                LONGITUDE_CLIMATE_FORECAST_ATTRIBUTES,
                DEPTH_CLIMATE_FORECAST_ATTRIBUTES,
                TIME_CLIMATE_FORECAST_ATTRIBUTES,
            ],
        ),
        dataset,
    )


def _fix_zos_depth_dimension(dataset: xarray.Dataset) -> xarray.Dataset:
    dataset["zos"] = dataset.zos.isel(depth=0)
    return dataset


def _to_1_4(glorys_dataset: Dataset) -> Dataset:
    initial_datetime = datetime.fromisoformat(str(glorys_dataset["time"][0].values)) - timedelta(days=1)
    initial_datetime_string = initial_datetime.strftime("%Y-%m-%d")
    return _fix_zos_depth_dimension(
        _add_climate_forecast_attributes(
            open_dataset(
                f"https://minio.dive.edito.eu/project-oceanbench/public/glorys14/{initial_datetime_string}.zarr",
                engine="zarr",
            )
        )
    )


def _glorys_datasets(challenger_dataset: Dataset) -> Dataset:
    start_datetime = datetime.fromisoformat(str(challenger_dataset["time"][0].values))
    return _to_1_4(_glorys_subset(start_datetime))


def glorys_datasets(challenger_datasets: List[Dataset]) -> List[Dataset]:
    return list(map(_glorys_datasets, challenger_datasets))
