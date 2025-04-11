# Open GLONET forecasts with xarray
from functools import reduce
import xarray
from typing import List
from datetime import datetime, timedelta

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


def generate_dates(start_date_str, end_date_str, delta_days):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    return [
        (start_date + timedelta(days=i * delta_days)).strftime("%Y-%m-%d")
        for i in range((end_date - start_date).days // delta_days + 1)
    ]


def _open_dataset(date_string: str) -> xarray.Dataset:
    return _add_climate_forecast_attributes(
        xarray.open_dataset(
            f"https://minio.dive.edito.eu/project-glonet/public/glonet_reforecast_2024/{date_string}.zarr",
            engine="zarr",
        )
    )


challenger_datasets: List[xarray.Dataset] = list(map(_open_dataset, generate_dates("2024-01-03", "2024-07-10", 7)))
