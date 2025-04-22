# Open GLONET forecasts with xarray
import xarray
from typing import List
from datetime import datetime, timedelta


def generate_dates(start_date_str, end_date_str, delta_days):
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    return [
        (start_date + timedelta(days=i * delta_days)).strftime("%Y%m%d")
        for i in range((end_date - start_date).days // delta_days + 1)
    ]


def _open_dataset(date_string: str) -> xarray.Dataset:
    return xarray.open_dataset(
        f"https://minio.dive.edito.eu/project-glonet/public/glonet_full_2024/{date_string}.zarr",
        engine="zarr",
    )


challenger_datasets: List[xarray.Dataset] = list(map(_open_dataset, generate_dates("2024-01-03", "2024-07-03", 7)))
