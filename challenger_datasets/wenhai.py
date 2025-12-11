# SPDX-FileCopyrightText: 2025 Mercator Ocean International <https://www.mercator-ocean.eu/>
#
# SPDX-License-Identifier: EUPL-1.2

# Open WenHai forecasts with xarray
import xarray
from datetime import datetime, timedelta


def generate_dates(start_date_str, end_date_str, delta_days) -> list[datetime]:
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    return [
        (start_date + timedelta(days=i * delta_days)) for i in range((end_date - start_date).days // delta_days + 1)
    ]


def _dataset_path(start_datetime: datetime) -> str:
    start_datetime_string = start_datetime.strftime("%Y%m%d")
    return f"https://minio.dive.edito.eu/project-oceanbench/public/WENHAI/{start_datetime_string}.zarr"


first_day_datetimes: list[datetime] = generate_dates("2024-01-03", "2024-12-25", 7)
challenger_dataset: xarray.Dataset = xarray.open_mfdataset(
    list(map(_dataset_path, first_day_datetimes)),
    engine="zarr",
    preprocess=lambda dataset: dataset.rename({"time": "lead_day_index"}).assign({"lead_day_index": range(10)}),
    combine="nested",
    concat_dim="first_day_datetime",
    parallel=True,
).assign({"first_day_datetime": first_day_datetimes})

challenger_dataset
