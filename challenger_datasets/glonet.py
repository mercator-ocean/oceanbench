# Open GLONET forecasts with xarray
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
    return f"https://minio.dive.edito.eu/project-glonet/public/glonet_refull_2024/{start_datetime_string}.zarr"


start_datetimes: list[datetime] = generate_dates("2024-01-03", "2024-12-25", 7)
challenger_dataset: xarray.Dataset = xarray.open_mfdataset(
    list(map(_dataset_path, start_datetimes)),
    engine="zarr",
    preprocess=lambda dataset: dataset.assign(time=range(10)),
    combine="nested",
    concat_dim="start_datetime",
    parallel=True,
).assign(start_datetime=start_datetimes)

challenger_dataset
