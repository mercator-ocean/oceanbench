# Open GLONET forecast sample with xarray
from datetime import datetime
import xarray

challenger_dataset: xarray.Dataset = xarray.open_mfdataset(
    [
        "https://minio.dive.edito.eu/project-glonet/public/glonet_refull_2024/20240103.zarr",
        "https://minio.dive.edito.eu/project-glonet/public/glonet_refull_2024/20240110.zarr",
    ],
    engine="zarr",
    preprocess=lambda dataset: dataset.assign(time=range(10)),
    combine="nested",
    concat_dim="start_datetime",
    parallel=True,
).assign(
    start_datetime=[
        datetime.fromisoformat("2024-01-03"),
        datetime.fromisoformat("2024-01-10"),
    ]
)
challenger_dataset
