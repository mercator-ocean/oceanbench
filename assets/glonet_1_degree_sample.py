# Open GLONET forecast sample with xarray
from datetime import datetime
import xarray
from oceanbench.core.interpolate import interpolate_1_degree

challenger_dataset: xarray.Dataset = xarray.open_mfdataset(
    [
        "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/ml-forecast-outputs/glonet/20240103.zarr",
        "https://s3.waw3-1.cloudferro.com/oceanbench-bucket/public/ml-forecast-outputs/glonet/20240110.zarr",
    ],
    engine="zarr",
    preprocess=lambda dataset: dataset.rename({"time": "lead_day_index"}).assign({"lead_day_index": range(10)}),
    combine="nested",
    concat_dim="first_day_datetime",
    parallel=True,
).assign(
    {
        "first_day_datetime": [
            datetime.fromisoformat("2024-01-03"),
            datetime.fromisoformat("2024-01-10"),
        ]
    }
)
challenger_dataset = interpolate_1_degree(challenger_dataset)
challenger_dataset
