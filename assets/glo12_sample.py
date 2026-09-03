# Open GLO12 forecast sample with xarray
from datetime import datetime
import xarray

FORECAST_URL = (
    "https://s3.waw3-1.cloudferro.com/oceanbench-bucket" "/dev/additionnal-data/GLO12/glo12_rg_1d-m_fcst_R20240104.zarr"
)

challenger_dataset: xarray.Dataset = (
    xarray.merge(
        [
            xarray.open_zarr(FORECAST_URL, group=variable_name, consolidated=True)[[variable_name]]
            for variable_name in ["so", "thetao", "uo", "vo", "zos"]
        ]
    )
    .isel(time=slice(0, 10))
    .rename({"time": "lead_day_index"})
    .assign_coords({"lead_day_index": range(10)})
    .expand_dims({"first_day_datetime": [datetime.fromisoformat("2024-01-03")]})
)
challenger_dataset
