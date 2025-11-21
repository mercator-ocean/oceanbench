import xarray as xr
from datetime import datetime

# 1. on ouvre le sample
ds = xr.open_mfdataset(
    ["https://minio.dive.edito.eu/project-oceanbench/public/XIHE/20240103.zarr"],
    engine="zarr",
    preprocess=lambda d: d.rename({"time": "lead_day_index"}).assign({"lead_day_index": range(10)}),
    combine="nested",
    concat_dim="first_day_datetime",
    parallel=True,
)

# 2. on remplace la dimension 'first_day_datetime' par une vraie coordonnée dimensionnelle
ds = ds.assign_coords(first_day_datetime=("first_day_datetime", [datetime(2024, 1, 3)]))

challenger_dataset = ds
print(challenger_dataset)
