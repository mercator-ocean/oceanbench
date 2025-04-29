# Open GLONET forecast sample with xarray
import xarray
from typing import List

challenger_datasets: List[xarray.Dataset] = [
    xarray.open_dataset(
        "https://minio.dive.edito.eu/project-glonet/public/glonet_full_2024/20240103.zarr",
        engine="zarr",
    ),
    xarray.open_dataset(
        "https://minio.dive.edito.eu/project-glonet/public/glonet_full_2024/20240110.zarr",
        engine="zarr",
    ),
]
