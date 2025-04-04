# Open GLONET forecast sample with xarray
import xarray

candidate_datasets = [
    xarray.open_dataset(
        "https://minio.dive.edito.eu/project-glonet/public/glonet_reforecast_2024/2024-01-03.zarr",
        engine="zarr",
    )
]
