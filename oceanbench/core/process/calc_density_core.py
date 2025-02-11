from pathlib import Path
import gsw
import xarray


def calc_density_core(dataset_path: Path, lead: int, lat: float, lon: float, output_path: Path):
    dataset = xarray.open_dataset(dataset_path)
    ds = dataset.isel(
        lat=(dataset["lat"] > lat[0]) & (dataset["lat"] < lat[1]),
        lon=(dataset["lon"] > lon[0]) & (dataset["lon"] < lon[1]),
    )
    temperature = ds["thetao"][lead]
    salinity = ds["so"][lead]
    latitude = ds["lat"]
    longitude = ds["lon"]

    if "depth" in ds:
        depth_negative = -ds["depth"]  # Depth should be negative
        pressure = xarray.apply_ufunc(
            gsw.p_from_z,
            depth_negative,
            latitude,
            dask="parallelized",
            vectorize=True,
        )
    else:
        pressure = ds["pressure"]

    # Calc absolute salinity and conservative temperature
    absolute_salinity = xarray.apply_ufunc(
        gsw.SA_from_SP,
        salinity,
        pressure,
        longitude,
        latitude,
        dask="parallelized",
        vectorize=True,
    )
    conservative_temperature = xarray.apply_ufunc(
        gsw.CT_from_pt,
        absolute_salinity,
        temperature,
        dask="parallelized",
        vectorize=True,
    )

    # Calc density
    density = xarray.apply_ufunc(
        gsw.rho,
        absolute_salinity,
        conservative_temperature,
        pressure,
        dask="parallelized",
        vectorize=True,
    )

    density.to_netcdf(output_path)
    return density
