import gsw
import xarray


def calc_density_core(
    dataset: xarray.Dataset,
    lead: int,
    minimum_latitude: float,
    maximum_latitude: float,
    minimum_longitude: float,
    maximum_longitude: float,
) -> xarray.Dataset:
    ds = dataset.isel(
        lat=(dataset["lat"] > minimum_latitude) & (dataset["lat"] < maximum_latitude),
        lon=(dataset["lon"] > minimum_longitude) & (dataset["lon"] < maximum_longitude),
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

    return density
