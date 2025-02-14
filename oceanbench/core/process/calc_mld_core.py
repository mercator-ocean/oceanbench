import numpy
import gsw
import xarray


def calc_mld_core(dataset: xarray.Dataset, lead: int) -> xarray.Dataset:
    density_threshold = 0.03  # kg/m^3 threshold for MLD definition
    temperature = dataset["thetao"][lead]
    salinity = dataset["so"][lead]
    depth = dataset["depth"]
    lat = dataset["lat"]
    lon = dataset["lon"]
    absolute_salinity = gsw.SA_from_SP(salinity, depth, lon, lat)
    potential_density = gsw.pot_rho_t_exact(absolute_salinity, temperature, depth, 0)
    surface_density = potential_density.isel(depth=0)
    delta_density = potential_density - surface_density
    mask = delta_density >= density_threshold
    mld_index = mask.argmax(dim="depth")
    mld_depth = depth.isel(depth=mld_index)
    temperature_mask = numpy.isfinite(dataset["thetao"].isel(depth=0))
    dataset["MLD"] = mld_depth
    dataset["MLD"] = dataset["MLD"].where(temperature_mask)

    return dataset
