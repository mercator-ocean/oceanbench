from pathlib import Path
import numpy
import gsw
import xarray


def calc_mld_core(glonet_path: Path, lead: int, output_path: Path) -> xarray.Dataset:
    glonet_dataset = xarray.open_dataset(glonet_path)
    density_threshold = 0.03  # kg/m^3 threshold for MLD definition
    temperature = glonet_dataset["thetao"][lead]
    salinity = glonet_dataset["so"][lead]
    depth = glonet_dataset["depth"]
    lat = glonet_dataset["lat"]
    lon = glonet_dataset["lon"]
    absolute_salinity = gsw.SA_from_SP(salinity, depth, lon, lat)
    # conservative_temperature = gsw.CT_from_t(
    #     absolute_salinity, temperature, depth
    # )
    potential_density = gsw.pot_rho_t_exact(absolute_salinity, temperature, depth, 0)
    surface_density = potential_density.isel(depth=0)
    delta_density = potential_density - surface_density
    mask = delta_density >= density_threshold
    mld_index = mask.argmax(dim="depth")
    mld_depth = depth.isel(depth=mld_index)
    temperature_mask = numpy.isfinite(glonet_dataset["thetao"].isel(depth=0))
    glonet_dataset["MLD"] = mld_depth
    glonet_dataset["MLD"] = glonet_dataset["MLD"].where(temperature_mask)

    glonet_dataset.to_netcdf(output_path)
    return glonet_dataset
