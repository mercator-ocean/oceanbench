import numpy
import xarray


def compute_kinetic_energy_core(dataset: xarray.Dataset):
    KE = 0.5 * (dataset.uo**2 + dataset.vo**2)
    return KE


def compute_vorticity_core(dataset: xarray.Dataset) -> xarray.DataArray:
    dvdx = dataset.vo.differentiate("lon")
    dudy = dataset.uo.differentiate("lat")
    vorticity = dvdx - dudy
    return xarray.DataArray(vorticity, dims=dataset.uo.dims, coords=dataset.uo.coords, name="vort")


def mass_conservation_core(dataset: xarray.Dataset, depth, deg_resolution=0.25):
    R = 6371e3  # earth radius
    deg_to_rad = numpy.pi / 180

    uo = dataset["uo"][:, depth, :, :]
    vo = dataset["vo"][:, depth, :, :]
    lat = dataset["lat"]

    # degree to meter
    dy = deg_resolution * deg_to_rad * R
    dx = deg_resolution * deg_to_rad * R * numpy.cos(numpy.radians(lat))

    # horizontal divergence: du/dx + dv/dy
    div_uo = uo.differentiate("lon") / dx
    div_vo = vo.differentiate("lat") / dy
    divergence = div_uo + div_vo  # Total horizontal divergence
    # if vertical velocity 'wo' exists
    if "wo" in dataset:
        dz = numpy.gradient(dataset.depth)
        dw_dz = numpy.gradient(dataset.wo, axis=-3) / dz
        divergence += dw_dz

    # mean divergence at each time step
    mean_divergence_time_series = divergence.mean(dim=["lon", "lat"])

    return mean_divergence_time_series
