import xarray


def compute_kinetic_energy_core(dataset: xarray.Dataset):
    KE = 0.5 * (dataset.uo**2 + dataset.vo**2)
    return KE


def compute_vorticity_core(dataset: xarray.Dataset) -> xarray.DataArray:
    dvdx = dataset.vo.differentiate("lon")
    dudy = dataset.uo.differentiate("lat")
    vorticity = dvdx - dudy
    return xarray.DataArray(vorticity, dims=dataset.uo.dims, coords=dataset.uo.coords, name="vort")
