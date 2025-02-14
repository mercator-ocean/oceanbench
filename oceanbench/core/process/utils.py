import xarray


def compute_kinetic_energy(dataset: xarray.Dataset):
    KE = 0.5 * (dataset.uo**2 + dataset.vo**2)
    return KE
