from matplotlib import pyplot
import xarray

from oceanbench.core.process.utils import compute_kinetic_energy


def plot_kinetic_energy_core(dataset: xarray.Dataset):
    aa = compute_kinetic_energy(dataset)
    aa[0, 0].plot(cmap="seismic")
    pyplot.show()
