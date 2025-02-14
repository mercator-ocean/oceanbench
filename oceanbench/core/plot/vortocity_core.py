from matplotlib import pyplot
import xarray

from oceanbench.core.process.utils import compute_vorticity_core


def plot_vortocity_core(dataset: xarray.Dataset):
    bb = compute_vorticity_core(dataset)
    bb[0, 0].plot(vmin=-2, vmax=2, cmap="seismic")
    pyplot.show()
