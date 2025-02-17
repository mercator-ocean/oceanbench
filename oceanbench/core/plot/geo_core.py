from matplotlib import pyplot
import xarray


def plot_geo_core(dataset: xarray.Dataset):
    dataset.u_geo.plot(vmin=-1.5, vmax=1.5)
    pyplot.show()
