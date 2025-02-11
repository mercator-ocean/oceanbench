from pathlib import Path

from matplotlib import pyplot
import xarray


def plot_density_core(dataset_path: Path, plot_output_path: Path, show_plot: bool):
    dataarray = xarray.open_dataarray(dataset_path)
    _, ax = pyplot.subplots(1, 2, figsize=(8, 4))
    im = dataarray[:, 0, :].plot(ax=ax[0])
    colorbar = im.colorbar
    colorbar.set_label("vertical section of density $kg/m^{3}$")
    ax[0].invert_yaxis()

    im = dataarray[0].plot(ax=ax[1])

    colorbar = im.colorbar  # TODO: not working
    colorbar.set_label("density GS $kg/m^{3}$")

    pyplot.tight_layout()
    pyplot.savefig(plot_output_path)

    if show_plot:
        pyplot.show()
