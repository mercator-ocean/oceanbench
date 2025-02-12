import xarray


def plot_mld_core(dataset: xarray.Dataset):
    dataset.MLD[:, :, 0].plot(vmin=0, vmax=1000)  # TODO: why not ploting?
