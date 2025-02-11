from pathlib import Path

import xarray


def plot_mld_core(dataset_path: Path):
    dataset = xarray.open_dataset(dataset_path)
    dataset.MLD[:, :, 0].plot(vmin=0, vmax=1000)  # TODO: why not ploting?
