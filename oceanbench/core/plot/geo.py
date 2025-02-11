from pathlib import Path

import xarray


def plot_geo_core(dataset_path: Path):
    dataset = xarray.open_dataset(dataset_path)
    dataset.u_geo.plot(vmin=-1.5, vmax=1.5)  # TODO: why not ploting?
