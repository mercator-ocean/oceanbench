from pathlib import Path

from oceanbench.core.plot.geo_core import plot_geo_core


def plot_geo(geo_datasets_path: Path):
    return plot_geo_core(geo_datasets_path)
