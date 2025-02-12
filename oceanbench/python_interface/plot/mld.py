from pathlib import Path

from oceanbench.core.plot.mld_core import plot_mld_core


def plot_mld(mld_datasets_path: Path):
    return plot_mld_core(mld_datasets_path)
